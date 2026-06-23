"""Registry → serving HTTP parity test.

Registers a tiny ONNX model in SqliteModelRegistry, resolves it, then starts
ffreis-python-model-serving pointed at the resolved artifact path and asserts
that HTTP /invocations predictions match a local ORT inference session.

Requires:
  - ffreis-ml-registry installed (``uv sync --extra registry``)
  - ffreis-python-model-serving present at the well-known sibling path

The test skips gracefully when either dependency is absent.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import httpx
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import pytest

try:
    import onnxruntime as ort  # type: ignore[import-untyped]

    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

try:
    from ml_registry.adapters.sqlite_registry import SqliteModelRegistry
    from ml_registry.core.entities import ModelVersion, Stage

    _REGISTRY_AVAILABLE = True
except ImportError:
    _REGISTRY_AVAILABLE = False

_SERVING_DIR = Path(__file__).resolve().parents[3] / "ffreis-python-model-serving"
_SERVING_AVAILABLE = (_SERVING_DIR / "src" / "serving.py").exists()

_SKIP_REASON = []
if not _ORT_AVAILABLE:
    _SKIP_REASON.append("onnxruntime not installed")
if not _REGISTRY_AVAILABLE:
    _SKIP_REASON.append("ffreis-ml-registry not installed (uv sync --extra registry)")
if not _SERVING_AVAILABLE:
    _SKIP_REASON.append(f"ffreis-python-model-serving not found at {_SERVING_DIR}")

pytestmark = pytest.mark.skipif(
    bool(_SKIP_REASON),
    reason="; ".join(_SKIP_REASON) if _SKIP_REASON else "",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free_port() -> int:
    """Return an available TCP port on 127.0.0.1."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_linear_onnx(output_path: Path, *, n_features: int = 4) -> None:
    """Write a tiny float32 linear model: x[batch, n_features] → y[batch, 1]."""
    float32 = onnx.TensorProto.FLOAT
    inp = oh.make_tensor_value_info("input", float32, ["batch", n_features])
    out = oh.make_tensor_value_info("output", float32, ["batch", 1])

    rng = np.random.default_rng(7)
    w = rng.standard_normal((n_features, 1)).astype(np.float32)
    b = np.array([0.5], dtype=np.float32)
    w_init = onh.from_array(w, name="W")
    b_init = onh.from_array(b, name="b")

    matmul = oh.make_node("MatMul", inputs=["input", "W"], outputs=["logits"])
    add = oh.make_node("Add", inputs=["logits", "b"], outputs=["output"])
    graph = oh.make_graph(
        [matmul, add],
        "linear",
        [inp],
        [out],
        initializer=[w_init, b_init],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))


def _wait_ready(host: str, port: int, *, timeout: float = 30.0, interval: float = 0.5) -> bool:
    """Poll GET /healthz until 200 or timeout. Returns True if ready."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"http://{host}:{port}/healthz", timeout=2.0)
            if resp.status_code == 200:
                return True
        except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
            pass
        time.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.timeout(90)
def test_registry_resolve_then_serve_parity(tmp_path: Path) -> None:
    """Register a tiny ONNX → resolve → serve → assert HTTP prediction ≈ ORT."""
    n_features = 4
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    onnx_path = model_dir / "model.onnx"
    _make_linear_onnx(onnx_path, n_features=n_features)

    # --- Register in SqliteModelRegistry ---
    db_path = str(tmp_path / "registry.db")
    registry = SqliteModelRegistry(path=db_path)
    version = ModelVersion(
        id="integration-hub-parity-test:1",
        name="integration-hub-parity-test",
        model_id="INTEGRATION_HUB_PARITY",
        version="1",
        onnx_uri=onnx_path.resolve().as_uri(),
        stage=Stage.PRODUCTION,
    )
    registry.register(version)

    # --- Resolve: confirm onnx_uri round-trips correctly ---
    resolved = registry.resolve(name="integration-hub-parity-test", stage=Stage.PRODUCTION)
    assert resolved.onnx_uri is not None, "resolved.onnx_uri must not be None"
    # Strip file:// to get the local path the serving needs
    resolved_path = Path(resolved.onnx_uri.removeprefix("file://"))
    assert resolved_path.exists(), f"resolved path does not exist: {resolved_path}"
    registry.close()

    # --- Start serving with SM_MODEL_DIR pointed at the model dir ---
    port = _free_port()
    env = {
        **os.environ,
        "SM_MODEL_DIR": str(model_dir),
        "MODEL_FILENAME": "model.onnx",
        "PORT": str(port),
        "GUNICORN_WORKERS": "1",
        "GUNICORN_THREADS": "2",
        "PROMETHEUS_ENABLED": "False",
        "OTEL_ENABLED": "False",
        "SWAGGER_ENABLED": "False",
        "LOG_LEVEL": "WARNING",
    }
    proc = subprocess.Popen(
        [
            "uv",
            "run",
            "uvicorn",
            "--app-dir",
            "src",
            "serving:application",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(_SERVING_DIR),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    try:
        ready = _wait_ready("127.0.0.1", port, timeout=30.0)
        if not ready:
            pytest.skip("python-model-serving did not start within 30 s (dependency missing?)")

        # --- Local ORT inference ---
        rng = np.random.default_rng(99)
        batch_input = rng.random((3, n_features)).astype(np.float32)
        sess = ort.InferenceSession(str(onnx_path))
        ort_output = sess.run(None, {"input": batch_input})[0]  # [3, 1]

        # --- HTTP /invocations ---
        payload = {"instances": batch_input.tolist()}
        resp = httpx.post(
            f"http://127.0.0.1:{port}/invocations",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10.0,
        )
        assert resp.status_code == 200, f"serving returned {resp.status_code}: {resp.text}"

        body = resp.json()
        assert "predictions" in body, f"missing 'predictions' key in response: {body}"
        http_output = np.array(body["predictions"], dtype=np.float32)  # [3, 1]

        np.testing.assert_allclose(
            ort_output,
            http_output,
            atol=1e-4,
            err_msg="HTTP /invocations output differs from local ORT inference",
        )
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_registry_resolution_roundtrip(tmp_path: Path) -> None:
    """SqliteModelRegistry register → resolve returns the correct onnx_uri (no serving needed)."""
    if not _REGISTRY_AVAILABLE:
        pytest.skip("ffreis-ml-registry not installed")

    db_path = str(tmp_path / "registry.db")
    onnx_path = tmp_path / "dummy.onnx"
    onnx_path.write_bytes(b"DUMMY")  # not a real model; only testing URI round-trip

    registry = SqliteModelRegistry(path=db_path)
    uri = onnx_path.resolve().as_uri()
    version = ModelVersion(
        id="roundtrip-test:1",
        name="roundtrip-test",
        model_id="ROUNDTRIP",
        version="1",
        onnx_uri=uri,
        stage=Stage.PRODUCTION,
    )
    registry.register(version)

    resolved = registry.resolve(name="roundtrip-test", stage=Stage.PRODUCTION)
    assert resolved.onnx_uri == uri
    assert resolved.name == "roundtrip-test"
    assert resolved.version == "1"
    registry.close()
