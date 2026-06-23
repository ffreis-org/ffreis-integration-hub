"""Registry → serve handoff contract test.

Confirms that the train → register → serve pipeline works end-to-end at the
contract level, without spinning up a live HTTP server:

1. Register a dummy ONNX artifact via ``SqliteModelRegistry``.
2. Resolve it back by name using the same registry.
3. Assert the artifact exists on disk at the returned ``onnx_uri`` path.
4. Assert the artifact is valid ONNX (parseable by ``onnx.load``).

This is a schema-level contract test — it proves that the registry round-trip
is intact and the file pointer resolves to a loadable model, which is the
contract ffreis-rust-onnx-model-serving (and ffreis-python-model-serving) rely
on: the serving layer receives an ``onnx_uri`` from the registry and must be
able to load it.

The test skips gracefully when ``ffreis-ml-registry`` is not installed.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Optional imports — skip if sibling package not installed
# ---------------------------------------------------------------------------

try:
    import onnx  # type: ignore[import-untyped]
    import onnx.helper as oh  # type: ignore[import-untyped]
    import onnx.numpy_helper as onh  # type: ignore[import-untyped]

    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False

try:
    from ml_registry.adapters.sqlite_registry import SqliteModelRegistry
    from ml_registry.core.entities import ModelVersion, Stage

    _REGISTRY_AVAILABLE = True
except ImportError:
    _REGISTRY_AVAILABLE = False

pytestmark = pytest.mark.integration

_SKIP_REASON = []
if not _ONNX_AVAILABLE:
    _SKIP_REASON.append("onnx not installed")
if not _REGISTRY_AVAILABLE:
    _SKIP_REASON.append(
        "ffreis-ml-registry not installed (uv pip install -e ../ffreis-ml-registry)"
    )

_SKIP = bool(_SKIP_REASON)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dummy_onnx(output_path: str, *, n_features: int = 10) -> None:
    """Write a minimal float32[batch, n_features] → float32[batch, 1] ONNX model.

    The shape mirrors the crypto-env preprocessing output (10 features) as model
    input, which is the exact contract ffreis-rust-onnx-model-serving expects.
    """
    float32 = onnx.TensorProto.FLOAT
    inp = oh.make_tensor_value_info("input", float32, ["batch", n_features])
    out = oh.make_tensor_value_info("output", float32, ["batch", 1])

    rng = np.random.default_rng(0)
    weights = rng.standard_normal((n_features, 1)).astype(np.float32)
    bias = np.zeros(1, dtype=np.float32)

    w_init = onh.from_array(weights, name="W")
    b_init = onh.from_array(bias, name="b")

    matmul = oh.make_node("MatMul", inputs=["input", "W"], outputs=["linear"])
    add = oh.make_node("Add", inputs=["linear", "b"], outputs=["output"])

    graph = oh.make_graph(
        [matmul, add],
        "dummy_model",
        inputs=[inp],
        outputs=[out],
        initializer=[w_init, b_init],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    onnx.save(model, output_path)


def _make_model_version(name: str, version: str, onnx_uri: str) -> ModelVersion:
    """Construct a ``ModelVersion`` with a unique id and the given artifact URI."""
    return ModelVersion(
        id=f"{name}:{version}:{uuid.uuid4().hex[:8]}",
        name=name,
        model_id=name.upper(),
        version=version,
        onnx_uri=onnx_uri,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_SKIP, reason="; ".join(_SKIP_REASON) if _SKIP_REASON else "")
def test_register_resolve_artifact_exists_and_is_valid_onnx() -> None:
    """Core contract: register → resolve → artifact path exists and is valid ONNX.

    This is the minimal end-to-end handoff the serving layer depends on.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = str(Path(tmpdir) / "preprocessing.onnx")
        _make_dummy_onnx(onnx_path)

        registry = SqliteModelRegistry(path=":memory:")
        mv = _make_model_version("crypto_preprocessing", "1", onnx_uri=f"file://{onnx_path}")
        registry.register(mv)

        resolved = registry.resolve("crypto_preprocessing", version="1")

        # The resolved onnx_uri must point to a real file.
        assert resolved.onnx_uri is not None, "Resolved model has no onnx_uri"
        uri = resolved.onnx_uri
        assert uri.startswith("file://"), f"Expected file:// URI, got: {uri!r}"

        artifact_path = Path(uri.removeprefix("file://"))
        assert artifact_path.exists(), f"Artifact not found at {artifact_path}"
        assert artifact_path.stat().st_size > 0, f"Artifact is empty at {artifact_path}"

        # The artifact must be parseable as ONNX (valid model bytes).
        loaded = onnx.load(str(artifact_path))
        onnx.checker.check_model(loaded)


@pytest.mark.skipif(_SKIP, reason="; ".join(_SKIP_REASON) if _SKIP_REASON else "")
def test_register_preserves_onnx_uri_and_stage() -> None:
    """The registry must round-trip onnx_uri and default stage through register/get."""
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = str(Path(tmpdir) / "model.onnx")
        _make_dummy_onnx(onnx_path)

        registry = SqliteModelRegistry(path=":memory:")
        mv = _make_model_version("policy", "2", onnx_uri=f"file://{onnx_path}")
        registry.register(mv)

        got = registry.get_version("policy", "2")
        assert got is not None, "get_version returned None after register"
        assert got.onnx_uri == f"file://{onnx_path}", (
            f"onnx_uri mismatch: {got.onnx_uri!r} != file://{onnx_path!r}"
        )
        assert got.stage is Stage.NONE, f"Expected stage NONE, got {got.stage}"


@pytest.mark.skipif(_SKIP, reason="; ".join(_SKIP_REASON) if _SKIP_REASON else "")
def test_register_multiple_versions_resolve_by_version() -> None:
    """Multiple registered versions must be independently resolvable by version string."""
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = {}
        for v in ("1", "2"):
            p = str(Path(tmpdir) / f"model_v{v}.onnx")
            _make_dummy_onnx(p)
            paths[v] = p

        registry = SqliteModelRegistry(path=":memory:")
        for v, p in paths.items():
            registry.register(_make_model_version("policy", v, onnx_uri=f"file://{p}"))

        for v, p in paths.items():
            resolved = registry.resolve("policy", version=v)
            assert resolved.onnx_uri == f"file://{p}", (
                f"Version {v}: onnx_uri {resolved.onnx_uri!r} != file://{p!r}"
            )
            artifact = Path(p)
            assert artifact.exists(), f"Artifact for version {v} missing at {artifact}"


@pytest.mark.skipif(_SKIP, reason="; ".join(_SKIP_REASON) if _SKIP_REASON else "")
def test_registered_artifact_has_correct_n_features_input_shape() -> None:
    """The artifact registered must expose the 10-feature input shape the serving layer expects.

    This links the registry contract to the ONNX feature boundary: the serving
    layer receives whatever onnx_uri the registry returns and loads it directly.
    If the feature count drifts, ORT will raise a shape mismatch at inference
    time — catching it here at contract level is cheaper.
    """
    n_features = 10  # N_MARKET_FEATURES from crypto-env
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = str(Path(tmpdir) / "preprocessing.onnx")
        _make_dummy_onnx(onnx_path, n_features=n_features)

        registry = SqliteModelRegistry(path=":memory:")
        registry.register(
            _make_model_version("crypto_preprocessing", "1", onnx_uri=f"file://{onnx_path}")
        )

        resolved = registry.resolve("crypto_preprocessing", version="1")
        assert resolved.onnx_uri is not None
        artifact_path = Path(resolved.onnx_uri.removeprefix("file://"))
        loaded = onnx.load(str(artifact_path))

        # The first runtime input must have feature dimension == n_features.
        initializer_names = {init.name for init in loaded.graph.initializer}
        runtime_inputs = [inp for inp in loaded.graph.input if inp.name not in initializer_names]
        assert runtime_inputs, "No runtime inputs in resolved ONNX artifact"
        shape = runtime_inputs[0].type.tensor_type.shape
        feature_dim = shape.dim[1].dim_value  # dim[0] is batch
        assert feature_dim == n_features, (
            f"Artifact feature dim is {feature_dim}, expected {n_features} "
            "(drift between registry artifact and serving contract)"
        )
