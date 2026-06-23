"""ML ONNX feature boundary contract test.

Schema-level contract between ffreis-ml-crypto-env (preprocessing) and
ffreis-rust-onnx-model-serving (inference).  The test builds the preprocessing
graph via the real ``build_onnx_graph`` helper from ``crypto_env``, runs ORT
inference on it with known inputs, and asserts:

1. Input names and dtypes match the ONNX-serving contract.
2. Output shape and dtype are exactly ``float32[10]`` (the ``N_MARKET_FEATURES``
   boundary that serving expects as model input).
3. Output values are finite (no NaN/inf from the ONNX graph).

These are schema-level assertions — they catch shape/dtype drift at import time,
without needing a live inference server.

The test skips gracefully when ``ffreis-ml-crypto-env`` is not installed.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Optional imports — skip if sibling package not installed
# ---------------------------------------------------------------------------

try:
    import onnx  # type: ignore[import-untyped]
    import onnxruntime as ort  # type: ignore[import-untyped]

    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

try:
    from crypto_env.core.features import (
        N_CHANNELS,
        N_MARKET_FEATURES,
        ONNX_OPSET,
        build_onnx_graph,
        fit_preprocessor,
    )

    _CRYPTO_ENV_AVAILABLE = True
except ImportError:
    _CRYPTO_ENV_AVAILABLE = False

pytestmark = pytest.mark.integration

_SKIP_REASON = []
if not _ORT_AVAILABLE:
    _SKIP_REASON.append("onnxruntime not installed")
if not _CRYPTO_ENV_AVAILABLE:
    _SKIP_REASON.append(
        "ffreis-ml-crypto-env not installed (uv pip install -e ../ffreis-ml-crypto-env)"
    )

_SKIP = bool(_SKIP_REASON)

# Known stable inputs for deterministic contract assertions
_OHLCV_T = np.array([101.0, 102.0, 99.0, 100.0, 5000.0], dtype=np.float32)
_OHLCV_PREV = np.array([100.0, 101.0, 98.0, 99.0, 4800.0], dtype=np.float32)


def _fit_dummy_stats() -> tuple[np.ndarray, np.ndarray]:
    """Fit mu/sigma on synthetic data so the ONNX graph is fully initialised."""
    rng = np.random.default_rng(42)
    n = 200
    base = rng.uniform(90.0, 110.0, n).astype(np.float32)
    return fit_preprocessor(
        market_open=base,
        market_high=base + rng.uniform(0, 3, n).astype(np.float32),
        market_low=base - rng.uniform(0, 3, n).astype(np.float32),
        market_close=base + rng.uniform(-1, 1, n).astype(np.float32),
        market_volume=rng.uniform(3000.0, 8000.0, n).astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_SKIP, reason="; ".join(_SKIP_REASON) if _SKIP_REASON else "")
def test_preprocessing_onnx_input_names_and_dtypes() -> None:
    """ONNX graph must expose exactly two float32 inputs: ohlcv_t and ohlcv_prev."""
    mu, sigma = _fit_dummy_stats()
    model = build_onnx_graph(mu, sigma)
    onnx.checker.check_model(model)

    import onnx as _onnx

    graph = model.graph
    # graph.input includes both runtime inputs AND initializers (baked-in stats)
    # so filter to the runtime inputs only (the ones without initializers).
    initializer_names = {init.name for init in graph.initializer}
    runtime_inputs = {inp.name: inp for inp in graph.input if inp.name not in initializer_names}

    assert "ohlcv_t" in runtime_inputs, f"Missing input 'ohlcv_t'; found: {set(runtime_inputs)}"
    assert "ohlcv_prev" in runtime_inputs, (
        f"Missing input 'ohlcv_prev'; found: {set(runtime_inputs)}"
    )

    float32_type = _onnx.TensorProto.FLOAT
    for name, info in runtime_inputs.items():
        elem_type = info.type.tensor_type.elem_type
        assert elem_type == float32_type, (
            f"Input '{name}' has elem_type={elem_type}, expected float32 ({float32_type})"
        )


@pytest.mark.skipif(_SKIP, reason="; ".join(_SKIP_REASON) if _SKIP_REASON else "")
def test_preprocessing_onnx_output_shape_and_dtype() -> None:
    """ONNX output 'features' must be float32[10] — the inference boundary width."""
    mu, sigma = _fit_dummy_stats()
    model = build_onnx_graph(mu, sigma)
    onnx.checker.check_model(model)

    with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
        onnx.save(model, f.name)
        sess = ort.InferenceSession(f.name)

    out = sess.run(["features"], {"ohlcv_t": _OHLCV_T, "ohlcv_prev": _OHLCV_PREV})[0]

    assert out.dtype == np.float32, f"Output dtype is {out.dtype}, expected float32"
    assert out.reshape(-1).shape == (N_MARKET_FEATURES,), (
        f"Output shape is {out.shape}, expected ({N_MARKET_FEATURES},)"
    )


@pytest.mark.skipif(_SKIP, reason="; ".join(_SKIP_REASON) if _SKIP_REASON else "")
def test_preprocessing_onnx_output_is_finite() -> None:
    """ORT inference on known inputs must produce finite values (no NaN/inf)."""
    mu, sigma = _fit_dummy_stats()
    model = build_onnx_graph(mu, sigma)

    with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
        onnx.save(model, f.name)
        sess = ort.InferenceSession(f.name)

    out = sess.run(["features"], {"ohlcv_t": _OHLCV_T, "ohlcv_prev": _OHLCV_PREV})[0]
    assert np.all(np.isfinite(out)), f"Output contains non-finite values: {out}"


@pytest.mark.skipif(_SKIP, reason="; ".join(_SKIP_REASON) if _SKIP_REASON else "")
def test_preprocessing_onnx_zero_prev_is_finite() -> None:
    """Episode-start bars (ohlcv_prev == 0) must not produce NaN or inf.

    This is a regression guard for the EPS floor in the ONNX Clip node: a zero
    denominator in the log-return Div would produce ±inf without it.
    """
    mu, sigma = _fit_dummy_stats()
    model = build_onnx_graph(mu, sigma)

    with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
        onnx.save(model, f.name)
        sess = ort.InferenceSession(f.name)

    out = sess.run(
        ["features"],
        {"ohlcv_t": _OHLCV_T, "ohlcv_prev": np.zeros(N_CHANNELS, dtype=np.float32)},
    )[0]
    assert np.all(np.isfinite(out)), f"Output non-finite on zero prev: {out}"


@pytest.mark.skipif(_SKIP, reason="; ".join(_SKIP_REASON) if _SKIP_REASON else "")
def test_preprocessing_onnx_opset_matches_serving_contract() -> None:
    """The ONNX graph's opset must match the expected serving opset (17).

    ffreis-rust-onnx-model-serving uses ort 1.18+ which supports opset ≤ 20;
    asserting opset == 17 keeps the boundary stable as the graph evolves.
    """
    mu, sigma = _fit_dummy_stats()
    model = build_onnx_graph(mu, sigma)

    opset_versions = {op.domain: op.version for op in model.opset_import}
    default_opset = opset_versions.get("", -1)
    assert default_opset == ONNX_OPSET, (
        f"ONNX graph uses opset {default_opset}, expected {ONNX_OPSET} "
        "(update serving contract if intentionally bumped)"
    )


@pytest.mark.skipif(_SKIP, reason="; ".join(_SKIP_REASON) if _SKIP_REASON else "")
def test_preprocessing_onnx_output_dim_matches_n_market_features() -> None:
    """Constant N_MARKET_FEATURES must equal the ONNX graph output dimension.

    This ensures that the Python constant (used by policy networks as the feature
    count) stays in sync with the serialised ONNX graph shape.
    """
    mu, sigma = _fit_dummy_stats()
    model = build_onnx_graph(mu, sigma)
    onnx.checker.check_model(model)

    graph = model.graph
    output_infos = {out.name: out for out in graph.output}
    assert "features" in output_infos, f"Missing output 'features'; found: {set(output_infos)}"

    shape = output_infos["features"].type.tensor_type.shape
    dims = [d.dim_value for d in shape.dim]
    # Dim 0 is the feature axis for the rank-1 (non-batched) graph.
    assert dims[0] == N_MARKET_FEATURES, (
        f"ONNX output dim is {dims[0]}, but N_MARKET_FEATURES == {N_MARKET_FEATURES}"
    )
