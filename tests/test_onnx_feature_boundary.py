"""Runtime ONNX preprocessing↔model boundary tests.

Builds toy ONNX graphs that mirror the crypto-rl boundary contract
(preprocessing: 3-input concat → observation; model: observation → allocation)
and verifies:

1. Static boundary assertions (names ⊆, dtype parity, shape compat, opset order).
2. Runtime parity: chained ORT sessions vs merged single-session agree within atol=1e-4.
3. Regression: a dtype mismatch in the boundary is detected and reported.

Also loads real crypto-rl export artifacts when they exist at the well-known sibling
path and runs the full boundary + parity check against them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from check_feature_contract import check_onnx_boundary

try:
    import onnxruntime as ort  # type: ignore[import-untyped]

    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

# Paths to real crypto-rl ONNX artifacts; populated by a C1 training run.
_CRYPTO_RL_ROOT = Path(__file__).resolve().parents[3] / "ffreis-ml-crypto-rl"
_CRYPTO_RL_PREPROC = _CRYPTO_RL_ROOT / "artifacts" / "preprocessing.onnx"
_CRYPTO_RL_POLICY = _CRYPTO_RL_ROOT / "artifacts" / "policy.onnx"
_CRYPTO_RL_MERGED = _CRYPTO_RL_ROOT / "artifacts" / "full_pipeline.onnx"

# Boundary spec for crypto-rl (must stay in sync with onnx_export.py constants)
_OPSET = 17
_OBS_DIM = 11
_N_CHANNELS = 5  # ohlcv_t and ohlcv_prev width


# ---------------------------------------------------------------------------
# Toy model builders
# ---------------------------------------------------------------------------


def _make_preprocessing_onnx(output_path: Path, *, opset: int = _OPSET) -> None:
    """Build a toy preprocessing graph: concat(ohlcv_t, ohlcv_prev, position_fraction) → observation.

    Matches the crypto-rl boundary spec:
      ohlcv_t         [batch, 5]  float32
      ohlcv_prev      [batch, 5]  float32
      position_fraction [batch, 1] float32
      → observation   [batch, 11] float32
    """
    float32 = onnx.TensorProto.FLOAT
    batch = "batch"

    ohlcv_t = oh.make_tensor_value_info("ohlcv_t", float32, [batch, _N_CHANNELS])
    ohlcv_prev = oh.make_tensor_value_info("ohlcv_prev", float32, [batch, _N_CHANNELS])
    position_fraction = oh.make_tensor_value_info("position_fraction", float32, [batch, 1])
    observation = oh.make_tensor_value_info("observation", float32, [batch, _OBS_DIM])

    concat_node = oh.make_node(
        "Concat",
        inputs=["ohlcv_t", "ohlcv_prev", "position_fraction"],
        outputs=["observation"],
        axis=1,
    )

    graph = oh.make_graph(
        [concat_node],
        "preprocessing",
        [ohlcv_t, ohlcv_prev, position_fraction],
        [observation],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", opset)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))


def _make_policy_onnx(output_path: Path, *, opset: int = _OPSET) -> None:
    """Build a toy policy graph: observation[batch, 11] → allocation[batch, 1] (matmul+bias).

    Weight W [11, 1] and bias b [1] are embedded as constants.
    """
    float32 = onnx.TensorProto.FLOAT
    batch = "batch"

    observation = oh.make_tensor_value_info("observation", float32, [batch, _OBS_DIM])
    allocation = oh.make_tensor_value_info("allocation", float32, [batch, 1])

    rng = np.random.default_rng(42)
    w_np = rng.standard_normal((_OBS_DIM, 1)).astype(np.float32)
    b_np = np.zeros((1,), dtype=np.float32)

    w_init = onh.from_array(w_np, name="W")
    b_init = onh.from_array(b_np, name="b")

    matmul_node = oh.make_node("MatMul", inputs=["observation", "W"], outputs=["logits"])
    add_node = oh.make_node("Add", inputs=["logits", "b"], outputs=["allocation"])

    graph = oh.make_graph(
        [matmul_node, add_node],
        "policy",
        [observation],
        [allocation],
        initializer=[w_init, b_init],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", opset)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))


def _make_mismatched_policy_onnx(
    output_path: Path, *, dtype: int = onnx.TensorProto.FLOAT16
) -> None:
    """Build a policy graph with a dtype-mismatched 'observation' input."""
    batch = "batch"
    observation = oh.make_tensor_value_info("observation", dtype, [batch, _OBS_DIM])
    allocation = oh.make_tensor_value_info("allocation", dtype, [batch, 1])

    w_init = onh.from_array(np.ones((_OBS_DIM, 1), dtype=np.float16), name="W")
    b_init = onh.from_array(np.zeros((1,), dtype=np.float16), name="b")

    matmul_node = oh.make_node("MatMul", inputs=["observation", "W"], outputs=["logits"])
    add_node = oh.make_node("Add", inputs=["logits", "b"], outputs=["allocation"])

    graph = oh.make_graph(
        [matmul_node, add_node],
        "policy_mismatched",
        [observation],
        [allocation],
        initializer=[w_init, b_init],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _merge_for_test(preprocessing_path: Path, policy_path: Path, merged_path: Path) -> None:
    """Merge two ONNX graphs with onnx.compose, aligning IR versions first."""
    preproc = onnx.load(str(preprocessing_path))
    policy = onnx.load(str(policy_path))
    if preproc.ir_version != policy.ir_version:
        preproc.ir_version = policy.ir_version
        onnx.save(preproc, str(preprocessing_path))
        preproc = onnx.load(str(preprocessing_path))

    merged = onnx.compose.merge_models(preproc, policy, io_map=[("observation", "observation")])
    onnx.checker.check_model(merged)
    onnx.save(merged, str(merged_path))


def _random_inputs(batch: int = 4, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "ohlcv_t": rng.random((batch, _N_CHANNELS)).astype(np.float32),
        "ohlcv_prev": rng.random((batch, _N_CHANNELS)).astype(np.float32),
        "position_fraction": rng.random((batch, 1)).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Static boundary tests (no ORT required)
# ---------------------------------------------------------------------------


class TestStaticBoundary:
    """check_onnx_boundary static assertions on toy models."""

    def test_clean_boundary_passes(self, tmp_path: Path) -> None:
        preprocessing = tmp_path / "preprocessing.onnx"
        policy = tmp_path / "policy.onnx"
        _make_preprocessing_onnx(preprocessing)
        _make_policy_onnx(policy)

        violations = check_onnx_boundary(str(preprocessing), str(policy))
        assert violations == [], f"unexpected violations: {violations}"

    def test_dtype_mismatch_is_detected(self, tmp_path: Path) -> None:
        preprocessing = tmp_path / "preprocessing.onnx"
        policy = tmp_path / "policy_bad.onnx"
        _make_preprocessing_onnx(preprocessing)
        _make_mismatched_policy_onnx(policy)

        violations = check_onnx_boundary(str(preprocessing), str(policy))
        assert any("dtype mismatch" in v for v in violations), (
            f"dtype mismatch not detected; got: {violations}"
        )

    def test_output_names_subset_check(self, tmp_path: Path) -> None:
        """Policy with different input name → name-subset violation."""
        preprocessing = tmp_path / "preprocessing.onnx"
        _make_preprocessing_onnx(preprocessing)

        # Build policy that expects 'features' not 'observation'
        float32 = onnx.TensorProto.FLOAT
        inp = oh.make_tensor_value_info("features", float32, ["batch", _OBS_DIM])
        out = oh.make_tensor_value_info("allocation", float32, ["batch", 1])
        w_init = onh.from_array(np.ones((_OBS_DIM, 1), dtype=np.float32), name="W")
        b_init = onh.from_array(np.zeros((1,), dtype=np.float32), name="b")
        matmul = oh.make_node("MatMul", inputs=["features", "W"], outputs=["logits"])
        add = oh.make_node("Add", inputs=["logits", "b"], outputs=["allocation"])
        graph = oh.make_graph(
            [matmul, add], "bad_policy", [inp], [out], initializer=[w_init, b_init]
        )
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", _OPSET)])
        model.ir_version = 8
        onnx.checker.check_model(model)
        policy_bad = tmp_path / "policy_wrongname.onnx"
        onnx.save(model, str(policy_bad))

        violations = check_onnx_boundary(str(preprocessing), str(policy_bad))
        assert any("not found in model inputs" in v for v in violations), (
            f"name-subset violation not detected; got: {violations}"
        )

    def test_missing_file_reported(self, tmp_path: Path) -> None:
        violations = check_onnx_boundary(
            str(tmp_path / "missing.onnx"),
            str(tmp_path / "also_missing.onnx"),
        )
        assert len(violations) == 2
        assert all("not found" in v.lower() or "File" in v for v in violations)


# ---------------------------------------------------------------------------
# Runtime parity tests (require onnxruntime)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _ORT_AVAILABLE, reason="onnxruntime not installed")
class TestRuntimeParity:
    """Merged single-session output ≈ chained two-session output within atol=1e-4."""

    def test_merged_vs_chained_toy_models(self, tmp_path: Path) -> None:
        preprocessing = tmp_path / "preprocessing.onnx"
        policy = tmp_path / "policy.onnx"
        merged = tmp_path / "full_pipeline.onnx"

        _make_preprocessing_onnx(preprocessing)
        _make_policy_onnx(policy)
        _merge_for_test(preprocessing, policy, merged)

        inputs = _random_inputs(batch=8)

        # Chained: preprocessing → policy
        sess_pre = ort.InferenceSession(str(preprocessing))
        obs = sess_pre.run(None, inputs)[0]
        sess_pol = ort.InferenceSession(str(policy))
        chained_out = sess_pol.run(None, {"observation": obs})[0]

        # Merged: single session
        sess_merged = ort.InferenceSession(str(merged))
        merged_out = sess_merged.run(None, inputs)[0]

        np.testing.assert_allclose(
            chained_out,
            merged_out,
            atol=1e-4,
            err_msg="merged output differs from chained output beyond atol=1e-4",
        )

    def test_merged_opset_compatible_with_both(self, tmp_path: Path) -> None:
        preprocessing = tmp_path / "preprocessing.onnx"
        policy = tmp_path / "policy.onnx"
        merged = tmp_path / "full_pipeline.onnx"

        _make_preprocessing_onnx(preprocessing)
        _make_policy_onnx(policy)
        _merge_for_test(preprocessing, policy, merged)

        merged_model = onnx.load(str(merged))
        pre_model = onnx.load(str(preprocessing))
        pol_model = onnx.load(str(policy))

        def _opset(m: onnx.ModelProto) -> int:
            for op in m.opset_import:
                if op.domain in ("", "ai.onnx"):
                    return int(op.version)
            return 0

        assert _opset(merged_model) >= _opset(pre_model)
        assert _opset(merged_model) >= _opset(pol_model)

    def test_merged_output_has_no_nan_inf(self, tmp_path: Path) -> None:
        preprocessing = tmp_path / "preprocessing.onnx"
        policy = tmp_path / "policy.onnx"
        merged = tmp_path / "full_pipeline.onnx"

        _make_preprocessing_onnx(preprocessing)
        _make_policy_onnx(policy)
        _merge_for_test(preprocessing, policy, merged)

        sess = ort.InferenceSession(str(merged))
        for seed in range(5):
            inputs = _random_inputs(batch=4, seed=seed)
            out = sess.run(None, inputs)[0]
            assert np.all(np.isfinite(out)), f"NaN/inf in merged output (seed={seed})"


# ---------------------------------------------------------------------------
# Real crypto-rl artifact tests (skipped when artifacts not present)
# ---------------------------------------------------------------------------


_REAL_ARTIFACTS_PRESENT = (
    _ORT_AVAILABLE and _CRYPTO_RL_PREPROC.exists() and _CRYPTO_RL_POLICY.exists()
)


@pytest.mark.skipif(not _REAL_ARTIFACTS_PRESENT, reason="crypto-rl ONNX artifacts not present")
class TestRealCryptoRlBoundary:
    """Boundary and parity tests against the real crypto-rl export artifacts."""

    def test_static_boundary_passes(self) -> None:
        violations = check_onnx_boundary(str(_CRYPTO_RL_PREPROC), str(_CRYPTO_RL_POLICY))
        assert violations == [], f"crypto-rl boundary violations: {violations}"

    def test_output_names_subset(self) -> None:
        preproc = onnx.load(str(_CRYPTO_RL_PREPROC))
        policy = onnx.load(str(_CRYPTO_RL_POLICY))
        preproc_out_names = {o.name for o in preproc.graph.output}
        policy_in_names = {i.name for i in policy.graph.input}
        assert preproc_out_names <= policy_in_names, (
            f"preprocessing outputs {preproc_out_names} not all in policy inputs {policy_in_names}"
        )

    def test_runtime_parity_real_artifacts(self, tmp_path: Path) -> None:
        preproc_copy = tmp_path / "preprocessing.onnx"
        import shutil

        shutil.copy(_CRYPTO_RL_PREPROC, preproc_copy)

        if _CRYPTO_RL_MERGED.exists():
            # C1 done: use the pre-merged artifact
            sess_merged = ort.InferenceSession(str(_CRYPTO_RL_MERGED))
        else:
            # C1 not done yet: build merged on-the-fly for this test
            merged_path = tmp_path / "full_pipeline.onnx"
            _merge_for_test(preproc_copy, _CRYPTO_RL_POLICY, merged_path)
            sess_merged = ort.InferenceSession(str(merged_path))

        inputs = _random_inputs(batch=4)
        sess_pre = ort.InferenceSession(str(_CRYPTO_RL_PREPROC))
        obs = sess_pre.run(None, inputs)[0]
        sess_pol = ort.InferenceSession(str(_CRYPTO_RL_POLICY))
        chained_out = sess_pol.run(None, {"observation": obs})[0]
        merged_out = sess_merged.run(None, inputs)[0]

        np.testing.assert_allclose(
            chained_out,
            merged_out,
            atol=1e-4,
            err_msg="real crypto-rl: merged output differs from chained output",
        )
