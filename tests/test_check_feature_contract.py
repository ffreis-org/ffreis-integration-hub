"""Tests for check_feature_contract.py.

Uses synthetic ONNX models (built with onnx.helper) to verify that
boundary violations are correctly detected and clean boundaries pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import pytest

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from check_feature_contract import (
    _dtype_name,
    _extract_tensor_info,
    _get_opset,
    _shape_compatible,
    check_onnx_boundary,
    main,
)


# ---------------------------------------------------------------------------
# Helpers: build tiny synthetic ONNX models
# ---------------------------------------------------------------------------


def _make_identity_model(
    input_name: str,
    output_name: str,
    dtype: int = onnx.TensorProto.FLOAT,
    shape: list[int] = None,
    opset: int = 17,
) -> onnx.ModelProto:
    """Build a minimal ONNX model: output = Identity(input)."""
    if shape is None:
        shape = [10]
    input_info = oh.make_tensor_value_info(input_name, dtype, shape)
    output_info = oh.make_tensor_value_info(output_name, dtype, shape)
    node = oh.make_node("Identity", inputs=[input_name], outputs=[output_name])
    graph = oh.make_graph([node], "test_graph", [input_info], [output_info])
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", opset)])
    onnx.checker.check_model(model)
    return model


def _make_preprocessing_model(
    output_name: str = "features",
    dtype: int = onnx.TensorProto.FLOAT,
    shape: list[int] = None,
    opset: int = 17,
) -> onnx.ModelProto:
    """Build a preprocessing ONNX model with a single output 'features'."""
    if shape is None:
        shape = [10]
    return _make_identity_model("raw_input", output_name, dtype=dtype, shape=shape, opset=opset)


def _make_policy_model(
    input_name: str = "features",
    dtype: int = onnx.TensorProto.FLOAT,
    shape: list[int] = None,
    opset: int = 17,
) -> onnx.ModelProto:
    """Build a policy ONNX model with a single input 'features'."""
    if shape is None:
        shape = [10]
    return _make_identity_model(input_name, "action", dtype=dtype, shape=shape, opset=opset)


def _save(model: onnx.ModelProto, path: Path) -> None:
    onnx.save(model, str(path))


# ---------------------------------------------------------------------------
# Tests: _shape_compatible
# ---------------------------------------------------------------------------


class TestShapeCompatible:
    def test_equal_shapes(self) -> None:
        assert _shape_compatible([10], [10]) is True

    def test_flexible_neg1_in_preproc(self) -> None:
        assert _shape_compatible([-1, 10], [5, 10]) is True

    def test_flexible_neg1_in_model(self) -> None:
        assert _shape_compatible([5, 10], [-1, 10]) is True

    def test_symbolic_dim(self) -> None:
        assert _shape_compatible(["batch", 10], [5, 10]) is True

    def test_mismatched_rank(self) -> None:
        assert _shape_compatible([10], [5, 10]) is False

    def test_mismatched_dim_value(self) -> None:
        assert _shape_compatible([10], [11]) is False

    def test_both_neg1(self) -> None:
        assert _shape_compatible([-1], [-1]) is True


# ---------------------------------------------------------------------------
# Tests: _get_opset
# ---------------------------------------------------------------------------


class TestGetOpset:
    def test_returns_correct_opset(self) -> None:
        m = _make_preprocessing_model(opset=15)
        assert _get_opset(m) == 15

    def test_returns_0_when_no_opset(self) -> None:
        m = onnx.ModelProto()
        assert _get_opset(m) == 0


# ---------------------------------------------------------------------------
# Tests: _dtype_name
# ---------------------------------------------------------------------------


class TestDtypeName:
    def test_float32(self) -> None:
        assert _dtype_name(onnx.TensorProto.FLOAT) == "float32"

    def test_int64(self) -> None:
        assert _dtype_name(onnx.TensorProto.INT64) == "int64"

    def test_unknown(self) -> None:
        name = _dtype_name(999)
        assert "999" in name


# ---------------------------------------------------------------------------
# Tests: _extract_tensor_info
# ---------------------------------------------------------------------------


class TestExtractTensorInfo:
    def test_extracts_float32_shape(self) -> None:
        m = _make_preprocessing_model(output_name="features", shape=[10])
        info = _extract_tensor_info(list(m.graph.output))
        assert "features" in info
        dtype, shape = info["features"]
        assert dtype == onnx.TensorProto.FLOAT
        assert shape == [10]


# ---------------------------------------------------------------------------
# Tests: check_onnx_boundary — pass cases
# ---------------------------------------------------------------------------


class TestCheckOnnxBoundaryPass:
    def test_matching_names_dtype_shape(self, tmp_path: Path) -> None:
        preproc_path = tmp_path / "preprocessing.onnx"
        model_path = tmp_path / "policy.onnx"
        _save(_make_preprocessing_model(output_name="features", shape=[10]), preproc_path)
        _save(_make_policy_model(input_name="features", shape=[10]), model_path)

        violations = check_onnx_boundary(str(preproc_path), str(model_path))
        assert violations == []

    def test_flexible_dim_is_accepted(self, tmp_path: Path) -> None:
        """preprocessing produces [10], model takes [-1] → compatible."""
        preproc_path = tmp_path / "preprocessing.onnx"
        model_path = tmp_path / "policy.onnx"
        _save(_make_preprocessing_model(output_name="features", shape=[10]), preproc_path)
        _save(_make_policy_model(input_name="features", shape=[-1]), model_path)

        violations = check_onnx_boundary(str(preproc_path), str(model_path))
        assert violations == []

    def test_same_opset_ok(self, tmp_path: Path) -> None:
        preproc_path = tmp_path / "preprocessing.onnx"
        model_path = tmp_path / "policy.onnx"
        _save(_make_preprocessing_model(shape=[10], opset=17), preproc_path)
        _save(_make_policy_model(shape=[10], opset=17), model_path)

        violations = check_onnx_boundary(str(preproc_path), str(model_path))
        assert violations == []

    def test_preproc_lower_opset_ok(self, tmp_path: Path) -> None:
        preproc_path = tmp_path / "preprocessing.onnx"
        model_path = tmp_path / "policy.onnx"
        _save(_make_preprocessing_model(shape=[10], opset=15), preproc_path)
        _save(_make_policy_model(shape=[10], opset=17), model_path)

        violations = check_onnx_boundary(str(preproc_path), str(model_path))
        assert violations == []


# ---------------------------------------------------------------------------
# Tests: check_onnx_boundary — fail cases
# ---------------------------------------------------------------------------


class TestCheckOnnxBoundaryFail:
    def test_missing_preprocessing_file(self, tmp_path: Path) -> None:
        model_path = tmp_path / "policy.onnx"
        _save(_make_policy_model(), model_path)

        violations = check_onnx_boundary(str(tmp_path / "nope.onnx"), str(model_path))
        assert len(violations) == 1
        assert "not found" in violations[0].lower()

    def test_missing_model_file(self, tmp_path: Path) -> None:
        preproc_path = tmp_path / "preprocessing.onnx"
        _save(_make_preprocessing_model(), preproc_path)

        violations = check_onnx_boundary(str(preproc_path), str(tmp_path / "nope.onnx"))
        assert len(violations) == 1
        assert "not found" in violations[0].lower()

    def test_both_files_missing(self, tmp_path: Path) -> None:
        violations = check_onnx_boundary(
            str(tmp_path / "a.onnx"), str(tmp_path / "b.onnx")
        )
        assert len(violations) == 2

    def test_name_mismatch(self, tmp_path: Path) -> None:
        """preprocessing outputs 'features' but model input is 'observation'."""
        preproc_path = tmp_path / "preprocessing.onnx"
        model_path = tmp_path / "policy.onnx"
        _save(_make_preprocessing_model(output_name="features"), preproc_path)
        _save(_make_policy_model(input_name="observation"), model_path)

        violations = check_onnx_boundary(str(preproc_path), str(model_path))
        assert any("features" in v for v in violations)

    def test_dtype_mismatch(self, tmp_path: Path) -> None:
        """preprocessing outputs float32, model expects float64."""
        preproc_path = tmp_path / "preprocessing.onnx"
        model_path = tmp_path / "policy.onnx"
        _save(
            _make_preprocessing_model(output_name="features", dtype=onnx.TensorProto.FLOAT),
            preproc_path,
        )
        _save(
            _make_policy_model(input_name="features", dtype=onnx.TensorProto.DOUBLE),
            model_path,
        )

        violations = check_onnx_boundary(str(preproc_path), str(model_path))
        assert any("dtype" in v for v in violations)

    def test_shape_mismatch(self, tmp_path: Path) -> None:
        """preprocessing outputs [10], model expects [11]."""
        preproc_path = tmp_path / "preprocessing.onnx"
        model_path = tmp_path / "policy.onnx"
        _save(_make_preprocessing_model(output_name="features", shape=[10]), preproc_path)
        _save(_make_policy_model(input_name="features", shape=[11]), model_path)

        violations = check_onnx_boundary(str(preproc_path), str(model_path))
        assert any("shape" in v for v in violations)

    def test_opset_violation(self, tmp_path: Path) -> None:
        """preprocessing opset (19) > model opset (17)."""
        preproc_path = tmp_path / "preprocessing.onnx"
        model_path = tmp_path / "policy.onnx"
        _save(_make_preprocessing_model(shape=[10], opset=19), preproc_path)
        _save(_make_policy_model(shape=[10], opset=17), model_path)

        violations = check_onnx_boundary(str(preproc_path), str(model_path))
        assert any("opset" in v for v in violations)

    def test_corrupt_file(self, tmp_path: Path) -> None:
        """A file that is not a valid ONNX model."""
        preproc_path = tmp_path / "preprocessing.onnx"
        model_path = tmp_path / "policy.onnx"
        preproc_path.write_bytes(b"NOT ONNX CONTENT")
        _save(_make_policy_model(), model_path)

        violations = check_onnx_boundary(str(preproc_path), str(model_path))
        assert len(violations) >= 1
        assert any("preprocessing" in v.lower() for v in violations)


# ---------------------------------------------------------------------------
# Tests: main() entrypoint
# ---------------------------------------------------------------------------


class TestMain:
    def test_returns_0_on_pass(self, tmp_path: Path) -> None:
        preproc_path = tmp_path / "preprocessing.onnx"
        model_path = tmp_path / "policy.onnx"
        _save(_make_preprocessing_model(output_name="features", shape=[10]), preproc_path)
        _save(_make_policy_model(input_name="features", shape=[10]), model_path)

        sys.argv = [
            "check_feature_contract.py",
            "--preprocessing",
            str(preproc_path),
            "--model",
            str(model_path),
        ]
        import check_feature_contract as cfc
        result = cfc.main()
        assert result == 0

    def test_returns_1_on_violation(self, tmp_path: Path) -> None:
        preproc_path = tmp_path / "preprocessing.onnx"
        model_path = tmp_path / "policy.onnx"
        _save(_make_preprocessing_model(output_name="features"), preproc_path)
        _save(_make_policy_model(input_name="observation"), model_path)

        sys.argv = [
            "check_feature_contract.py",
            "--preprocessing",
            str(preproc_path),
            "--model",
            str(model_path),
        ]
        import check_feature_contract as cfc
        result = cfc.main()
        assert result == 1

    def test_verbose_flag_shows_details(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        preproc_path = tmp_path / "preprocessing.onnx"
        model_path = tmp_path / "policy.onnx"
        _save(_make_preprocessing_model(output_name="features", shape=[10]), preproc_path)
        _save(_make_policy_model(input_name="features", shape=[10]), model_path)

        sys.argv = [
            "check_feature_contract.py",
            "--preprocessing",
            str(preproc_path),
            "--model",
            str(model_path),
            "--verbose",
        ]
        import check_feature_contract as cfc
        result = cfc.main()
        captured = capsys.readouterr()
        assert result == 0
        assert "opset" in captured.out.lower()
