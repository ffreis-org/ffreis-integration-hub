#!/usr/bin/env python3
"""ONNX preprocessing↔model boundary contract checker.

Verifies that a preprocessing ONNX graph and a model ONNX graph are
shape/dtype/name-compatible at their shared boundary, using static graph
inspection only (no onnxruntime required).

Checks:
  1. Both files exist
  2. Both are valid ONNX models (onnx.checker.check_model)
  3. preprocessing output names ⊆ model input names (names match)
  4. preprocessing output dtypes == model input dtypes
  5. preprocessing output shapes compatible with model input shapes
     (-1 dims are flexible and always accepted)
  6. preprocessing opset ≤ model opset (both use same domain base)

Usage:
    python scripts/check_feature_contract.py \\
        --preprocessing artifacts/preprocessing.onnx \\
        --model artifacts/policy.onnx

    # From Makefile:
    make check-feature-contract ARGS="--preprocessing ... --model ..."

Exit 0 on no violations; exit 1 on any violation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import onnx
    from onnx import TensorProto
except ImportError as exc:
    print(f"ERROR: onnx not installed — install with 'uv add onnx': {exc}", file=sys.stderr)
    sys.exit(2)


# Map ONNX data type integer codes to human-readable names
_DTYPE_NAMES: dict[int, str] = {
    TensorProto.FLOAT: "float32",
    TensorProto.DOUBLE: "float64",
    TensorProto.INT32: "int32",
    TensorProto.INT64: "int64",
    TensorProto.BOOL: "bool",
    TensorProto.STRING: "string",
    TensorProto.UINT8: "uint8",
    TensorProto.INT8: "int8",
    TensorProto.UINT16: "uint16",
    TensorProto.INT16: "int16",
    TensorProto.UINT32: "uint32",
    TensorProto.UINT64: "uint64",
    TensorProto.FLOAT16: "float16",
    TensorProto.BFLOAT16: "bfloat16",
    TensorProto.COMPLEX64: "complex64",
    TensorProto.COMPLEX128: "complex128",
}


def _dtype_name(code: int) -> str:
    return _DTYPE_NAMES.get(code, f"dtype({code})")


def _get_opset(model: onnx.ModelProto) -> int:
    """Return the highest opset version for the default (empty-domain) opset."""
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            return int(opset.version)
    return 0


def _shape_compatible(preproc_shape: list[int | str], model_shape: list[int | str]) -> bool:
    """Return True if preproc_shape is compatible with model_shape.

    Rules:
    - Dimension counts must match.
    - A -1 (or a symbolic dim string) in either side is flexible.
    - Otherwise dims must be equal.
    """
    if len(preproc_shape) != len(model_shape):
        return False
    for p, m in zip(preproc_shape, model_shape):
        # -1 or symbolic string = flexible dim
        if p == -1 or m == -1:
            continue
        if isinstance(p, str) or isinstance(m, str):
            continue
        if p != m:
            return False
    return True


def _extract_tensor_info(
    value_infos: list[onnx.ValueInfoProto],
) -> dict[str, tuple[int, list[int | str]]]:
    """Return {name: (dtype_code, shape)} for each tensor value_info."""
    result: dict[str, tuple[int, list[int | str]]] = {}
    for vi in value_infos:
        t = vi.type.tensor_type
        dtype_code = t.elem_type
        shape: list[int | str] = []
        if t.HasField("shape"):
            for dim in t.shape.dim:
                if dim.HasField("dim_value"):
                    shape.append(dim.dim_value)
                elif dim.HasField("dim_param"):
                    shape.append(dim.dim_param)
                else:
                    shape.append(-1)
        result[vi.name] = (dtype_code, shape)
    return result


def check_onnx_boundary(preprocessing_path: str, model_path: str) -> list[str]:
    """Return a list of violation strings (empty list = pass).

    Checks:
    1. preprocessing_path and model_path exist
    2. Both are valid ONNX models (onnx.checker.check_model)
    3. preprocessing output names ⊆ model input names
    4. preprocessing output dtypes == model input dtypes
    5. preprocessing output shapes compatible with model input shapes
    6. preprocessing opset ≤ model opset
    """
    violations: list[str] = []
    preproc_p = Path(preprocessing_path)
    model_p = Path(model_path)

    # Check 1: files exist
    if not preproc_p.exists():
        violations.append(f"File not found: {preproc_p}")
    if not model_p.exists():
        violations.append(f"File not found: {model_p}")
    if violations:
        return violations

    # Load models
    try:
        preproc_model = onnx.load(str(preproc_p))
    except Exception as exc:
        violations.append(f"Failed to load preprocessing model: {exc}")
        return violations

    try:
        policy_model = onnx.load(str(model_p))
    except Exception as exc:
        violations.append(f"Failed to load policy model: {exc}")
        return violations

    # Check 2: valid ONNX models
    try:
        onnx.checker.check_model(preproc_model)
    except onnx.checker.ValidationError as exc:
        violations.append(f"preprocessing.onnx is invalid: {exc}")

    try:
        onnx.checker.check_model(policy_model)
    except onnx.checker.ValidationError as exc:
        violations.append(f"model.onnx is invalid: {exc}")

    if violations:
        return violations

    # Extract output info from preprocessing model
    preproc_outputs = _extract_tensor_info(list(preproc_model.graph.output))

    # Extract input info from policy model
    policy_inputs = _extract_tensor_info(list(policy_model.graph.input))

    # Check 3: preprocessing output names ⊆ model input names
    # (preprocessing may produce a subset of what the model takes)
    preproc_output_names = set(preproc_outputs.keys())
    policy_input_names = set(policy_inputs.keys())
    unknown_names = preproc_output_names - policy_input_names
    if unknown_names:
        violations.append(
            f"preprocessing outputs not found in model inputs: {sorted(unknown_names)}"
            f" | model inputs are: {sorted(policy_input_names)}"
        )

    # Checks 4 & 5: for matched names, check dtype + shape compatibility
    for name in preproc_output_names & policy_input_names:
        p_dtype, p_shape = preproc_outputs[name]
        m_dtype, m_shape = policy_inputs[name]

        if p_dtype != m_dtype:
            violations.append(
                f"dtype mismatch for tensor '{name}': "
                f"preprocessing={_dtype_name(p_dtype)} vs model={_dtype_name(m_dtype)}"
            )

        if not _shape_compatible(p_shape, m_shape):
            violations.append(
                f"shape incompatibility for tensor '{name}': "
                f"preprocessing={p_shape} vs model={m_shape}"
            )

    # Check 6: opset compatibility
    preproc_opset = _get_opset(preproc_model)
    model_opset = _get_opset(policy_model)
    if preproc_opset > model_opset and model_opset > 0:
        violations.append(
            f"preprocessing opset ({preproc_opset}) > model opset ({model_opset}); "
            "model may not support all ops used by preprocessing"
        )

    return violations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check ONNX preprocessing↔model boundary coherence (static, no runtime)."
    )
    parser.add_argument(
        "--preprocessing",
        required=True,
        metavar="ONNX_FILE",
        help="Path to the preprocessing ONNX model (e.g. artifacts/preprocessing.onnx)",
    )
    parser.add_argument(
        "--model",
        required=True,
        metavar="ONNX_FILE",
        help="Path to the policy/model ONNX file (e.g. artifacts/policy.onnx)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print details even when no violations are found",
    )
    args = parser.parse_args()

    print("Checking ONNX boundary:")
    print(f"  preprocessing : {args.preprocessing}")
    print(f"  model         : {args.model}")

    violations = check_onnx_boundary(args.preprocessing, args.model)

    if not violations:
        print("OK — no boundary violations detected.")
        if args.verbose:
            preproc_model = onnx.load(args.preprocessing)
            policy_model = onnx.load(args.model)
            preproc_outputs = _extract_tensor_info(list(preproc_model.graph.output))
            policy_inputs = _extract_tensor_info(list(policy_model.graph.input))
            matched = set(preproc_outputs) & set(policy_inputs)
            print(f"  matched tensors ({len(matched)}): {sorted(matched)}")
            print(f"  preprocessing opset: {_get_opset(preproc_model)}")
            print(f"  model opset: {_get_opset(policy_model)}")
        return 0

    print(f"FAIL — {len(violations)} violation(s) found:")
    for i, v in enumerate(violations, 1):
        print(f"  {i}. {v}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
