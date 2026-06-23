#!/usr/bin/env python3
"""Generate a sklearn artifact and convert it through converter gRPC API."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import grpc
import joblib
from converter_grpc import converter_pb2, converter_pb2_grpc
from sklearn.linear_model import LogisticRegression


def _wait_grpc(target: str, timeout_seconds: float = 60.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with grpc.insecure_channel(target) as channel:
                grpc.channel_ready_future(channel).result(timeout=3)
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"converter gRPC did not become ready at {target}")


def _request_chunks(
    artifact: bytes,
    expected_sha: str,
) -> list[converter_pb2.ConvertRequestChunk]:
    chunks: list[converter_pb2.ConvertRequestChunk] = [
        converter_pb2.ConvertRequestChunk(
            metadata=converter_pb2.ConvertMetadata(
                framework="sklearn",
                filename="model.joblib",
                expected_sha256=expected_sha,
                n_features=3,
                opset_version=14,
                allow_unsafe=True,
            )
        )
    ]
    chunk_size = 1 << 20
    for offset in range(0, len(artifact), chunk_size):
        chunks.append(
            converter_pb2.ConvertRequestChunk(data=artifact[offset : offset + chunk_size])
        )
    return chunks


def main() -> None:
    converter_target = os.environ.get("CONVERTER_GRPC_TARGET", "converter-grpc:8091")
    out_dir = Path(os.environ.get("OUT_DIR", "/shared"))
    out_dir.mkdir(parents=True, exist_ok=True)

    _wait_grpc(converter_target)

    x_train = [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ]
    y_train = [0, 0, 0, 1, 0, 1, 1, 1]
    model = LogisticRegression(random_state=42, solver="liblinear")
    model.fit(x_train, y_train)

    artifact_path = out_dir / "model.joblib"
    onnx_path = out_dir / "model.onnx"
    metadata_path = out_dir / "conversion_metadata_grpc.json"
    joblib.dump(model, artifact_path)
    payload = artifact_path.read_bytes()
    expected_sha = hashlib.sha256(payload).hexdigest()

    with grpc.insecure_channel(converter_target) as channel:
        stub = converter_pb2_grpc.ConverterServiceStub(channel)
        replies = list(stub.Convert(iter(_request_chunks(payload, expected_sha)), timeout=120))

    if not replies:
        raise RuntimeError("converter gRPC returned no replies")

    first = replies[0]
    if not first.HasField("result"):
        raise RuntimeError("first converter gRPC reply did not carry result metadata")
    result = first.result
    onnx_bytes = b"".join(reply.data for reply in replies[1:])
    if not onnx_bytes:
        raise RuntimeError("converter gRPC returned empty ONNX payload")

    output_sha = hashlib.sha256(onnx_bytes).hexdigest()
    if result.output_sha256 and output_sha != result.output_sha256.lower():
        raise RuntimeError(f"output sha mismatch: body={output_sha} result={result.output_sha256}")

    onnx_path.write_bytes(onnx_bytes)
    metadata_path.write_text(
        json.dumps(
            {
                "input_sha256": result.input_sha256 or expected_sha,
                "output_sha256": output_sha,
                "output_filename": result.output_filename or "model.onnx",
                "bytes": len(onnx_bytes),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"converted ONNX written to {onnx_path} ({len(onnx_bytes)} bytes)")


if __name__ == "__main__":
    main()
