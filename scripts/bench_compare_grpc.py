#!/usr/bin/env python3
"""Benchmark Python and Rust serving gRPC APIs and assert prediction parity."""

from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import grpc
from onnx_serving_grpc import inference_pb2, inference_pb2_grpc


def _wait_ready(target: str, timeout_seconds: float = 90.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with grpc.insecure_channel(target) as channel:
                grpc.channel_ready_future(channel).result(timeout=3)
                stub = inference_pb2_grpc.InferenceServiceStub(channel)
                live = stub.Live(inference_pb2.LiveRequest(), timeout=3)
                ready = stub.Ready(inference_pb2.ReadyRequest(), timeout=3)
                if live.ok and ready.ok:
                    return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"service did not become ready at {target}")


def _predict(target: str, payload: bytes) -> tuple[Any, float]:
    started = time.perf_counter()
    with grpc.insecure_channel(target) as channel:
        stub = inference_pb2_grpc.InferenceServiceStub(channel)
        reply = stub.Predict(
            inference_pb2.PredictRequest(
                payload=payload,
                content_type="application/json",
                accept="application/json",
            ),
            timeout=10,
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return json.loads(reply.body.decode("utf-8")), elapsed_ms


def _numbers_close(a: Any, b: Any, atol: float = 1e-5) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False
        return all(_numbers_close(a[key], b[key], atol) for key in a)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_numbers_close(x, y, atol) for x, y in zip(a, b, strict=True))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), abs_tol=atol, rel_tol=0.0)
    return a == b


def _summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    n = len(ordered)
    return {
        "count": float(n),
        "mean_ms": sum(ordered) / max(n, 1),
        "p50_ms": ordered[max(0, int(n * 0.50) - 1)],
        "p95_ms": ordered[max(0, int(n * 0.95) - 1)],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def main() -> None:
    python_target = os.environ.get("PYTHON_GRPC_TARGET", "python-grpc:50052")
    rust_target = os.environ.get("RUST_GRPC_TARGET", "rust-grpc:50052")
    iterations = int(os.environ.get("BENCH_ITERATIONS", "100"))
    artifacts_dir = Path(os.environ.get("ARTIFACTS_DIR", "/shared"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    _wait_ready(python_target)
    _wait_ready(rust_target)

    payload = json.dumps({"instances": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0]]}).encode(
        "utf-8"
    )
    py_latencies: list[float] = []
    rs_latencies: list[float] = []
    mismatches: list[dict[str, Any]] = []

    for idx in range(iterations):
        with ThreadPoolExecutor(max_workers=2) as pool:
            py_future = pool.submit(_predict, python_target, payload)
            rs_future = pool.submit(_predict, rust_target, payload)
            py_output, py_ms = py_future.result()
            rs_output, rs_ms = rs_future.result()
        py_latencies.append(py_ms)
        rs_latencies.append(rs_ms)
        if not _numbers_close(py_output, rs_output):
            mismatches.append(
                {
                    "iteration": idx,
                    "python_output": py_output,
                    "rust_output": rs_output,
                }
            )

    report = {
        "iterations": iterations,
        "python": _summarize(py_latencies),
        "rust": _summarize(rs_latencies),
        "throughput_rps": {
            "python": round(1000.0 / (sum(py_latencies) / max(len(py_latencies), 1)), 2),
            "rust": round(1000.0 / (sum(rs_latencies) / max(len(rs_latencies), 1)), 2),
        },
        "mismatch_count": len(mismatches),
        "mismatches_sample": mismatches[:3],
    }
    output_path = artifacts_dir / "converter_serving_bench_grpc.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if mismatches:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
