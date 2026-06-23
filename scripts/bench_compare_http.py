#!/usr/bin/env python3
"""Benchmark Python and Rust serving HTTP APIs and assert prediction parity."""

from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests


def _wait_ready(base_url: str, timeout_seconds: float = 90.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            health = requests.get(f"{base_url}/healthz", timeout=3)
            ready = requests.get(f"{base_url}/readyz", timeout=3)
            if health.status_code == 200 and ready.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"service did not become ready at {base_url}")


def _post_predict(base_url: str, payload: dict[str, Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    response = requests.post(
        f"{base_url}/invocations",
        json=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        timeout=10,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if response.status_code != 200:
        raise RuntimeError(f"{base_url} failed status={response.status_code} body={response.text}")
    return response.json(), elapsed_ms


def _numbers_close(a: Any, b: Any, atol: float = 1e-5) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False
        return all(_numbers_close(a[key], b[key], atol) for key in a)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_numbers_close(va, vb, atol) for va, vb in zip(a, b, strict=True))
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
    python_base = os.environ.get("PYTHON_API_BASE", "http://python-api:8080")
    rust_base = os.environ.get("RUST_API_BASE", "http://rust-api:8080")
    iterations = int(os.environ.get("BENCH_ITERATIONS", "100"))
    artifacts_dir = Path(os.environ.get("ARTIFACTS_DIR", "/shared"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    _wait_ready(python_base)
    _wait_ready(rust_base)

    payload = {"instances": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0]]}
    python_latencies: list[float] = []
    rust_latencies: list[float] = []
    mismatches: list[dict[str, Any]] = []

    for idx in range(iterations):
        with ThreadPoolExecutor(max_workers=2) as pool:
            py_future = pool.submit(_post_predict, python_base, payload)
            rs_future = pool.submit(_post_predict, rust_base, payload)
            py_output, py_ms = py_future.result()
            rs_output, rs_ms = rs_future.result()

        python_latencies.append(py_ms)
        rust_latencies.append(rs_ms)
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
        "python": _summarize(python_latencies),
        "rust": _summarize(rust_latencies),
        "throughput_rps": {
            "python": round(1000.0 / (sum(python_latencies) / max(len(python_latencies), 1)), 2),
            "rust": round(1000.0 / (sum(rust_latencies) / max(len(rust_latencies), 1)), 2),
        },
        "mismatch_count": len(mismatches),
        "mismatches_sample": mismatches[:3],
    }

    output_path = artifacts_dir / "converter_serving_bench.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if mismatches:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
