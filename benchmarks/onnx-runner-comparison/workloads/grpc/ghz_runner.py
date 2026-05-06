"""Optional ghz wrapper (gRPC load)."""

from __future__ import annotations

import subprocess


def run_ghz(target: str, proto_path: str, duration_s: int = 30) -> int:
    cmd = [
        "ghz",
        "--insecure",
        "--proto",
        proto_path,
        "--call",
        "onnxserving.grpc.InferenceService.Live",
        "--duration",
        f"{duration_s}s",
        target,
    ]
    return subprocess.call(cmd)
