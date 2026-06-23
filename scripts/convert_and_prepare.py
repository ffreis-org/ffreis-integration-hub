#!/usr/bin/env python3
"""Generate a sklearn artifact and convert it through converter HTTP API."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import joblib
import requests
from sklearn.linear_model import LogisticRegression


def _wait_ready(base_url: str, timeout_seconds: float = 60.0) -> None:
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
    raise RuntimeError(f"converter API did not become ready at {base_url}")


def main() -> None:
    converter_base = os.environ.get("CONVERTER_API_BASE", "http://converter-api:8090")
    out_dir = Path(os.environ.get("OUT_DIR", "/shared"))
    out_dir.mkdir(parents=True, exist_ok=True)

    _wait_ready(converter_base)

    # Deterministic tiny binary classifier dataset.
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
    metadata_path = out_dir / "conversion_metadata.json"
    joblib.dump(model, artifact_path)

    payload = artifact_path.read_bytes()
    expected_sha = hashlib.sha256(payload).hexdigest()
    with artifact_path.open("rb") as handle:
        response = requests.post(
            f"{converter_base}/v1/convert/upload",
            data={
                "framework": "sklearn",
                "expected_sha256": expected_sha,
                "n_features": "3",
                "opset_version": "14",
                "allow_unsafe": "true",
            },
            files={"artifact": ("model.joblib", handle, "application/octet-stream")},
            timeout=120,
        )

    if response.status_code != 200:
        raise RuntimeError(f"conversion failed with status={response.status_code}: {response.text}")

    output_sha = hashlib.sha256(response.content).hexdigest()
    header_output_sha = response.headers.get("x-output-sha256", "").lower()
    if header_output_sha and output_sha != header_output_sha:
        raise RuntimeError(f"output sha mismatch: body={output_sha} header={header_output_sha}")

    onnx_path.write_bytes(response.content)
    metadata_path.write_text(
        json.dumps(
            {
                "input_sha256": expected_sha,
                "output_sha256": output_sha,
                "output_filename": response.headers.get("x-output-filename", "model.onnx"),
                "bytes": len(response.content),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"converted ONNX written to {onnx_path} ({len(response.content)} bytes)")


if __name__ == "__main__":
    main()
