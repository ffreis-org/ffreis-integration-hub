"""Readiness helpers."""

from __future__ import annotations

import time

import httpx


def wait_http_ok(url: str, timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=3.0)
            if response.status_code == 200:
                return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")
