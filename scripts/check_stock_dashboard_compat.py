#!/usr/bin/env python3
"""Compatibility smoke check for stock simulator + Go dashboard."""

from __future__ import annotations

import json
import os
import time
from http.client import HTTPConnection, HTTPSConnection
from urllib.parse import urlsplit


def _validate_base(base_url: str) -> tuple[str, str]:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"BASE_URL must use http or https; got {parsed.scheme or '<empty>'}")
    if not parsed.netloc:
        raise ValueError("BASE_URL must include host[:port]")
    return parsed.scheme, parsed.netloc


def _request(
    scheme: str,
    netloc: str,
    path: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout_seconds: float = 5.0,
) -> tuple[int, bytes]:
    conn_cls = HTTPSConnection if scheme == "https" else HTTPConnection
    conn = conn_cls(netloc, timeout=timeout_seconds)
    conn.request(method, path, body=body, headers=headers or {})
    response = conn.getresponse()
    status = response.status
    payload = response.read()
    conn.close()
    return status, payload


def _wait_ok(scheme: str, netloc: str, path: str, timeout_seconds: float = 40.0) -> bytes:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            status, payload = _request(scheme, netloc, path, timeout_seconds=3.0)
            if status == 200:
                return payload
        except Exception as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"timed out waiting for HTTP 200 at {path}: {last_error}")


def _must_json(raw: bytes) -> dict[str, object]:
    return json.loads(raw.decode("utf-8"))


def _assert_state_payload(payload: dict[str, object]) -> None:
    readyz = payload.get("readyz")
    assert isinstance(readyz, dict), payload
    assert readyz.get("status") in {"ready", "not_ready"}, payload

    observation = payload.get("observation")
    if observation is None:
        # Allowed only when simulator is not ready.
        assert readyz.get("status") != "ready", payload
        return

    assert isinstance(observation, dict), payload
    market = observation.get("market_window_handle")
    portfolio = observation.get("portfolio_vector")
    order_summary = observation.get("order_summary_vector")

    assert isinstance(market, dict), observation
    assert isinstance(portfolio, list) and len(portfolio) == 4, observation
    assert isinstance(order_summary, list) and len(order_summary) == 3, observation


def main() -> int:
    base_url = os.getenv("DASHBOARD_BASE_URL", "http://127.0.0.1:18080")
    scheme, netloc = _validate_base(base_url)

    # Wait dashboard health endpoint.
    _wait_ok(scheme, netloc, "/healthz")

    # Dashboard state endpoint must proxy simulator readiness and observation.
    state_raw = _wait_ok(scheme, netloc, "/api/state")
    state_payload = _must_json(state_raw)
    _assert_state_payload(state_payload)

    # Reset through dashboard proxy.
    status, reset_raw = _request(
        scheme,
        netloc,
        "/api/reset",
        method="POST",
        body=json.dumps({"seed": 1234}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        timeout_seconds=5.0,
    )
    assert status == 200, (status, reset_raw.decode("utf-8", errors="replace"))

    # Step through dashboard proxy using market order buy action.
    step_payload = {
        "side": "buy",
        "units": 1.0,
        "order_type": "market",
        "limit_price": None,
    }
    status, step_raw = _request(
        scheme,
        netloc,
        "/api/step",
        method="POST",
        body=json.dumps(step_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        timeout_seconds=5.0,
    )
    assert status == 200, (status, step_raw.decode("utf-8", errors="replace"))
    step_json = _must_json(step_raw)
    assert isinstance(step_json.get("observations"), list), step_json
    assert isinstance(step_json.get("rewards"), list), step_json
    assert isinstance(step_json.get("dones"), list), step_json

    # After reset+step, t should be present and non-negative.
    status, state_after_raw = _request(scheme, netloc, "/api/state", timeout_seconds=5.0)
    assert status == 200
    state_after = _must_json(state_after_raw)
    _assert_state_payload(state_after)

    observation = state_after.get("observation")
    assert isinstance(observation, dict), state_after
    market = observation.get("market_window_handle")
    assert isinstance(market, dict), state_after
    assert int(market.get("t", -1)) >= 0, state_after

    print("stock simulator + dashboard compatibility check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
