#!/usr/bin/env python3
"""Compatibility checks for stock simulator payloads and RL agent schema expectations."""

from __future__ import annotations

import ast
import json
import os
import time
from collections.abc import Mapping
from http.client import HTTPConnection, HTTPSConnection
from pathlib import Path
from urllib.parse import urlsplit


def _validate_base(base_url: str) -> tuple[str, str]:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"SIMULATOR_BASE_URL must use http/https, got {parsed.scheme!r}")
    if not parsed.netloc:
        raise ValueError("SIMULATOR_BASE_URL must include host[:port]")
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


def _wait_ready(scheme: str, netloc: str, timeout_seconds: float = 40.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            status, payload = _request(scheme, netloc, "/readyz", timeout_seconds=3.0)
            if status == 200:
                body = json.loads(payload.decode("utf-8"))
                if body.get("status") == "ready":
                    return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("timed out waiting for simulator readiness on /readyz")


def _must_json(raw: bytes) -> object:
    return json.loads(raw.decode("utf-8"))


def _assert_number_list(value: object, size: int, *, field: str) -> list[float]:
    assert isinstance(value, list), f"{field} must be a list"
    assert len(value) == size, f"{field} must have length {size}, got {len(value)}"
    out: list[float] = []
    for idx, item in enumerate(value):
        assert isinstance(item, (int, float)), f"{field}[{idx}] must be numeric"
        out.append(float(item))
    return out


def _assert_observation_payload(payload: object) -> tuple[list[float], list[float], list[float]]:
    assert isinstance(payload, Mapping), "observation payload must be an object"
    market = payload.get("market_window_handle")
    assert isinstance(market, Mapping), "market_window_handle must be an object"
    for key in ("start", "end", "t", "current_price"):
        assert isinstance(market.get(key), (int, float)), (
            f"market_window_handle.{key} must be numeric"
        )

    market_vec = [
        float(market["start"]),
        float(market["end"]),
        float(market["t"]),
        float(market["current_price"]),
    ]
    portfolio = _assert_number_list(payload.get("portfolio_vector"), 4, field="portfolio_vector")
    order_summary = _assert_number_list(
        payload.get("order_summary_vector"), 3, field="order_summary_vector"
    )
    assert isinstance(payload.get("done"), bool), "done must be a bool"
    return market_vec, portfolio, order_summary


def _dataclass_fields(path: Path, class_name: str) -> set[str]:
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            names: set[str] = set()
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    names.add(item.target.id)
            return names
    raise ValueError(f"class {class_name!r} not found in {path}")


def _check_runtime_event_contract(sim_base_url: str) -> None:
    scheme, netloc = _validate_base(sim_base_url)
    _wait_ready(scheme, netloc)

    status, reset_raw = _request(
        scheme,
        netloc,
        "/v1/reset",
        method="POST",
        body=json.dumps({"seed": 2026}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    assert status == 200, reset_raw.decode("utf-8", errors="replace")

    status, observe_raw = _request(scheme, netloc, "/v1/observe")
    assert status == 200, observe_raw.decode("utf-8", errors="replace")
    observe_body = _must_json(observe_raw)
    assert isinstance(observe_body, Mapping), "observe response must be an object"
    obs = observe_body.get("observation")
    market_vec, portfolio, orders = _assert_observation_payload(obs)
    combined = [*market_vec, *portfolio, *orders]
    assert len(combined) == 11, "agent policy expects 11 input features per observation"

    # Agent-encoded actions must remain accepted by simulator step_many payload contract.
    valid_actions = {
        "actions": [
            {"side_code": 1, "units": 1.0, "order_type_code": 0, "has_limit_price": False},
            {
                "side_code": -1,
                "units": 0.5,
                "order_type_code": 1,
                "has_limit_price": True,
                "limit_price": 100.0,
            },
            {"side_code": 0, "units": 0.0, "order_type_code": 0, "has_limit_price": False},
        ]
    }
    status, step_raw = _request(
        scheme,
        netloc,
        "/v1/step_many",
        method="POST",
        body=json.dumps(valid_actions).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    assert status == 200, step_raw.decode("utf-8", errors="replace")
    step_body = _must_json(step_raw)
    assert isinstance(step_body, Mapping), "step_many response must be an object"

    observations = step_body.get("observations")
    rewards = step_body.get("rewards")
    dones = step_body.get("dones")
    assert isinstance(observations, list) and len(observations) == 3
    assert isinstance(rewards, list) and len(rewards) == 3
    assert isinstance(dones, list) and len(dones) == 3
    for item in observations:
        _assert_observation_payload(item)

    for idx, reward in enumerate(rewards):
        assert isinstance(reward, (int, float)), f"rewards[{idx}] must be numeric"
    for idx, done in enumerate(dones):
        assert isinstance(done, bool), f"dones[{idx}] must be bool"


def _check_replay_schema_contract(root_dir: Path) -> None:
    simulator_repo = Path(
        os.getenv("SIMULATOR_REPO", str(root_dir.parent / "ffreis-stock-simulator"))
    )
    agent_repo = Path(os.getenv("AGENT_REPO", str(root_dir.parent / "ffreis-stock-rl-agent")))

    simulator_recorder = simulator_repo / "src" / "stock_simulator" / "recorder.py"
    agent_replay = agent_repo / "src" / "stock_rl_agent" / "replay.py"

    if not simulator_recorder.exists():
        raise FileNotFoundError(f"missing simulator recorder file: {simulator_recorder}")
    if not agent_replay.exists():
        raise FileNotFoundError(f"missing agent replay file: {agent_replay}")

    recorded_step_fields = _dataclass_fields(simulator_recorder, "RecordedStep")
    replay_row_fields = _dataclass_fields(agent_replay, "ReplayRow")

    required_recorded_step_fields = {
        "episode_id",
        "step",
        "seed",
        "side",
        "order_type",
        "units",
        "limit_price",
        "equity",
        "leverage",
        "price",
    }
    required_replay_row_fields = {
        "state_seq",
        "side_idx",
        "order_idx",
        "units_value",
        "reward",
        "done",
    }

    missing_recorded_step = required_recorded_step_fields - recorded_step_fields
    missing_replay_row = required_replay_row_fields - replay_row_fields
    assert not missing_recorded_step, (
        f"RecordedStep missing fields: {sorted(missing_recorded_step)}"
    )
    assert not missing_replay_row, f"ReplayRow missing fields: {sorted(missing_replay_row)}"


def main() -> int:
    root_dir = Path(__file__).resolve().parent.parent
    sim_base_url = os.getenv("SIMULATOR_BASE_URL", "http://127.0.0.1:18000")

    _check_runtime_event_contract(sim_base_url)
    _check_replay_schema_contract(root_dir)

    print("stock simulator <-> rl agent schema compatibility check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
