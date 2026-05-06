"""Property-based parity checks between python and rust services."""

from __future__ import annotations

import httpx
from hypothesis import given, settings

from workloads.hypothesis.strategies import csv_floats, matrix_to_csv


def _invoke(base_url: str, payload: bytes) -> list[list[float]]:
    response = httpx.post(
        f"{base_url}/invocations",
        content=payload,
        headers={"Content-Type": "text/csv", "Accept": "application/json"},
        timeout=5.0,
    )
    response.raise_for_status()
    return response.json()


@given(rows=csv_floats)
@settings(max_examples=25, deadline=None)
def parity_property(rows: list[list[float]], python_base: str, rust_base: str) -> None:
    payload = matrix_to_csv(rows)
    py_out = _invoke(python_base, payload)
    rs_out = _invoke(rust_base, payload)
    assert py_out == rs_out
