"""Hypothesis input strategies for parity checks."""

from __future__ import annotations

from hypothesis import strategies as st


csv_floats = st.lists(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, width=32), min_size=3, max_size=3),
    min_size=1,
    max_size=8,
)


def matrix_to_csv(rows: list[list[float]]) -> bytes:
    lines = [",".join(str(x) for x in row) for row in rows]
    return ("\n".join(lines) + "\n").encode("utf-8")
