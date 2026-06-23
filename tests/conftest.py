"""pytest configuration for integration-hub tests.

Adds scripts/ to sys.path so test files can import the scripts as modules,
and ensures coverage tracks them.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add scripts/ to sys.path so tests can do: import check_ml_component_contracts
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
