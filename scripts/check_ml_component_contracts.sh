#!/usr/bin/env bash
# check_ml_component_contracts.sh — run ML component contract tests and report PASS/FAIL per contract.
#
# Usage:
#   bash scripts/check_ml_component_contracts.sh [--no-color]
#
# Exit codes:
#   0  all contracts PASS
#   1  one or more contracts FAIL
#
# Requires: uv (for running pytest in the project venv)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_NO_COLOR="${1:-}"
_GREEN=""
_RED=""
_RESET=""
if [[ "${_NO_COLOR}" != "--no-color" ]] && [[ -t 1 ]]; then
    _GREEN="\033[0;32m"
    _RED="\033[0;31m"
    _RESET="\033[0m"
fi

_pass() { printf "%sPASS%s  %s\n" "${_GREEN}" "${_RESET}" "$1"; }
_fail() { printf "%sFAIL%s  %s\n" "${_RED}" "${_RESET}" "$1"; }

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_exit_code=0
declare -a _results=()

# ---------------------------------------------------------------------------
# Contract runner
# ---------------------------------------------------------------------------

# run_contract <label> <test_file>
#   Runs pytest against a single test file. Records PASS or FAIL in _results[].
run_contract() {
    local label="$1"
    local test_file="$2"

    printf "\n=== Contract: %s ===\n" "${label}"
    printf "    file: %s\n" "${test_file}"

    if uv run --project "${REPO_ROOT}" pytest \
            --no-header -q \
            --tb=short \
            -m integration \
            "${REPO_ROOT}/${test_file}" 2>&1; then
        _results+=("PASS:${label}")
    else
        _results+=("FAIL:${label}")
        _exit_code=1
    fi
}

# ---------------------------------------------------------------------------
# Run contracts
# ---------------------------------------------------------------------------

printf "ML component contract checks — %s\n" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
printf "repo: %s\n" "${REPO_ROOT}"

run_contract \
    "ONNX feature boundary (crypto-env → serving)" \
    "tests/test_ml_onnx_feature_boundary.py"

run_contract \
    "Registry → serve parity (ml-registry → serving)" \
    "tests/test_ml_registry_serve_parity.py"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

printf "\n%-60s %s\n" "Contract" "Result"
printf "%s\n" "-----------------------------------------------------------------------"

for result in "${_results[@]}"; do
    status="${result%%:*}"
    label="${result#*:}"
    if [[ "${status}" == "PASS" ]]; then
        _pass "${label}"
    else
        _fail "${label}"
    fi
done

printf "\n"
if [[ "${_exit_code}" -eq 0 ]]; then
    printf "%sAll ML component contracts PASS.%s\n" "${_GREEN}" "${_RESET}"
else
    printf "%sOne or more ML component contracts FAILED.%s\n" "${_RED}" "${_RESET}"
fi

exit "${_exit_code}"
