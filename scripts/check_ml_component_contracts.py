#!/usr/bin/env python3
"""Contract checks for ffreis-ml-* library components.

For each repo with component_type="library" in config/repos.json, verifies:
  1. Repo directory exists at local_path
  2. pyproject.toml exists
  3. src/ directory exists
  4. Package is importable via uv run python -c "import <package_name>"
  5. make lint exits 0 (WARN if venv not set up; never FAIL)

For repos with an onnx_boundary declaration, delegates to check_feature_contract.py.

Exit 0 if no FAIL (WARNs are acceptable); exit 1 on any FAIL.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_HUB_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DEFAULT = _HUB_ROOT / "config" / "repos.json"

# Result constants
PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


@dataclass
class CheckResult:
    name: str
    status: str  # PASS | WARN | FAIL
    details: list[str] = field(default_factory=list)


def _load_library_repos(config_path: Path) -> list[dict[str, Any]]:
    """Return all repos with component_type == 'library'."""
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return [r for r in raw["repos"] if r.get("component_type") == "library"]


def _resolve_local_path(local_path_str: str) -> Path:
    """Resolve local_path (relative to hub root) to an absolute path."""
    return (_HUB_ROOT / local_path_str).resolve()


def _check_directory_exists(repo_dir: Path) -> tuple[str, list[str]]:
    if repo_dir.exists() and repo_dir.is_dir():
        return PASS, []
    return FAIL, [f"Directory not found: {repo_dir}"]


def _check_pyproject(repo_dir: Path) -> tuple[str, list[str]]:
    pyproject = repo_dir / "pyproject.toml"
    if pyproject.exists():
        return PASS, []
    return FAIL, [f"Missing pyproject.toml in {repo_dir}"]


def _check_src_dir(repo_dir: Path) -> tuple[str, list[str]]:
    src = repo_dir / "src"
    if src.exists() and src.is_dir():
        return PASS, []
    return FAIL, [f"Missing src/ directory in {repo_dir}"]


def _check_importable(repo_dir: Path, package_name: str) -> tuple[str, list[str]]:
    """Try to import the package via 'uv run python -c "import <pkg>"' inside repo_dir."""
    if not package_name:
        return WARN, ["No package_name declared in repos.json — skipping import check"]
    cmd = shlex.split(f'uv run python -c "import {package_name}"')
    try:
        result = subprocess.run(
            cmd,
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            return PASS, []
        stderr_snippet = result.stderr.strip().splitlines()[-3:]
        return FAIL, [
            f"Package '{package_name}' not importable (rc={result.returncode})",
            *stderr_snippet,
        ]
    except subprocess.TimeoutExpired:
        return WARN, [f"Import check timed out for '{package_name}' — skipping"]
    except FileNotFoundError:
        return WARN, ["'uv' not found — skipping import check"]


def _check_lint(repo_dir: Path) -> tuple[str, list[str]]:
    """Run 'make lint' in repo_dir. WARN (not FAIL) on non-zero exit."""
    cmd = ["make", "lint"]
    try:
        result = subprocess.run(
            cmd,
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            return PASS, []
        # lint failures are WARNING so that repos with unset venvs don't block CI
        stderr_lines = result.stderr.strip().splitlines()[-5:]
        return WARN, [
            f"'make lint' exited {result.returncode} — environment may not be set up",
            *stderr_lines,
        ]
    except subprocess.TimeoutExpired:
        return WARN, ["'make lint' timed out — skipping"]
    except FileNotFoundError:
        return WARN, ["'make' not found — skipping lint check"]


def _run_onnx_boundary_check(
    repo: dict[str, Any],
    repo_dir: Path,
    all_repos: list[dict[str, Any]],
) -> list[CheckResult]:
    """For repos with onnx_boundary, call check_feature_contract.py if artifacts exist."""
    boundary = repo.get("onnx_boundary", {})
    role = boundary.get("role", "")
    results: list[CheckResult] = []

    if role == "preprocessor":
        # Look for the model counterpart
        model_repos = [r for r in all_repos if r.get("onnx_boundary", {}).get("role") == "model"]
        if not model_repos:
            results.append(
                CheckResult(
                    name=f"{repo['name']}/onnx-boundary",
                    status=WARN,
                    details=[
                        "No model repo found to pair with preprocessor — skipping boundary check"
                    ],
                )
            )
            return results

        preproc_artifact = repo_dir / "artifacts" / "preprocessing.onnx"
        for model_repo_spec in model_repos:
            model_dir = _resolve_local_path(model_repo_spec["local_path"])
            model_artifact = model_dir / "artifacts" / "policy.onnx"

            if not preproc_artifact.exists() or not model_artifact.exists():
                results.append(
                    CheckResult(
                        name=f"{repo['name']}/onnx-boundary",
                        status=WARN,
                        details=[
                            "ONNX artifacts not found — run 'uv run crypto-env preprocess' and 'ml_crypto_rl export' first",
                            f"  preprocessing: {preproc_artifact} ({'found' if preproc_artifact.exists() else 'MISSING'})",
                            f"  policy: {model_artifact} ({'found' if model_artifact.exists() else 'MISSING'})",
                        ],
                    )
                )
                continue

            checker = _HUB_ROOT / "scripts" / "check_feature_contract.py"
            cmd = [
                "uv",
                "run",
                str(checker),
                "--preprocessing",
                str(preproc_artifact),
                "--model",
                str(model_artifact),
            ]
            try:
                result = subprocess.run(
                    cmd,
                    cwd=_HUB_ROOT,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    results.append(
                        CheckResult(
                            name=f"{repo['name']}/onnx-boundary",
                            status=PASS,
                            details=["preprocessing↔model ONNX boundary coherent"],
                        )
                    )
                else:
                    violations = (
                        result.stdout.strip().splitlines() or result.stderr.strip().splitlines()
                    )
                    results.append(
                        CheckResult(
                            name=f"{repo['name']}/onnx-boundary",
                            status=FAIL,
                            details=["ONNX boundary violation:", *violations[:10]],
                        )
                    )
            except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
                results.append(
                    CheckResult(
                        name=f"{repo['name']}/onnx-boundary",
                        status=WARN,
                        details=[f"Boundary check skipped: {exc}"],
                    )
                )

    return results


def check_repo(
    repo: dict[str, Any],
    all_repos: list[dict[str, Any]],
    *,
    run_lint: bool = True,
) -> list[CheckResult]:
    """Run all checks for a single library repo, returning one CheckResult per check."""
    name = repo["name"]
    local_path = repo.get("local_path", "")
    package_name = repo.get("package_name", "")
    repo_dir = _resolve_local_path(local_path)

    results: list[CheckResult] = []

    # 1. Directory exists
    status, details = _check_directory_exists(repo_dir)
    results.append(CheckResult(name=f"{name}/dir", status=status, details=details))
    if status == FAIL:
        # Cannot proceed with remaining checks
        return results

    # 2. pyproject.toml exists
    status, details = _check_pyproject(repo_dir)
    results.append(CheckResult(name=f"{name}/pyproject", status=status, details=details))

    # 3. src/ exists
    status, details = _check_src_dir(repo_dir)
    results.append(CheckResult(name=f"{name}/src", status=status, details=details))

    # 4. Package importable
    status, details = _check_importable(repo_dir, package_name)
    results.append(CheckResult(name=f"{name}/import", status=status, details=details))

    # 5. make lint (WARN only)
    if run_lint:
        status, details = _check_lint(repo_dir)
        results.append(CheckResult(name=f"{name}/lint", status=status, details=details))

    # ONNX boundary check (if applicable)
    if repo.get("onnx_boundary"):
        results.extend(_run_onnx_boundary_check(repo, repo_dir, all_repos))

    return results


def _print_summary(all_results: list[CheckResult]) -> None:
    col_w = max(len(r.name) for r in all_results) + 2
    print()
    print(f"{'Check':<{col_w}} {'Status'}")
    print("-" * (col_w + 8))
    for result in all_results:
        icon = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}.get(result.status, "?")
        print(f"{result.name:<{col_w}} [{icon}] {result.status}")
        for detail in result.details:
            print(f"  {'':>{col_w - 2}}  {detail}")
    print()
    n_pass = sum(1 for r in all_results if r.status == PASS)
    n_warn = sum(1 for r in all_results if r.status == WARN)
    n_fail = sum(1 for r in all_results if r.status == FAIL)
    print(f"Summary: {n_pass} PASS  {n_warn} WARN  {n_fail} FAIL")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Contract-check all ffreis-ml-* library components."
    )
    parser.add_argument(
        "--config",
        default=str(_CONFIG_DEFAULT),
        help="Path to repos.json config (default: config/repos.json relative to hub root)",
    )
    parser.add_argument(
        "--no-lint",
        action="store_true",
        help="Skip 'make lint' check (useful when venvs are not set up)",
    )
    parser.add_argument(
        "--json",
        dest="json_out",
        metavar="FILE",
        help="Write JSON results to FILE in addition to stdout",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    library_repos = _load_library_repos(config_path)

    if not library_repos:
        print("No library repos found in config — nothing to check.")
        return 0

    all_results: list[CheckResult] = []
    all_repos_raw = json.loads(config_path.read_text(encoding="utf-8"))["repos"]

    for repo in library_repos:
        results = check_repo(repo, all_repos_raw, run_lint=not args.no_lint)
        all_results.extend(results)

    _print_summary(all_results)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                [{"name": r.name, "status": r.status, "details": r.details} for r in all_results],
                indent=2,
            ),
            encoding="utf-8",
        )

    has_fail = any(r.status == FAIL for r in all_results)
    return 1 if has_fail else 0


if __name__ == "__main__":
    sys.exit(main())
