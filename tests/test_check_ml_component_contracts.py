"""Tests for check_ml_component_contracts.py.

Uses a mock filesystem to exercise PASS/WARN/FAIL logic without touching
real sibling repos or running make/uv commands.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from check_ml_component_contracts import (
    FAIL,
    PASS,
    WARN,
    CheckResult,
    _check_directory_exists,
    _check_importable,
    _check_lint,
    _check_pyproject,
    _check_src_dir,
    _load_library_repos,
    _print_summary,
    _resolve_local_path,
    check_repo,
    main,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_repo_dir(tmp_path: Path) -> Path:
    """A directory that looks like a valid minimal ml component."""
    repo = tmp_path / "my-component"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='my-component'\n", encoding="utf-8")
    return repo


@pytest.fixture()
def repos_json(tmp_path: Path, minimal_repo_dir: Path) -> Path:
    """A minimal repos.json with one library repo pointing at minimal_repo_dir."""
    config = {
        "repos": [
            {
                "name": "ml-test-lib",
                "url": "https://github.com/example/test.git",
                "ref": "main",
                "local_path": str(minimal_repo_dir),
                "component_type": "library",
                "package_name": "my_component",
                "checks": ["make lint", "make test"],
            }
        ],
        "contracts": [],
    }
    config_path = tmp_path / "repos.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


@pytest.fixture()
def service_repos_json(tmp_path: Path) -> Path:
    """repos.json with only service-type repos (no library)."""
    config = {
        "repos": [
            {
                "name": "converter",
                "url": "https://example.com/converter.git",
                "ref": "main",
                "local_path": "/nonexistent/converter",
                "component_type": "service",
                "checks": ["make grpc-check"],
            }
        ],
        "contracts": [],
    }
    config_path = tmp_path / "repos.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


# ---------------------------------------------------------------------------
# Unit tests: individual check functions
# ---------------------------------------------------------------------------


class TestCheckDirectoryExists:
    def test_pass_when_exists(self, tmp_path: Path) -> None:
        status, details = _check_directory_exists(tmp_path)
        assert status == PASS
        assert details == []

    def test_fail_when_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / "no-such-dir"
        status, details = _check_directory_exists(missing)
        assert status == FAIL
        assert len(details) == 1
        assert "not found" in details[0].lower()

    def test_fail_when_path_is_file(self, tmp_path: Path) -> None:
        f = tmp_path / "afile.txt"
        f.write_text("x", encoding="utf-8")
        status, details = _check_directory_exists(f)
        assert status == FAIL


class TestCheckPyproject:
    def test_pass_when_present(self, minimal_repo_dir: Path) -> None:
        status, details = _check_pyproject(minimal_repo_dir)
        assert status == PASS
        assert details == []

    def test_fail_when_missing(self, tmp_path: Path) -> None:
        status, details = _check_pyproject(tmp_path)
        assert status == FAIL
        assert "pyproject.toml" in details[0]


class TestCheckSrcDir:
    def test_pass_when_present(self, minimal_repo_dir: Path) -> None:
        status, details = _check_src_dir(minimal_repo_dir)
        assert status == PASS
        assert details == []

    def test_fail_when_missing(self, tmp_path: Path) -> None:
        status, details = _check_src_dir(tmp_path)
        assert status == FAIL
        assert "src/" in details[0]


class TestCheckImportable:
    def test_pass_on_success(self, tmp_path: Path) -> None:
        with patch("check_ml_component_contracts.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            status, details = _check_importable(tmp_path, "my_pkg")
        assert status == PASS
        assert details == []

    def test_fail_on_nonzero_exit(self, tmp_path: Path) -> None:
        with patch("check_ml_component_contracts.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stderr="ModuleNotFoundError: No module named 'my_pkg'"
            )
            status, details = _check_importable(tmp_path, "my_pkg")
        assert status == FAIL
        assert "not importable" in details[0].lower()

    def test_warn_on_timeout(self, tmp_path: Path) -> None:
        with patch(
            "check_ml_component_contracts.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="uv", timeout=60),
        ):
            status, details = _check_importable(tmp_path, "my_pkg")
        assert status == WARN
        assert "timed out" in details[0].lower()

    def test_warn_when_uv_missing(self, tmp_path: Path) -> None:
        with patch(
            "check_ml_component_contracts.subprocess.run",
            side_effect=FileNotFoundError("uv not found"),
        ):
            status, details = _check_importable(tmp_path, "my_pkg")
        assert status == WARN

    def test_warn_when_no_package_name(self, tmp_path: Path) -> None:
        status, details = _check_importable(tmp_path, "")
        assert status == WARN
        assert "package_name" in details[0]


class TestCheckLint:
    def test_pass_on_success(self, tmp_path: Path) -> None:
        with patch("check_ml_component_contracts.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            status, details = _check_lint(tmp_path)
        assert status == PASS

    def test_warn_on_nonzero_exit(self, tmp_path: Path) -> None:
        with patch("check_ml_component_contracts.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=2, stderr="ruff: command not found")
            status, details = _check_lint(tmp_path)
        assert status == WARN  # lint failures are WARN, not FAIL
        assert "make lint" in details[0]

    def test_warn_on_timeout(self, tmp_path: Path) -> None:
        with patch(
            "check_ml_component_contracts.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="make", timeout=120),
        ):
            status, details = _check_lint(tmp_path)
        assert status == WARN

    def test_warn_when_make_missing(self, tmp_path: Path) -> None:
        with patch(
            "check_ml_component_contracts.subprocess.run",
            side_effect=FileNotFoundError("make not found"),
        ):
            status, details = _check_lint(tmp_path)
        assert status == WARN


# ---------------------------------------------------------------------------
# Integration-level tests: check_repo and _load_library_repos
# ---------------------------------------------------------------------------


class TestLoadLibraryRepos:
    def test_filters_service_repos(self, service_repos_json: Path) -> None:
        repos = _load_library_repos(service_repos_json)
        assert repos == []

    def test_returns_library_repos(self, repos_json: Path) -> None:
        repos = _load_library_repos(repos_json)
        assert len(repos) == 1
        assert repos[0]["name"] == "ml-test-lib"


class TestCheckRepo:
    def test_all_pass_on_good_repo(self, minimal_repo_dir: Path) -> None:
        repo = {
            "name": "ml-good",
            "local_path": str(minimal_repo_dir),
            "package_name": "my_pkg",
        }
        with (
            patch("check_ml_component_contracts.subprocess.run") as mock_run,
            patch("check_ml_component_contracts._HUB_ROOT", minimal_repo_dir.parent),
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
            results = check_repo(repo, [repo])

        statuses = {r.name.split("/")[1]: r.status for r in results}
        assert statuses["dir"] == PASS
        assert statuses["pyproject"] == PASS
        assert statuses["src"] == PASS
        assert statuses["import"] == PASS
        assert statuses["lint"] == PASS  # mock returncode=0 → lint PASS

    def test_fail_stops_early_on_missing_dir(self, tmp_path: Path) -> None:
        missing = tmp_path / "no-such"
        repo = {
            "name": "ml-missing",
            "local_path": str(missing),
            "package_name": "my_pkg",
        }
        with patch("check_ml_component_contracts._HUB_ROOT", tmp_path):
            results = check_repo(repo, [repo])

        # Should only have the dir check (early exit)
        assert len(results) == 1
        assert results[0].status == FAIL

    def test_no_lint_skips_lint_check(self, minimal_repo_dir: Path) -> None:
        repo = {
            "name": "ml-skiplint",
            "local_path": str(minimal_repo_dir),
            "package_name": "",
        }
        with (
            patch("check_ml_component_contracts.subprocess.run") as mock_run,
            patch("check_ml_component_contracts._HUB_ROOT", minimal_repo_dir.parent),
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
            results = check_repo(repo, [repo], run_lint=False)

        # No result should have "/lint" as its check suffix
        check_suffixes = [r.name.split("/", 1)[1] for r in results]
        assert "lint" not in check_suffixes

    def test_warn_when_onnx_artifacts_missing(self, minimal_repo_dir: Path) -> None:
        repo = {
            "name": "ml-preproc",
            "local_path": str(minimal_repo_dir),
            "package_name": "",
            "onnx_boundary": {"role": "preprocessor"},
        }
        model_repo = {
            "name": "ml-model",
            "local_path": str(minimal_repo_dir),
            "onnx_boundary": {"role": "model"},
        }
        with (
            patch("check_ml_component_contracts.subprocess.run") as mock_run,
            patch("check_ml_component_contracts._HUB_ROOT", minimal_repo_dir.parent),
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
            results = check_repo(repo, [repo, model_repo], run_lint=False)

        boundary_results = [r for r in results if "onnx-boundary" in r.name]
        assert len(boundary_results) == 1
        assert boundary_results[0].status == WARN
        assert any("artifacts" in d.lower() for d in boundary_results[0].details)


# ---------------------------------------------------------------------------
# Print summary smoke test
# ---------------------------------------------------------------------------


class TestPrintSummary:
    def test_prints_without_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        results = [
            CheckResult(name="a/dir", status=PASS),
            CheckResult(name="b/import", status=WARN, details=["timed out"]),
            CheckResult(name="c/lint", status=FAIL, details=["lint error"]),
        ]
        _print_summary(results)
        captured = capsys.readouterr()
        assert "PASS" in captured.out
        assert "WARN" in captured.out
        assert "FAIL" in captured.out
        assert "Summary:" in captured.out


# ---------------------------------------------------------------------------
# Main entrypoint tests
# ---------------------------------------------------------------------------


class TestMain:
    def test_returns_0_on_all_pass(self, repos_json: Path, minimal_repo_dir: Path) -> None:
        with (
            patch("check_ml_component_contracts.subprocess.run") as mock_run,
            patch("check_ml_component_contracts._HUB_ROOT", minimal_repo_dir.parent),
            patch("sys.argv", ["check_ml_component_contracts.py", "--config", str(repos_json), "--no-lint"]),
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
            rc = main()
        assert rc == 0

    def test_returns_0_when_no_library_repos(self, service_repos_json: Path) -> None:
        with patch("sys.argv", ["x.py", "--config", str(service_repos_json)]):
            rc = main()
        assert rc == 0

    def test_returns_1_on_fail(self, tmp_path: Path) -> None:
        """A config pointing at a non-existent dir should exit 1."""
        config = {
            "repos": [
                {
                    "name": "ml-bad",
                    "url": "https://example.com/bad.git",
                    "ref": "main",
                    "local_path": str(tmp_path / "nonexistent"),
                    "component_type": "library",
                    "package_name": "bad_pkg",
                    "checks": [],
                }
            ],
            "contracts": [],
        }
        config_path = tmp_path / "repos.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        with (
            patch("sys.argv", ["x.py", "--config", str(config_path), "--no-lint"]),
            patch("check_ml_component_contracts._HUB_ROOT", tmp_path),
        ):
            rc = main()
        assert rc == 1

    def test_json_out_written(self, repos_json: Path, minimal_repo_dir: Path, tmp_path: Path) -> None:
        out_file = tmp_path / "results.json"
        with (
            patch("check_ml_component_contracts.subprocess.run") as mock_run,
            patch("check_ml_component_contracts._HUB_ROOT", minimal_repo_dir.parent),
            patch(
                "sys.argv",
                ["x.py", "--config", str(repos_json), "--no-lint", "--json", str(out_file)],
            ),
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
            main()
        assert out_file.exists()
        data = json.loads(out_file.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert all("name" in item and "status" in item for item in data)
