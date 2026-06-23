#!/usr/bin/env python3
"""Cross-repository parity checks runner."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RepoSpec:
    name: str
    url: str
    ref: str
    local_path: str | None
    checks: list[str]


@dataclass(frozen=True)
class ContractSpec:
    name: str
    repos: list[str]
    targets: list[str]


def _run(
    command: str,
    *,
    cwd: Path,
    log_file: Path,
    env: dict[str, str] | None = None,
) -> tuple[int, float]:
    start = time.time()
    with log_file.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            shlex.split(command),
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
        rc = proc.wait()
    return rc, time.time() - start


def _clone_or_update(
    spec: RepoSpec,
    *,
    workdir: Path,
    use_local_repos: bool,
    token: str | None,
) -> Path:
    if use_local_repos and spec.local_path:
        return (Path(__file__).resolve().parent.parent / spec.local_path).resolve()

    repo_dir = workdir / spec.name
    if not repo_dir.exists():
        clone_url = spec.url
        if token and clone_url.startswith("https://"):
            clone_url = clone_url.replace("https://", f"https://x-access-token:{token}@", 1)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", spec.ref, clone_url, str(repo_dir)],
            check=True,
        )
        return repo_dir

    subprocess.run(["git", "fetch", "origin", spec.ref], cwd=repo_dir, check=True)
    subprocess.run(["git", "checkout", spec.ref], cwd=repo_dir, check=True)
    subprocess.run(["git", "reset", "--hard", f"origin/{spec.ref}"], cwd=repo_dir, check=True)
    return repo_dir


def _read_make_targets(repo_dir: Path) -> set[str]:
    targets: set[str] = set()
    for makefile in [repo_dir / "Makefile", repo_dir / "app" / "Makefile"]:
        if not makefile.exists():
            continue
        content = makefile.read_text(encoding="utf-8")
        for line in content.splitlines():
            if line.startswith(".PHONY:"):
                names = line.replace(".PHONY:", "").strip().split()
                targets.update(names)
                continue
            match = re.match(r"^([A-Za-z0-9_.-]+):", line)
            if match:
                targets.add(match.group(1))
    return targets


def _load_specs(config_path: Path) -> tuple[list[RepoSpec], list[ContractSpec]]:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    repos = [
        RepoSpec(
            name=item["name"],
            url=item["url"],
            ref=item.get("ref", "main"),
            local_path=item.get("local_path"),
            checks=item.get("checks", []),
        )
        for item in raw["repos"]
    ]
    contracts = [
        ContractSpec(
            name=item["name"],
            repos=item["repos"],
            targets=item["targets"],
        )
        for item in raw.get("contracts", [])
    ]
    return repos, contracts


def main() -> int:
    parser = argparse.ArgumentParser(description="Run weekly cross-repo parity checks.")
    parser.add_argument(
        "--config",
        default="config/repos.json",
        help="Path to integration config JSON.",
    )
    parser.add_argument(
        "--workdir",
        default=".work/repos",
        help="Working directory for cloned repositories.",
    )
    parser.add_argument(
        "--artifacts",
        default="artifacts",
        help="Directory for logs and summary output.",
    )
    parser.add_argument(
        "--use-local-repos",
        action="store_true",
        help="Use local_path entries in config instead of cloning.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    config_path = (root / args.config).resolve()
    workdir = (root / args.workdir).resolve()
    artifacts = (root / args.artifacts).resolve()
    shim_dir = (root / "scripts" / "shims").resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    repos, contracts = _load_specs(config_path)
    repo_paths: dict[str, Path] = {}
    summary: dict[str, Any] = {
        "started_at": int(time.time()),
        "repos": {},
        "contracts": {},
        "ok": True,
    }

    token = os.getenv("INTEGRATION_REPO_TOKEN")
    base_env = os.environ.copy()
    base_env["PATH"] = f"{shim_dir}:{base_env.get('PATH', '')}"

    for spec in repos:
        repo_dir = _clone_or_update(
            spec,
            workdir=workdir,
            use_local_repos=args.use_local_repos,
            token=token,
        )
        repo_paths[spec.name] = repo_dir
        summary["repos"][spec.name] = {
            "path": str(repo_dir),
            "checks": [],
            "ok": True,
        }

        for idx, command in enumerate(spec.checks):
            log_file = artifacts / f"{spec.name}-{idx + 1}.log"
            rc, seconds = _run(command, cwd=repo_dir, log_file=log_file, env=base_env)
            ok = rc == 0
            summary["repos"][spec.name]["checks"].append(
                {
                    "command": command,
                    "ok": ok,
                    "returncode": rc,
                    "seconds": round(seconds, 2),
                    "log_file": str(log_file),
                }
            )
            if not ok:
                summary["repos"][spec.name]["ok"] = False
                summary["ok"] = False

    for contract in contracts:
        contract_ok = True
        missing_targets: dict[str, list[str]] = {}
        for repo_name in contract.repos:
            repo_dir = repo_paths[repo_name]
            targets = _read_make_targets(repo_dir)
            missing = [target for target in contract.targets if target not in targets]
            if missing:
                missing_targets[repo_name] = missing
                contract_ok = False
        summary["contracts"][contract.name] = {
            "ok": contract_ok,
            "missing_targets": missing_targets,
        }
        if not contract_ok:
            summary["ok"] = False

    summary["finished_at"] = int(time.time())
    summary_path = artifacts / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
