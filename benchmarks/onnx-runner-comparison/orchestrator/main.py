"""Main entrypoint for ONNX runner comparison."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path
from statistics import mean
from typing import Any

import httpx
import yaml

from orchestrator.startup import ModeRunner
from orchestrator.wait_ready import wait_http_ok


def _invoke(
    *,
    base_url: str,
    path: str,
    payload: bytes,
    content_type: str,
    accept: str,
) -> httpx.Response:
    headers = {"Content-Type": content_type, "Accept": accept}
    return httpx.post(f"{base_url}{path}", content=payload, headers=headers, timeout=10.0)


def _p95_ms(latencies_ms: list[float]) -> float:
    ordered = sorted(latencies_ms)
    if not ordered:
        return 0.0
    idx = int(round(0.95 * (len(ordered) - 1)))
    return ordered[idx]


def _canonicalize_prediction(value: Any) -> Any:
    """Normalize numeric prediction payloads for parity comparison."""
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, list):
        normalized = [_canonicalize_prediction(item) for item in value]
        if normalized and all(
            isinstance(item, list) and len(item) == 1 for item in normalized
        ):
            return [item[0] for item in normalized]
        return normalized
    if isinstance(value, dict):
        return {k: _canonicalize_prediction(v) for k, v in sorted(value.items())}
    return value


def _measure_latency(
    *,
    base_url: str,
    path: str,
    payload: bytes,
    content_type: str,
    accept: str,
    warmup_requests: int,
    measured_requests: int,
) -> dict[str, float]:
    for _ in range(warmup_requests):
        response = _invoke(
            base_url=base_url,
            path=path,
            payload=payload,
            content_type=content_type,
            accept=accept,
        )
        response.raise_for_status()

    latencies_ms: list[float] = []
    start_all = time.perf_counter()
    for _ in range(measured_requests):
        start = time.perf_counter()
        response = _invoke(
            base_url=base_url,
            path=path,
            payload=payload,
            content_type=content_type,
            accept=accept,
        )
        response.raise_for_status()
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    elapsed_all = time.perf_counter() - start_all
    rps = measured_requests / elapsed_all if elapsed_all > 0 else 0.0
    return {
        "mean_ms": mean(latencies_ms),
        "p95_ms": _p95_ms(latencies_ms),
        "rps": rps,
    }


def _run_parity(
    *,
    service_bases: dict[str, str],
    parity_services: list[str] | None,
    path: str,
    payload: bytes,
    content_type: str,
    accept: str,
) -> None:
    if parity_services is None:
        ordered_services = sorted(service_bases.keys())
    else:
        ordered_services = sorted(parity_services)
        unknown = [name for name in ordered_services if name not in service_bases]
        if unknown:
            raise RuntimeError(f"Unknown parity service(s): {unknown}")
    if not ordered_services:
        raise RuntimeError("No services configured for parity")
    baseline_name = ordered_services[0]
    baseline_response = _invoke(
        base_url=service_bases[baseline_name],
        path=path,
        payload=payload,
        content_type=content_type,
        accept=accept,
    )
    baseline_response.raise_for_status()
    baseline_json = _canonicalize_prediction(baseline_response.json())
    for service_name in ordered_services[1:]:
        candidate = _invoke(
            base_url=service_bases[service_name],
            path=path,
            payload=payload,
            content_type=content_type,
            accept=accept,
        )
        candidate.raise_for_status()
        candidate_json = _canonicalize_prediction(candidate.json())
        if candidate_json != baseline_json:
            raise RuntimeError(
                "Parity mismatch between services: "
                f"{baseline_name}={baseline_json} {service_name}={candidate_json}"
            )


def _load_scenarios(root: Path, selected: str) -> list[dict[str, object]]:
    scenarios_root = root / "scenarios"
    discovered = sorted(
        [p for p in scenarios_root.iterdir() if p.is_dir() and (p / "scenario.yaml").exists()],
        key=lambda p: p.name,
    )
    if not discovered:
        raise RuntimeError(f"No scenarios found in {scenarios_root}")

    scenario_map: dict[str, dict[str, object]] = {}
    for folder in discovered:
        spec = yaml.safe_load((folder / "scenario.yaml").read_text(encoding="utf-8"))
        if not isinstance(spec, dict):
            raise RuntimeError(f"Invalid scenario.yaml in {folder}")
        if bool(spec.get("enabled", True)) is False:
            continue
        spec["id"] = folder.name
        spec["_folder"] = folder
        scenario_map[folder.name] = spec

    if selected == "all":
        return [scenario_map[k] for k in sorted(scenario_map.keys())]

    selected_ids = [s.strip() for s in selected.split(",") if s.strip()]
    missing = [s for s in selected_ids if s not in scenario_map]
    if missing:
        raise RuntimeError(f"Unknown scenario(s): {missing}. Available: {sorted(scenario_map.keys())}")
    return [scenario_map[s] for s in selected_ids]


def _prepare_scenario_model(hub_root: Path, scenario: dict[str, object]) -> None:
    folder = Path(scenario["_folder"])
    model_cfg = scenario.get("model", {})
    if not isinstance(model_cfg, dict):
        return

    prepare = model_cfg.get("prepare")
    if isinstance(prepare, dict):
        cmd = prepare.get("cmd")
        cwd = prepare.get("cwd")
        if not isinstance(cmd, list) or not isinstance(cwd, str):
            raise RuntimeError(f"Invalid model.prepare in {folder / 'scenario.yaml'}")
        subprocess.check_call(cmd, cwd=hub_root / cwd)
        return

    source = model_cfg.get("source")
    if isinstance(source, str):
        src = folder / source
        runtime_path = Path(
            str(model_cfg.get("runtime_path", "/tmp/onnx-runner-comparison/model.onnx"))
        )
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, runtime_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Python vs Rust ONNX runners")
    parser.add_argument("--mode", choices=["container", "native"], required=True)
    parser.add_argument("--checks", default="parity,property,perf")
    parser.add_argument(
        "--scenario",
        default="all",
        help="Scenario id or comma-separated list from benchmarks/onnx-runner-comparison/scenarios",
    )
    args = parser.parse_args()

    hub_root = Path(__file__).resolve().parents[3]
    root = hub_root / "benchmarks" / "onnx-runner-comparison"
    selected_scenarios = _load_scenarios(root, args.scenario)

    checks = {c.strip() for c in args.checks.split(",") if c.strip()}
    all_results: list[dict[str, object]] = []

    for scenario in selected_scenarios:
        scenario_id = str(scenario["id"])
        _prepare_scenario_model(hub_root, scenario)

        request_cfg = scenario.get("request", {})
        if not isinstance(request_cfg, dict):
            raise RuntimeError(f"Invalid request section for scenario={scenario_id}")
        path = str(request_cfg.get("path", "/invocations"))
        health_path = str(request_cfg.get("health_path", "/healthz"))
        content_type = str(request_cfg.get("content_type", "text/csv"))
        accept = str(request_cfg.get("accept", "application/json"))
        payload_file = request_cfg.get("payload_file")
        if not isinstance(payload_file, str):
            raise RuntimeError(f"Scenario={scenario_id} must define request.payload_file")
        payload = (Path(scenario["_folder"]) / payload_file).read_bytes()
        compare_cfg = scenario.get("compare", {})
        if not isinstance(compare_cfg, dict):
            compare_cfg = {}

        active_services = None
        if isinstance(compare_cfg.get("active_services"), list):
            active_services = {str(name) for name in compare_cfg["active_services"]}

        with ModeRunner(
            hub_root=hub_root,
            mode=args.mode,
            active_services=active_services,
        ) as runner:
            raw_services = runner.config["services"]
            service_bases = {
                str(name): str(value["base_url"])
                for name, value in raw_services.items()
                if active_services is None or str(name) in active_services
            }
            for service_name, base_url in service_bases.items():
                wait_http_ok(f"{base_url}{health_path}")

            scenario_result: dict[str, object] = {"scenario": scenario_id}

            if "parity" in checks:
                _run_parity(
                    service_bases=service_bases,
                    parity_services=(
                        list(compare_cfg["parity_services"])
                        if isinstance(compare_cfg.get("parity_services"), list)
                        else None
                    ),
                    path=path,
                    payload=payload,
                    content_type=content_type,
                    accept=accept,
                )
                print(f"[parity] scenario={scenario_id} ok")
                scenario_result["parity"] = "ok"

            if "property" in checks:
                print(
                    f"[property] scenario={scenario_id} scaffold ready at workloads/hypothesis/parity_props.py"
                )
                scenario_result["property"] = "scaffold"

            if "perf" in checks:
                warmup_requests = int(compare_cfg.get("warmup_requests", 5))
                measured_requests = int(compare_cfg.get("measured_requests", 40))
                max_p95_ratio = float(compare_cfg.get("max_p95_ratio", 2.0))
                max_mean_ratio = float(compare_cfg.get("max_mean_ratio", 2.0))
                baseline_service = str(
                    compare_cfg.get("baseline_service", sorted(service_bases.keys())[0])
                )
                if baseline_service not in service_bases:
                    raise RuntimeError(
                        f"Scenario={scenario_id} baseline_service={baseline_service} is not configured"
                    )

                perf_stats: dict[str, dict[str, float]] = {}
                for service_name, base_url in sorted(service_bases.items()):
                    perf_stats[service_name] = _measure_latency(
                        base_url=base_url,
                        path=path,
                        payload=payload,
                        content_type=content_type,
                        accept=accept,
                        warmup_requests=warmup_requests,
                        measured_requests=measured_requests,
                    )

                baseline_stats = perf_stats[baseline_service]
                ratio_summary: dict[str, dict[str, float]] = {}
                for service_name, stats in perf_stats.items():
                    if service_name == baseline_service:
                        continue
                    p95_ratio = (
                        stats["p95_ms"] / baseline_stats["p95_ms"]
                        if baseline_stats["p95_ms"]
                        else 1.0
                    )
                    mean_ratio = (
                        stats["mean_ms"] / baseline_stats["mean_ms"]
                        if baseline_stats["mean_ms"]
                        else 1.0
                    )
                    if p95_ratio > max_p95_ratio:
                        raise RuntimeError(
                            f"Scenario={scenario_id} service={service_name} failed p95 ratio: "
                            f"{p95_ratio:.2f} > {max_p95_ratio:.2f}"
                        )
                    if mean_ratio > max_mean_ratio:
                        raise RuntimeError(
                            f"Scenario={scenario_id} service={service_name} failed mean ratio: "
                            f"{mean_ratio:.2f} > {max_mean_ratio:.2f}"
                        )
                    ratio_summary[service_name] = {
                        "ratio_mean": mean_ratio,
                        "ratio_p95": p95_ratio,
                    }

                print(f"[perf] scenario={scenario_id} baseline={baseline_service}")
                for service_name in sorted(perf_stats.keys()):
                    stats = perf_stats[service_name]
                    print(
                        "  - %s mean=%.2fms p95=%.2fms rps=%.2f"
                        % (service_name, stats["mean_ms"], stats["p95_ms"], stats["rps"])
                    )
                for service_name in sorted(ratio_summary.keys()):
                    ratios = ratio_summary[service_name]
                    print(
                        "  - ratio(%s/%s) mean=%.2f p95=%.2f"
                        % (
                            service_name,
                            baseline_service,
                            ratios["ratio_mean"],
                            ratios["ratio_p95"],
                        )
                    )
                scenario_result["perf"] = {
                    "baseline_service": baseline_service,
                    "services": perf_stats,
                    "ratios": ratio_summary,
                }
            all_results.append(scenario_result)

    print("[summary]", all_results)


if __name__ == "__main__":
    main()
