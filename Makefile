.DEFAULT_GOAL := help
SHELL := /usr/bin/env bash

IMAGE_PROVIDER ?= localhost
IMAGE_PREFIX ?= ffreis
IMAGE_TAG ?= integration
IMAGE_ROOT := $(if $(IMAGE_PROVIDER),$(IMAGE_PROVIDER)/,)$(IMAGE_PREFIX)
BENCH_DIR ?= benchmarks/onnx-runner-comparison
COMPARE_REPO_DIR ?= ../ffreis-onnx-runner-comparison
COMPARE_REPORT ?= artifacts/compare-native-sepal-report.json

.PHONY: help
help:
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "\033[36m%-26s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: weekly-check
weekly-check: ## Clone/update configured repos and run all parity checks
	./scripts/weekly_check.py

.PHONY: weekly-check-local
weekly-check-local: ## Run checks against local sibling repos
	./scripts/weekly_check.py --use-local-repos

.PHONY: print-summary
print-summary: ## Print latest JSON summary
	@cat artifacts/summary.json

.PHONY: typing-debt-report
typing-debt-report: ## Report Python typing debt (Any/object) across sibling repos
	./scripts/check_python_typing_debt.py --json-out artifacts/typing-debt.json

.PHONY: smoke-converter-serving-parity
smoke-converter-serving-parity: ## Convert via converter API, then benchmark Python vs Rust serving parity
	-IMAGE_ROOT="$(IMAGE_ROOT)" IMAGE_TAG="$(IMAGE_TAG)" ./scripts/compose.sh -f examples/docker-compose.converter-serving-parity.yml down --remove-orphans
	IMAGE_ROOT="$(IMAGE_ROOT)" IMAGE_TAG="$(IMAGE_TAG)" ./scripts/compose.sh -f examples/docker-compose.converter-serving-parity.yml up --build --abort-on-container-exit --exit-code-from bench

.PHONY: smoke-converter-serving-parity-grpc
smoke-converter-serving-parity-grpc: ## Convert via converter gRPC, then benchmark Python vs Rust serving gRPC parity
	-IMAGE_ROOT="$(IMAGE_ROOT)" IMAGE_TAG="$(IMAGE_TAG)" ./scripts/compose.sh -f examples/docker-compose.converter-serving-parity-grpc.yml down --remove-orphans
	IMAGE_ROOT="$(IMAGE_ROOT)" IMAGE_TAG="$(IMAGE_TAG)" ./scripts/compose.sh -f examples/docker-compose.converter-serving-parity-grpc.yml up --build --abort-on-container-exit --exit-code-from bench-grpc

.PHONY: smoke-stock-sim-dashboard
smoke-stock-sim-dashboard: ## Validate compatibility between stock simulator and Go dashboard
	@test -d "../ffreis-stock-simulator" || (echo "Missing repo at ../ffreis-stock-simulator"; exit 1)
	@test -d "../ffreis-stock-simulator-dashboard-go" || (echo "Missing repo at ../ffreis-stock-simulator-dashboard-go"; exit 1)
	@set -euo pipefail; \
	cleanup() { ./scripts/compose.sh -f ../ffreis-stock-simulator-dashboard-go/docker-compose.yml down --remove-orphans || true; }; \
	trap cleanup EXIT; \
	./scripts/compose.sh -f ../ffreis-stock-simulator-dashboard-go/docker-compose.yml up --build -d; \
	DASHBOARD_BASE_URL="http://127.0.0.1:18080" ./scripts/check_stock_dashboard_compat.py

.PHONY: smoke-stock-sim-agent-compat
smoke-stock-sim-agent-compat: ## Validate stock simulator event/replay schema compatibility with RL agent expectations
	@test -d "../ffreis-stock-simulator" || (echo "Missing repo at ../ffreis-stock-simulator"; exit 1)
	@test -d "../ffreis-stock-simulator-dashboard-go" || (echo "Missing repo at ../ffreis-stock-simulator-dashboard-go"; exit 1)
	@test -d "../ffreis-stock-rl-agent" || (echo "Missing repo at ../ffreis-stock-rl-agent"; exit 1)
	@set -euo pipefail; \
	cleanup() { ./scripts/compose.sh -f ../ffreis-stock-simulator-dashboard-go/docker-compose.yml down --remove-orphans || true; }; \
	trap cleanup EXIT; \
	./scripts/compose.sh -f ../ffreis-stock-simulator-dashboard-go/docker-compose.yml up --build -d; \
	SIMULATOR_BASE_URL="http://127.0.0.1:18000" ./scripts/check_stock_sim_agent_compat.py

.PHONY: stock-sim-full-stack
stock-sim-full-stack: ## Run simulator + dashboard + rl-agent + experiment-runner in one compose stack
	@test -d "../ffreis-stock-simulator" || (echo "Missing repo at ../ffreis-stock-simulator"; exit 1)
	@test -d "../ffreis-stock-simulator-dashboard-go" || (echo "Missing repo at ../ffreis-stock-simulator-dashboard-go"; exit 1)
	@test -d "../ffreis-stock-rl-agent" || (echo "Missing repo at ../ffreis-stock-rl-agent"; exit 1)
	-./scripts/compose.sh -f examples/docker-compose.stock-sim-dashboard-agent-experiments.yml down --remove-orphans
	./scripts/compose.sh -f examples/docker-compose.stock-sim-dashboard-agent-experiments.yml up --build --abort-on-container-exit --exit-code-from experiment-runner

.PHONY: compare-container
compare-container: ## Run ONNX runner comparison harness in container mode
	cd $(BENCH_DIR) && $(MAKE) compare-container

.PHONY: compare-native
compare-native: ## Run ONNX runner comparison harness in native process mode
	cd $(BENCH_DIR) && $(MAKE) compare-native

.PHONY: compare-native-triple
compare-native-triple: ## Run native 3-way ONNX runner comparison (python onnx/sklearn + rust onnx)
	cd $(BENCH_DIR) && $(MAKE) compare-native-triple

.PHONY: compare-native-raw-all
compare-native-raw-all: ## Run native 5-way comparison (python onnx/sklearn/pytorch/tensorflow + rust onnx)
	cd $(BENCH_DIR) && $(MAKE) compare-native-raw-all

.PHONY: compare-all
compare-all: ## Run both container and native ONNX runner comparison modes
	cd $(BENCH_DIR) && $(MAKE) compare-container && $(MAKE) compare-native

.PHONY: compare-repo-native
compare-repo-native: ## Run standalone comparison repo in native mode and validate JSON report
	@test -d "$(COMPARE_REPO_DIR)" || (echo "Missing repo at $(COMPARE_REPO_DIR)"; exit 1)
	$(MAKE) -C "$(COMPARE_REPO_DIR)" install
	$(MAKE) -C "$(COMPARE_REPO_DIR)" report MODE=native SCENARIO=sepal-sum REPORT="$(COMPARE_REPORT)"
	@test -s "$(COMPARE_REPO_DIR)/$(COMPARE_REPORT)" || (echo "Missing report: $(COMPARE_REPO_DIR)/$(COMPARE_REPORT)"; exit 1)
	mkdir -p artifacts
	cp -f "$(COMPARE_REPO_DIR)/$(COMPARE_REPORT)" artifacts/standalone-comparison-report.json
