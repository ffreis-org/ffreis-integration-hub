.DEFAULT_GOAL := help
SHELL := /usr/bin/env bash

IMAGE_PROVIDER ?= localhost
IMAGE_PREFIX ?= ffreis
IMAGE_TAG ?= integration
IMAGE_ROOT := $(if $(IMAGE_PROVIDER),$(IMAGE_PROVIDER)/,)$(IMAGE_PREFIX)
BENCH_DIR ?= benchmarks/onnx-runner-comparison
COMPARE_REPO_DIR ?= ../ffreis-onnx-runner-comparison
COMPARE_REPORT ?= artifacts/compare-native-sepal-report.json

GITLEAKS         ?= gitleaks
LEFTHOOK_VERSION ?= 1.7.10
LEFTHOOK_DIR     ?= $(CURDIR)/.bin
LEFTHOOK_BIN     ?= $(LEFTHOOK_DIR)/lefthook

.PHONY: help
help:
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "\033[36m%-26s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ------------------------------------------------------------------------------
# Standard interface targets
# ------------------------------------------------------------------------------

.PHONY: fmt
fmt: ## Format Python scripts in place (ruff format)
	uv run ruff format scripts

.PHONY: lint
lint: ## Run lint/type checks on scripts
	uv run ruff check scripts
	uv run ruff format --check scripts
	uv run mypy scripts

.PHONY: test
test: ## Run integration smoke checks against local sibling repos
	$(MAKE) weekly-check-local

.PHONY: validate
validate: ## Static type checking (mypy)
	uv run mypy scripts

.PHONY: plan
plan: ## Not applicable — use 'make validate' or 'make test' for this repo
	@echo "INFO: 'plan' is Terraform-specific and does not apply to this repo."
	@echo "      To type-check: make validate"
	@echo "      To run smoke checks: make test"

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

.PHONY: secrets-scan-staged lefthook-bootstrap lefthook-install lefthook-run lefthook

secrets-scan-staged: ## Scan staged diff for secrets
	@command -v $(GITLEAKS) >/dev/null 2>&1 || (echo "Missing tool: $(GITLEAKS). Install: https://github.com/gitleaks/gitleaks#installing" && exit 1)
	$(GITLEAKS) protect --staged --redact

lefthook-bootstrap: ## Download lefthook binary into ./.bin
	LEFTHOOK_VERSION="$(LEFTHOOK_VERSION)" BIN_DIR="$(LEFTHOOK_DIR)" bash ./scripts/bootstrap_lefthook.sh

lefthook-install: lefthook-bootstrap ## Install git hooks (runs bootstrap first)
	@if [ -x "$(LEFTHOOK_BIN)" ] && [ -x ".git/hooks/pre-commit" ] && [ -x ".git/hooks/pre-push" ] && [ -x ".git/hooks/commit-msg" ]; then \
		echo "lefthook hooks already installed"; \
		exit 0; \
	fi
	LEFTHOOK="$(LEFTHOOK_BIN)" "$(LEFTHOOK_BIN)" install

lefthook-run: lefthook-bootstrap ## Run all hooks locally (pre-commit + commit-msg + pre-push)
	LEFTHOOK="$(LEFTHOOK_BIN)" "$(LEFTHOOK_BIN)" run pre-commit
	@tmp_msg="$$(mktemp)"; \
	echo "chore(hooks): validate commit-msg hook" > "$$tmp_msg"; \
	LEFTHOOK="$(LEFTHOOK_BIN)" "$(LEFTHOOK_BIN)" run commit-msg -- "$$tmp_msg"; \
	rm -f "$$tmp_msg"
	LEFTHOOK="$(LEFTHOOK_BIN)" "$(LEFTHOOK_BIN)" run pre-push

lefthook: lefthook-bootstrap lefthook-install lefthook-run ## Install hooks and run them

PLATFORM_STANDARDS_SHA ?= 3c787edb4e96ddea2e86b2add2c32139685e8db7  # v1.2.1
PLATFORM_STANDARDS_RAW ?= https://raw.githubusercontent.com/FelipeFuhr/ffreis-platform-standards

install-act: ## Download pinned act binary into .bin/
	@mkdir -p scripts
	@curl -fsSL "$(PLATFORM_STANDARDS_RAW)/$(PLATFORM_STANDARDS_SHA)/scripts/install_act.sh" \
		-o scripts/install_act.sh && chmod +x scripts/install_act.sh
	@bash ./scripts/install_act.sh

ci-local: ## Run workflows locally via act (GH Actions quota fallback). Args via ARGS=...
	@mkdir -p scripts
	@curl -fsSL "$(PLATFORM_STANDARDS_RAW)/$(PLATFORM_STANDARDS_SHA)/scripts/run-ci-local.sh" \
		-o scripts/run-ci-local.sh && chmod +x scripts/run-ci-local.sh
	@PATH="$(CURDIR)/.bin:$(PATH)" bash ./scripts/run-ci-local.sh $(ARGS)
