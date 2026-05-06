.DEFAULT_GOAL := help
SHELL := /usr/bin/env bash

MODE ?= container
SCENARIO ?= all

.PHONY: help
help: ## Show help
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: env
env: ## Create local virtual environment for harness
	uv venv .venv
	@echo "Activate with: . .venv/bin/activate"

.PHONY: install
install: ## Install harness dependencies
	uv sync

.PHONY: compare
compare: ## Run full comparison suite (set MODE=container|native)
	env -u VIRTUAL_ENV uv run --project . python -m orchestrator.main --mode "$(MODE)" --scenario "$(SCENARIO)" --checks parity,property,perf

.PHONY: compare-container
compare-container: ## Run comparison in container mode
	$(MAKE) compare MODE=container

.PHONY: compare-native
compare-native: ## Run comparison in native mode
	$(MAKE) compare MODE=native

.PHONY: compare-native-sepal
compare-native-sepal: ## Run native comparison for the sepal-sum scenario
	$(MAKE) compare MODE=native SCENARIO=sepal-sum

.PHONY: compare-native-triple
compare-native-triple: ## Run native 3-way comparison (python-sklearn, python-onnx, rust-onnx)
	$(MAKE) compare MODE=native SCENARIO=sepal-sum

.PHONY: compare-native-raw-all
compare-native-raw-all: ## Run native 5-way comparison (python onnx/sklearn/pytorch/tensorflow + rust onnx)
	$(MAKE) compare MODE=native SCENARIO=raw-all-frameworks
