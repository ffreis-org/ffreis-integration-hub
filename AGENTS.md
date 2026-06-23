# Agent Context

**This repo:** `ffreis-integration-hub` — cross-repo integration and parity testing
harness. Validates that Python serving, Rust serving, and the stock simulator maintain
compatible API contracts; and contract-checks all `ffreis-ml-*` library components.

## Non-obvious facts

- **Source of truth for cross-repo API contracts.** `config/repos.json` lists all repos
  and their required make targets. If a repo's interface changes, this hub detects it.

- **Two component types in `config/repos.json`:**
  - `"service"` — gRPC services (converter, python-serving, rust-serving). Must expose
    `make grpc-check`, `make test-grpc-parity`, `make smoke-api-grpc`.
  - `"library"` — Python library components (`ffreis-ml-*`). Must expose `make lint`,
    `make test`, have `pyproject.toml`, and a `src/` directory.

- **All gRPC service repos MUST expose:** `make grpc-check`, `make test-grpc-parity`,
  `make smoke-api-grpc`. Removing these targets from a serving repo will break this
  hub's CI.

- **gRPC reflection is intentionally disabled** across all repos for security/performance.
  Do not enable it in runtime code. Integration tests use direct proto registration
  instead.

- **Discovers sibling repos via `local_path` in `config/repos.json`** — does not clone
  repos in local mode. Paths are relative to the real hub root
  (`/media/ffreis/second/projects/ml/ffreis-integration-hub`), not a worktree path.

- **Weekly CI jobs** orchestrate: converter→serving parity bench, stock sim dashboard
  compatibility, RL agent schema compatibility. `check-ml-components` is also called
  as part of `weekly-check-local`.

- **ONNX boundary check** (`scripts/check_feature_contract.py`) is a static-only
  check (no onnxruntime). It compares `preprocessing.onnx` outputs ↔ `policy.onnx`
  inputs: names, dtypes, shapes, opsets. Artifacts must be generated first:
  `uv run crypto-env preprocess` (in ffreis-ml-crypto-env) and
  `uv run ml_crypto_rl export` (in ffreis-ml-crypto-rl).

- **`pyproject.toml` added at hub root (P4.1).** This adds `onnx` as a runtime
  dependency and pins the dev toolchain (ruff, mypy, pytest). All scripts/ are now
  linted and type-checked. Run `uv sync --dev` after checkout.

## Structure

```
config/repos.json                            ← registry of sibling repos + contracts
scripts/                                     ← parity checks, orchestration, benchmarking
  check_ml_component_contracts.py            ← library-component contract checker (P4.1)
  check_feature_contract.py                  ← ONNX preprocessing↔model boundary (P4.1)
tests/                                       ← pytest suite for hub scripts (P4.1)
examples/                                    ← docker-compose stacks for end-to-end scenarios
pyproject.toml                               ← Python project + ruff/mypy/pytest config (P4.1)
```

## Build/run

```bash
# Library component checks (P4.1)
make check-ml-components                     # dir/pyproject/src/import/lint for all ffreis-ml-* repos
make check-feature-contract ARGS="--preprocessing artifacts/preprocessing.onnx --model artifacts/policy.onnx"

# Unit tests + coverage
uv run pytest                                # 55+ tests, ≥80% coverage on new scripts

# gRPC serving / stock-sim checks (pre-existing)
make weekly-check-local                      # includes check-ml-components
make smoke-converter-serving-parity
make smoke-stock-sim-dashboard
make stock-sim-full-stack
```

## Cross-repo dependencies

- gRPC services: `ml/ffreis-python-model-serving`, `ml/ffreis-rust-onnx-model-serving`,
  `ml/ffreis-python-onnx-model-converter`, `stock/ffreis-stock-simulator`
- ML library components: `ml/ffreis-ml-{core,train,tracking,registry,data,features,hpo,
  pipelines,monitoring,examples,crypto-env,crypto-rl}`

## Keeping this file current

- **If you discover a fact not reflected here:** add it before finishing your task.
- **If something here is wrong or outdated:** correct it in the same commit as the code change.
- **If you rename a file, command, or concept referenced here:** update the reference.
