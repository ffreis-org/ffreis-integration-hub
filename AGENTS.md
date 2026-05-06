# Agent Context

**This repo:** `ffreis-integration-hub` — cross-repo integration and parity testing
harness. Validates that Python serving, Rust serving, and the stock simulator maintain
compatible API contracts.

## Non-obvious facts

- **Source of truth for cross-repo API contracts.** `config/repos.json` lists all repos
  and their required make targets. If a repo's interface changes, this hub detects it.

- **All sibling repos MUST expose:** `make grpc-check`, `make test-grpc-parity`,
  `make smoke-api-grpc`. Removing these targets from a serving repo will break this
  hub's CI.

- **gRPC reflection is intentionally disabled** across all repos for security/performance.
  Do not enable it in runtime code. Integration tests use direct proto registration
  instead.

- **Discovers sibling repos via `local_path` in `config/repos.json`** — does not clone
  repos in local mode. Paths must be relative to the workspace root.

- **Weekly CI jobs** orchestrate: converter→serving parity bench, stock sim dashboard
  compatibility, RL agent schema compatibility.

## Structure

```
config/repos.json        ← registry of sibling repos + contracts
scripts/                 ← parity checks, orchestration, benchmarking
examples/                ← docker-compose stacks for end-to-end scenarios
```

## Build/run

```bash
make weekly-check-local                  # uses local paths
make smoke-converter-serving-parity
make smoke-stock-sim-dashboard
make stock-sim-full-stack
```

## Cross-repo dependencies

All of: `ml/ffreis-python-model-serving`, `ml/ffreis-rust-onnx-model-serving`,
`ml/ffreis-python-onnx-model-converter`, `stock/ffreis-stock-simulator`

## Keeping this file current

- **If you discover a fact not reflected here:** add it before finishing your task.
- **If something here is wrong or outdated:** correct it in the same commit as the code change.
- **If you rename a file, command, or concept referenced here:** update the reference.
