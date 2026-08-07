# Pioneer ML

Commercial-grade foundation for advanced LLM and ML applications. Pioneer ML provides a typed, modular Python codebase with production-ready patterns for training, inference, RAG, agents, evaluation, and observability.

## Architecture

```
src/pioneer/
├── core/           # Config, logging, exceptions, plugin registry
├── data/           # Dataset loading and validation
├── models/         # Model lifecycle + LLM providers
├── training/       # Training orchestrator and callbacks
├── inference/      # Batch inference engine
├── evaluation/     # Metrics and benchmarks
├── agents/         # ReAct-style agent framework
├── rag/            # Retrieval-augmented generation
├── observability/  # Telemetry and metrics
├── serving/        # FastAPI production API
└── cli/            # Train, serve, and eval CLIs
```

## Quick Start

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run quality checks
make ci

# Start training (baseline loop)
pioneer-train --experiment baseline --epochs 3

# Start API server
pip install -e ".[serve]"
pioneer-serve

# Run evaluation
pioneer-eval --metric exact_match --predictions "hello" --references "hello"
```

## Configuration

Settings load from environment variables (prefix `PIONEER_`) and optional YAML:

```bash
cp .env.example .env
# Edit .env with your API keys and paths
```

See `configs/default.yaml` for the full schema.

## Optional Extras

| Extra | Purpose |
|-------|---------|
| `llm` | OpenAI-compatible LLM providers |
| `training` | PyTorch, Transformers, PEFT, TRL |
| `rag` | ChromaDB + sentence-transformers |
| `serve` | FastAPI + Uvicorn |
| `observability` | OpenTelemetry + Prometheus |
| `dev` | pytest, ruff, mypy, pre-commit |
| `all` | Everything |

```bash
pip install -e ".[all]"
```

## Docker

```bash
# Production API
docker compose up api

# Development shell
docker compose --profile dev run dev
```

## Development

```bash
make install-dev    # Install with dev deps
make lint           # Ruff lint
make typecheck      # Mypy strict mode
make test           # Full test suite
make coverage       # Coverage report (80% threshold)
pre-commit install  # Git hooks
```

## Design Principles

- **Typed & validated** — Pydantic settings, strict mypy, structured errors
- **Extensible** — Registry pattern for loaders, metrics, LLM providers
- **Observable** — Structured logging (structlog), telemetry hooks
- **Resilient** — Retries on external calls, health checks, CI security audit
- **Modular** — Optional dependency groups; use only what you need

## License

Pioneer ML is released under the [MIT License](LICENSE).

Third-party dependency licenses are documented in
[licenses/THIRD_PARTY_LICENSES.md](licenses/THIRD_PARTY_LICENSES.md).
See [docs/OPEN_SOURCE.md](docs/OPEN_SOURCE.md) for the open source policy.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). By contributing, you agree to license
your contributions under MIT.
