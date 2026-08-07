# Pioneer ML

Commercial-grade foundation for advanced LLM and ML applications with **Pioneer Intelligence** — an autonomous risk advisory platform for business owners.

## Pioneer Intelligence (NEW)

Autonomous multi-factor risk assessment, scenario testing, news ingestion, and competitor benchmarking.

```
src/pioneer/intelligence/
├── risk/           # Building, weather, pipe-age, accident likelihood models
├── ingestion/      # RSS/news feeds with risk-keyword relevance scoring
├── scenarios/      # Autonomous what-if scenario stress testing
├── advisory/       # Diverse business-owner recommendations
└── benchmark/      # Head-to-head vs legacy competitor baselines
```

### Risk factors

| Factor | Model | Output |
|--------|-------|--------|
| Building risk | Age, condition, occupancy, seismic, inspections | Weighted severity score |
| Weather | Flood, wind, storms, heat/freeze forecasts | Regional disruption risk |
| Pipe age | Weibull failure by material (cast iron, PVC, steel…) | Failure probability |
| Accident/injury | Logistic regression over operational features | 12-month likelihood |

### Commands

```bash
pip install -e ".[dev]"

# Full advisory report with recommendations
pioneer-advise

# Autonomous scenario stress-test (6 what-if scenarios)
pioneer-scenarios

# Benchmark vs HeuristicRules, IndustryAverage, ManualConsultant
pioneer-benchmark

# API endpoints (requires [serve])
pioneer-serve
# POST /v1/intelligence/risk
# POST /v1/intelligence/advisory
# POST /v1/intelligence/scenarios
# GET  /v1/intelligence/benchmark
```

### Competitor benchmark dimensions

Pioneer is measured against three legacy-style baselines:

- **HeuristicRules v1** — static rule checklist (typical legacy SaaS)
- **IndustryAverage Report** — generic consulting averages
- **ManualConsultant Review** — annual manual assessment simulation

Benchmark metrics: risk discrimination, recommendation diversity, scenario coverage, response latency, accident model availability.

### Hugging Face Hub benchmark

Evaluate Pioneer on public HF datasets with `pioneer-hf-benchmark`:

| Task | HF Dataset | Metric |
|------|------------|--------|
| Injury severity prediction | [electricsheepafrica/africa-synth-mining-safety-incidents-all](https://huggingface.co/datasets/electricsheepafrica/africa-synth-mining-safety-incidents-all) | F1 (high injury) |
| Risk severity ranking | Same mining dataset | Spearman correlation |
| Severity calibration | Fatal vs first-aid incidents | Score separation gap |
| News risk relevance | Curated English headlines | F1 vs keyword baseline |

```bash
pip install -e ".[huggingface]"
pioneer-hf-benchmark
pioneer-hf-benchmark --mining-samples 1000 --json
# Optional HF zero-shot model comparison:
pioneer-hf-benchmark --with-hf-model --model facebook/bart-large-mnli
```

API: `GET /v1/intelligence/hf-benchmark`

### Knowledge base (chemical inventory + CDC ER visits)

Populate the intelligence knowledge base from public datasets:

| Source | Dataset | Stored as |
|--------|---------|-----------|
| EPA TRI | [TRI_CHEM_INFO](https://www.epa.gov/toxics-release-inventory-tri-program) chemical metadata | `data/intelligence/chemical_inventory.jsonl` |
| CDC NCHS | [Emergency Department Visits 2016–2022](https://data.cdc.gov/NCHS/Estimates-of-Emergency-Department-Visits-in-the-Un/ycxr-emue) | `data/intelligence/cdc_er_visits.jsonl` |

```bash
# Ingest EPA carcinogen inventory + CDC injury/poisoning ER statistics
pioneer-ingest
pioneer-ingest --chemical-limit 500 --er-limit 200 --json

# API (requires [serve])
# POST /v1/intelligence/ingestion/run
# GET  /v1/intelligence/knowledge
# GET  /v1/intelligence/knowledge/recent?category=health
```

Advisory reports automatically include `chemical_signals` and `health_signals` from the knowledge base.

## Architecture

```
src/pioneer/
├── core/           # Config, logging, exceptions, plugin registry
├── intelligence/   # Risk, ingestion, scenarios, advisory, benchmark
├── data/           # Dataset loading and validation
├── models/         # Model lifecycle + LLM providers
├── training/       # Training orchestrator and callbacks
├── inference/      # Batch inference engine
├── evaluation/     # Metrics and benchmarks
├── agents/         # ReAct-style agent framework
├── rag/            # Retrieval-augmented generation
├── observability/  # Telemetry and metrics
├── serving/        # FastAPI production API
└── cli/            # Train, serve, eval, advise, scenarios, benchmark
```

## Quick Start

```bash
pip install -e ".[dev]"
make ci
pioneer-benchmark
pioneer-advise --json
```

## Configuration

Settings load from environment variables (prefix `PIONEER_`) and optional YAML:

```bash
cp .env.example .env
```

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

## Docker

```bash
docker compose up api
```

## Development

```bash
make install-dev
make ci
make license-check
pre-commit install
```

## License

Pioneer ML is released under the [MIT License](LICENSE).

Third-party dependency licenses: [licenses/THIRD_PARTY_LICENSES.md](licenses/THIRD_PARTY_LICENSES.md)

See [docs/OPEN_SOURCE.md](docs/OPEN_SOURCE.md) for the open source policy.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
