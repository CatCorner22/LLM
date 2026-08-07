"""Intelligence API routes for risk, advisory, scenarios, and benchmarks."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from pioneer.intelligence.advisory.advisor import BusinessAdvisor
from pioneer.intelligence.benchmark.huggingface import (
    HFBenchmarkConfig,
    HFBenchmarkReport,
    HFBenchmarkRunner,
)
from pioneer.intelligence.benchmark.runner import BenchmarkReport, BenchmarkRunner
from pioneer.intelligence.ingestion.knowledge import KnowledgeCategory, KnowledgeRecord
from pioneer.intelligence.ingestion.scheduler import IngestionScheduler, KnowledgeStore
from pioneer.intelligence.relationships.map import KnowledgeRelationshipMap, RelationshipMapBuilder
from pioneer.intelligence.risk.composite import CompositeRiskEngine
from pioneer.intelligence.risk.models import AssetPortfolio, RiskAssessment
from pioneer.intelligence.samples import sample_portfolio
from pioneer.intelligence.scenarios.runner import AutonomousScenarioRunner, ScenarioRunResult

router = APIRouter(prefix="/v1/intelligence", tags=["intelligence"])


class PortfolioRequest(BaseModel):
    portfolio: AssetPortfolio | None = None
    use_sample: bool = True


class AdvisoryResponse(BaseModel):
    report: dict[str, Any]


class KnowledgeStatsResponse(BaseModel):
    store: dict[str, int] = Field(default_factory=dict)
    recent_chemicals: list[KnowledgeRecord] = Field(default_factory=list)
    recent_health: list[KnowledgeRecord] = Field(default_factory=list)
    recent_governance: list[KnowledgeRecord] = Field(default_factory=list)


@router.post("/risk", response_model=RiskAssessment)
async def assess_risk(request: PortfolioRequest) -> RiskAssessment:
    portfolio = request.portfolio if request.portfolio else sample_portfolio()
    return CompositeRiskEngine().assess(portfolio)


@router.post("/advisory")
async def advisory(request: PortfolioRequest) -> AdvisoryResponse:
    portfolio = request.portfolio if request.portfolio else sample_portfolio()
    report = BusinessAdvisor().generate_report(portfolio)
    return AdvisoryResponse(report=report.model_dump())


@router.post("/scenarios")
async def run_scenarios(request: PortfolioRequest) -> ScenarioRunResult:
    portfolio = request.portfolio if request.portfolio else sample_portfolio()
    return AutonomousScenarioRunner().run_autonomous_suite(portfolio)


@router.get("/benchmark", response_model=BenchmarkReport)
async def benchmark() -> BenchmarkReport:
    return BenchmarkRunner().run(sample_portfolio())


@router.get("/hf-benchmark")
async def hf_benchmark() -> HFBenchmarkReport:
    return HFBenchmarkRunner(HFBenchmarkConfig(mining_sample_size=200)).run()


@router.post("/ingestion/run")
async def run_ingestion() -> dict[str, int]:
    return await IngestionScheduler().run_once()


@router.get("/knowledge", response_model=KnowledgeStatsResponse)
async def knowledge_base(
    limit: int = Query(default=20, ge=1, le=200),
) -> KnowledgeStatsResponse:
    store = KnowledgeStore()
    store.seed_governance_knowledge()
    return KnowledgeStatsResponse(
        store=store.stats(),
        recent_chemicals=store.load_knowledge(limit, category=KnowledgeCategory.CHEMICAL),
        recent_health=store.load_knowledge(limit, category=KnowledgeCategory.HEALTH),
        recent_governance=store.load_knowledge(limit, category=KnowledgeCategory.EMPLOYEE_CONDUCT)
        + store.load_knowledge(limit, category=KnowledgeCategory.SEGREGATION_OF_DUTIES),
    )


@router.get("/knowledge/recent", response_model=list[KnowledgeRecord])
async def recent_knowledge(
    limit: int = Query(default=50, ge=1, le=500),
    category: KnowledgeCategory | None = None,
) -> list[KnowledgeRecord]:
    store = KnowledgeStore()
    if category is None:
        store.seed_governance_knowledge()
        return store.load_all_knowledge(limit)
    return store.load_knowledge(limit, category=category)


@router.get("/relationship-map", response_model=KnowledgeRelationshipMap)
async def relationship_map() -> KnowledgeRelationshipMap:
    store = KnowledgeStore()
    store.seed_governance_knowledge()
    records = store.load_all_knowledge(limit=100)
    assessment = CompositeRiskEngine().assess(sample_portfolio())
    portfolio = sample_portfolio()
    return RelationshipMapBuilder().build(records, assessment, portfolio)
