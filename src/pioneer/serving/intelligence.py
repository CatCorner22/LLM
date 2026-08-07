"""Intelligence API routes for risk, advisory, scenarios, and benchmarks."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from pioneer.intelligence.advisory.advisor import BusinessAdvisor
from pioneer.intelligence.benchmark.huggingface import HFBenchmarkReport
from pioneer.intelligence.benchmark.runner import BenchmarkReport, BenchmarkRunner
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
    from pioneer.intelligence.benchmark.huggingface import HFBenchmarkConfig, HFBenchmarkRunner

    return HFBenchmarkRunner(HFBenchmarkConfig(mining_sample_size=200)).run()
