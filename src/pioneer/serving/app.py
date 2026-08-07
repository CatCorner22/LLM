"""FastAPI serving application."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from pioneer.__version__ import __version__
from pioneer.core.config import get_settings
from pioneer.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = __version__
    environment: str


class PredictRequest(BaseModel):
    input: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class PredictResponse(BaseModel):
    output: str
    metadata: dict[str, Any] = Field(default_factory=dict)


from pioneer.serving.intelligence import router as intelligence_router


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()

    app = FastAPI(
        title="Pioneer ML",
        description="Commercial-grade LLM and ML serving API",
        version=__version__,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(environment=settings.env)

    @app.post("/v1/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest) -> PredictResponse:
        logger.info("predict_request", input_length=len(request.input))
        return PredictResponse(output=f"echo: {request.input}", metadata=request.metadata)

    app.include_router(intelligence_router)

    return app
