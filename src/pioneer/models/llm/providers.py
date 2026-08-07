"""LLM provider implementations with retries and structured errors."""

from __future__ import annotations

from abc import ABC, abstractmethod

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from pioneer.core.config import Settings, get_settings
from pioneer.core.exceptions import ExternalServiceError, ValidationError
from pioneer.core.logging import get_logger
from pioneer.core.registry import Registry
from pioneer.models.llm.base import LLMRequest, LLMResponse

logger = get_logger(__name__)


class LLMProvider(ABC):
    """Abstract LLM provider interface."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate a completion for the given request."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for input texts."""


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible API provider."""

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        api_key = self.settings.openai_api_key
        if api_key is None:
            raise ValidationError(
                "OpenAI API key required",
                details={"env_var": "PIONEER_OPENAI_API_KEY"},
            )
        self._client = httpx.AsyncClient(
            base_url=self.settings.openai_base_url or "https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {api_key.get_secret_value()}"},
            timeout=self.settings.request_timeout_seconds,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def complete(self, request: LLMRequest) -> LLMResponse:
        model = request.model or self.settings.default_model
        payload = {
            "model": model,
            "messages": [message.model_dump(exclude_none=True) for message in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.stop:
            payload["stop"] = request.stop

        try:
            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("llm_request_failed", provider="openai", model=model, error=str(exc))
            raise ExternalServiceError(
                "LLM request failed",
                details={"provider": "openai", "model": model},
            ) from exc

        data = response.json()
        choice = data["choices"][0]
        return LLMResponse(
            content=choice["message"]["content"],
            model=data.get("model", model),
            finish_reason=choice.get("finish_reason"),
            usage=data.get("usage", {}),
            raw=data,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await self._client.post(
                "/embeddings",
                json={"model": "text-embedding-3-small", "input": texts},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ExternalServiceError(
                "Embedding request failed",
                details={"provider": "openai"},
            ) from exc

        data = response.json()
        return [item["embedding"] for item in data["data"]]

    async def aclose(self) -> None:
        await self._client.aclose()


LLM_PROVIDER_REGISTRY: Registry[type[LLMProvider]] = Registry("llm_provider")
LLM_PROVIDER_REGISTRY.register("openai", OpenAIProvider)


def get_llm_provider(name: str = "openai", settings: Settings | None = None) -> LLMProvider:
    """Instantiate a registered LLM provider."""
    provider_cls = LLM_PROVIDER_REGISTRY.get(name)
    return provider_cls(settings)
