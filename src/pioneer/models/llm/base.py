"""LLM message and request/response types."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class LLMRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class LLMMessage(BaseModel):
    role: LLMRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None


class LLMRequest(BaseModel):
    messages: list[LLMMessage]
    model: str | None = None
    temperature: float = 0.7
    max_tokens: int = 4096
    stop: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    content: str
    model: str
    finish_reason: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)
