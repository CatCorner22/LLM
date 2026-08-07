"""ReAct-style agent with pluggable tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from pioneer.core.exceptions import AgentError
from pioneer.core.logging import get_logger
from pioneer.models.llm.base import LLMMessage, LLMRequest, LLMRole
from pioneer.models.llm.providers import LLMProvider

logger = get_logger(__name__)


class AgentConfig(BaseModel):
    """Agent runtime configuration."""

    name: str
    system_prompt: str
    max_iterations: int = Field(default=10, ge=1, le=50)
    model: str | None = None
    temperature: float = 0.0


@dataclass
class ToolResult:
    tool_name: str
    output: str
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    answer: str
    iterations: int
    tool_calls: list[ToolResult] = field(default_factory=list)
    messages: list[LLMMessage] = field(default_factory=list)


class Tool(ABC):
    """Pluggable agent tool."""

    name: str
    description: str

    @abstractmethod
    async def run(self, input_text: str) -> ToolResult:
        """Execute the tool with natural language input."""


class Agent:
    """LLM agent with tool-use loop."""

    def __init__(
        self,
        config: AgentConfig,
        provider: LLMProvider,
        tools: list[Tool] | None = None,
    ) -> None:
        self.config = config
        self.provider = provider
        self.tools = {tool.name: tool for tool in (tools or [])}

    def _build_tool_prompt(self) -> str:
        if not self.tools:
            return ""
        lines = ["Available tools:"]
        for tool in self.tools.values():
            lines.append(f"- {tool.name}: {tool.description}")
        lines.append("To use a tool, respond with: TOOL: <name> | INPUT: <text>")
        return "\n".join(lines)

    async def _parse_and_execute(self, content: str) -> ToolResult | None:
        if not content.startswith("TOOL:"):
            return None
        try:
            _, rest = content.split("TOOL:", 1)
            tool_part, input_part = rest.split("| INPUT:", 1)
            tool_name = tool_part.strip()
            input_text = input_part.strip()
        except ValueError as exc:
            raise AgentError("Malformed tool call", details={"content": content}) from exc

        tool = self.tools.get(tool_name)
        if tool is None:
            return ToolResult(
                tool_name=tool_name,
                output=f"Unknown tool: {tool_name}",
                success=False,
            )
        return await tool.run(input_text)

    async def run(self, user_input: str) -> AgentResult:
        system_content = self.config.system_prompt
        tool_prompt = self._build_tool_prompt()
        if tool_prompt:
            system_content = f"{system_content}\n\n{tool_prompt}"

        messages = [
            LLMMessage(role=LLMRole.SYSTEM, content=system_content),
            LLMMessage(role=LLMRole.USER, content=user_input),
        ]
        tool_calls: list[ToolResult] = []

        for iteration in range(1, self.config.max_iterations + 1):
            request = LLMRequest(
                messages=messages,
                model=self.config.model,
                temperature=self.config.temperature,
            )
            response = await self.provider.complete(request)
            assistant_message = LLMMessage(role=LLMRole.ASSISTANT, content=response.content)
            messages.append(assistant_message)

            tool_result = await self._parse_and_execute(response.content)
            if tool_result is None:
                logger.info("agent_completed", iterations=iteration)
                return AgentResult(
                    answer=response.content,
                    iterations=iteration,
                    tool_calls=tool_calls,
                    messages=messages,
                )

            tool_calls.append(tool_result)
            messages.append(
                LLMMessage(
                    role=LLMRole.TOOL,
                    content=tool_result.output,
                    name=tool_result.tool_name,
                )
            )

        raise AgentError(
            "Agent exceeded maximum iterations",
            details={"max_iterations": self.config.max_iterations},
        )
