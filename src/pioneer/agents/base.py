"""ReAct-style agent with pluggable tools.

The tool-use loop interleaves reasoning traces and actions as described in Yao
et al., "ReAct: Synergizing Reasoning and Acting in Language Models"
(arXiv:2210.03629). ``Agent.run_self_consistent`` additionally implements the
self-consistency decoding strategy from Wang et al., "Self-Consistency Improves
Chain of Thought Reasoning in Language Models" (arXiv:2203.11171).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from pioneer.core.exceptions import AgentError
from pioneer.core.logging import get_logger
from pioneer.models.llm.base import LLMMessage, LLMRequest, LLMRole
from pioneer.models.llm.providers import LLMProvider

logger = get_logger(__name__)

_WHITESPACE = re.compile(r"\s+")


def majority_vote(answers: list[str]) -> str:
    """Return the most common answer under light normalization.

    Implements the aggregation step of self-consistency: sample several reasoning
    paths and marginalize by taking the majority final answer (Wang et al.,
    arXiv:2203.11171). Ties break toward the earliest sampled answer.
    """
    if not answers:
        raise AgentError("Cannot vote over an empty set of answers")

    normalized = [_WHITESPACE.sub(" ", a.strip().lower()) for a in answers]
    counts = Counter(normalized)
    winner, _ = max(counts.items(), key=lambda item: (item[1], -normalized.index(item[0])))
    return answers[normalized.index(winner)]


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
        lines.append(
            "Reason step by step. Prefix your reasoning with 'Thought:'. "
            "When you need a tool, emit a line 'TOOL: <name> | INPUT: <text>'. "
            "When you have the final answer, respond without a TOOL line."
        )
        return "\n".join(lines)

    async def _parse_and_execute(self, content: str) -> ToolResult | None:
        marker = content.find("TOOL:")
        if marker == -1:
            return None
        try:
            rest = content[marker + len("TOOL:") :]
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

    async def run(self, user_input: str, *, temperature: float | None = None) -> AgentResult:
        system_content = self.config.system_prompt
        tool_prompt = self._build_tool_prompt()
        if tool_prompt:
            system_content = f"{system_content}\n\n{tool_prompt}"

        messages = [
            LLMMessage(role=LLMRole.SYSTEM, content=system_content),
            LLMMessage(role=LLMRole.USER, content=user_input),
        ]
        tool_calls: list[ToolResult] = []
        sampling_temperature = temperature if temperature is not None else self.config.temperature

        for iteration in range(1, self.config.max_iterations + 1):
            request = LLMRequest(
                messages=messages,
                model=self.config.model,
                temperature=sampling_temperature,
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

    async def run_self_consistent(
        self,
        user_input: str,
        *,
        samples: int = 5,
        temperature: float = 0.7,
    ) -> AgentResult:
        """Run the agent several times and return the majority-vote answer.

        Samples multiple independent reasoning paths at non-zero temperature and
        marginalizes over them by majority vote, following self-consistency
        decoding (Wang et al., arXiv:2203.11171). The returned ``AgentResult`` is
        the first run whose answer matches the winning vote.
        """
        if samples < 1:
            raise AgentError("samples must be >= 1", details={"samples": samples})

        results: list[AgentResult] = []
        for _ in range(samples):
            results.append(await self.run(user_input, temperature=temperature))

        winning_answer = majority_vote([result.answer for result in results])
        logger.info("agent_self_consistency", samples=samples)
        for result in results:
            if result.answer == winning_answer:
                return result
        return results[0]
