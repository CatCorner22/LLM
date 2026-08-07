"""Unit tests for the ReAct agent and self-consistency decoding."""

import pytest

from pioneer.agents.base import Agent, AgentConfig, Tool, ToolResult, majority_vote
from pioneer.models.llm.base import LLMResponse


class _ScriptedProvider:
    """Returns queued responses, repeating the last one when exhausted."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls = 0

    async def complete(self, _request: object) -> LLMResponse:
        content = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return LLMResponse(content=content, model="stub")


class _EchoTool(Tool):
    name = "echo"
    description = "Echo the input back."

    async def run(self, input_text: str) -> ToolResult:
        return ToolResult(tool_name=self.name, output=f"echoed:{input_text}")


def _config() -> AgentConfig:
    return AgentConfig(name="test", system_prompt="You are helpful.")


@pytest.mark.unit
def test_majority_vote() -> None:
    assert majority_vote(["4", "4", "5"]) == "4"
    # Tie resolves to the earliest sampled answer.
    assert majority_vote(["blue", "red", "red", "blue"]) == "blue"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_agent_direct_answer() -> None:
    agent = Agent(config=_config(), provider=_ScriptedProvider(["The answer is 42."]))  # type: ignore[arg-type]
    result = await agent.run("What is the answer?")
    assert result.answer == "The answer is 42."
    assert result.iterations == 1
    assert result.tool_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_agent_react_thought_then_tool() -> None:
    provider = _ScriptedProvider(
        [
            "Thought: I should echo the value.\nTOOL: echo | INPUT: hi",
            "Final answer: hi",
        ]
    )
    agent = Agent(config=_config(), provider=provider, tools=[_EchoTool()])  # type: ignore[arg-type]
    result = await agent.run("Please echo hi")
    assert result.iterations == 2
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].output == "echoed:hi"
    assert result.answer == "Final answer: hi"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_self_consistent_majority() -> None:
    provider = _ScriptedProvider(["4", "4", "5", "4", "5"])
    agent = Agent(config=_config(), provider=provider)  # type: ignore[arg-type]
    result = await agent.run_self_consistent("2 + 2 = ?", samples=5, temperature=0.7)
    assert provider.calls == 5
    assert result.answer == "4"
