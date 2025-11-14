# tests/test_agent.py
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from janus.agent import StandardPlannerAgent


@pytest.fixture
def mock_tool_registry():
    registry = MagicMock()
    registry.get_schemas.return_value = [{"name": "get_weather", "parameters": {}}]
    registry.execute = AsyncMock(return_value="Sunny")
    return registry


@pytest.fixture
def agent():
    return StandardPlannerAgent()


@pytest.mark.asyncio
async def test_agent_execute(agent: StandardPlannerAgent, mock_tool_registry):
    agent._mock_llm_call = AsyncMock(
        return_value=json.dumps(
            {"tool_name": "get_weather", "parameters": {"location": "SF"}}
        )
    )
    state = {"messages": [{"role": "user", "content": "What's the weather in SF?"}]}
    result = await agent.execute(state, mock_tool_registry)

    assert result["new_message"]["role"] == "assistant"
    assert "Result: Sunny" in result["new_message"]["content"]
    mock_tool_registry.execute.assert_awaited_once_with(
        "get_weather", location="SF"
    )


@pytest.mark.asyncio
async def test_agent_llm_parse_error(agent: StandardPlannerAgent, mock_tool_registry):
    agent._mock_llm_call = AsyncMock(return_value="invalid json")
    state = {"messages": []}
    result = await agent.execute(state, mock_tool_registry)
    assert "error" in result


@pytest.mark.asyncio
async def test_agent_tool_execution_error(
    agent: StandardPlannerAgent, mock_tool_registry
):
    agent._mock_llm_call = AsyncMock(
        return_value=json.dumps(
            {"tool_name": "get_weather", "parameters": {"location": "SF"}}
        )
    )
    mock_tool_registry.execute.side_effect = Exception("Tool failed")
    state = {"messages": []}
    result = await agent.execute(state, mock_tool_registry)
    assert "Error executing tool" in result["new_message"]["content"]
