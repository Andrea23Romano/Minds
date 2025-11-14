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
    return StandardPlannerAgent(api_key="test-key")


@pytest.mark.asyncio
async def test__agent_execute(agent: StandardPlannerAgent, mock_tool_registry):
    mock_function = MagicMock()
    mock_function.name = "get_weather"
    mock_function.arguments = json.dumps({"location": "SF"})
    mock_tool_call = MagicMock()
    mock_tool_call.function = mock_function
    agent._call_openai_api = AsyncMock(
        return_value=MagicMock(
            tool_calls=[mock_tool_call]
        )
    )
    state = {"messages": [{"role": "user", "content": "What's the weather in SF?"}]}
    result = await agent.execute(state, mock_tool_registry)
    assert result["messages_to_add"][1].role == "tool"
    assert "Sunny" in result["messages_to_add"][1].content
    mock_tool_registry.execute.assert_awaited_once_with(
        "get_weather", location="SF"
    )


@pytest.mark.asyncio
async def test_agent_llm_parse_error(agent: StandardPlannerAgent, mock_tool_registry):
    mock_function = MagicMock()
    mock_function.name = "get_weather"
    mock_function.arguments = "invalid json"
    mock_tool_call = MagicMock()
    mock_tool_call.function = mock_function
    agent._call_openai_api = AsyncMock(
        return_value=MagicMock(
            tool_calls=[mock_tool_call]
        )
    )
    state = {"messages": []}
    result = await agent.execute(state, mock_tool_registry)
    assert "Error parsing arguments" in result["messages_to_add"][1].content


@pytest.mark.asyncio
async def test_agent_tool_execution_error(
    agent: StandardPlannerAgent, mock_tool_registry
):
    mock_function = MagicMock()
    mock_function.name = "get_weather"
    mock_function.arguments = json.dumps({"location": "SF"})
    mock_tool_call = MagicMock()
    mock_tool_call.function = mock_function
    agent._call_openai_api = AsyncMock(
        return_value=MagicMock(
            tool_calls=[mock_tool_call]
        )
    )
    mock_tool_registry.execute.side_effect = Exception("Tool failed")
    state = {"messages": []}
    result = await agent.execute(state, mock_tool_registry)
    assert "Error executing tool" in result["messages_to_add"][1].content
