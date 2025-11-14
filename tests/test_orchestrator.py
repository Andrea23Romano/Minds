# tests/test_orchestrator.py
from unittest.mock import AsyncMock, MagicMock

import pytest

from janus.orchestrator import AsyncLocalOrchestrator
from janus.models import ChatMessage, Role


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.execute = AsyncMock(
        return_value={
            "messages_to_add": [
                ChatMessage(role=Role.ASSISTANT, content="Test response")
            ]
        }
    )
    return agent


@pytest.fixture
def mock_memory():
    memory = MagicMock()
    memory.add = AsyncMock()
    memory.get_context = AsyncMock(
        return_value={"messages": [{"role": "user", "content": "Hello"}]}
    )
    return memory


@pytest.fixture
def mock_tool_registry():
    return MagicMock()


@pytest.fixture
def orchestrator(mock_agent, mock_tool_registry, mock_memory):
    return AsyncLocalOrchestrator(
        agent=mock_agent, tools=mock_tool_registry, memory=mock_memory
    )


@pytest.mark.asyncio
async def test_orchestrator_run(
    orchestrator: AsyncLocalOrchestrator, mock_agent, mock_memory
):
    initial_state = {"messages": [{"role": "user", "content": "Hello"}]}
    final_state = {}
    async for state in orchestrator.run(initial_state, max_steps=2):
        final_state = state

    assert mock_agent.execute.call_count == 2
    assert mock_memory.add.call_count == 3  # 1 initial + 2 new
    assert "messages" in final_state
