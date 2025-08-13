import pytest
from unittest.mock import MagicMock

from minds.kernel.kernel import Kernel
from minds.core.models import Message
from minds.agents.base import BaseAgent

def test_agent_registration(mocker):
    """
    Tests that an agent can be registered with the kernel.
    """
    kernel = Kernel()
    mock_agent = mocker.MagicMock(spec=BaseAgent)
    mock_agent.agent_id = "mock_agent_123"

    kernel.register_agent(mock_agent)

    assert "mock_agent_123" in kernel.agents
    assert kernel.agents["mock_agent_123"] is mock_agent

def test_register_duplicate_agent_raises_error(mocker):
    """
    Tests that registering an agent with a duplicate ID raises a ValueError.
    """
    kernel = Kernel()
    mock_agent = mocker.MagicMock(spec=BaseAgent)
    mock_agent.agent_id = "duplicate_id"
    kernel.register_agent(mock_agent)

    with pytest.raises(ValueError):
        # Registering another agent with the same ID should fail
        another_mock_agent = mocker.MagicMock(spec=BaseAgent)
        another_mock_agent.agent_id = "duplicate_id"
        kernel.register_agent(another_mock_agent)


def test_message_dispatch_and_processing(mocker):
    """
    Tests that the kernel correctly dispatches a message to the target agent.
    """
    kernel = Kernel()

    # Create a mock agent and register it
    mock_agent = mocker.MagicMock(spec=BaseAgent)
    mock_agent.agent_id = "test_agent"
    # The handle_message method should return None or an empty list for this test
    mock_agent.handle_message.return_value = None
    kernel.register_agent(mock_agent)

    # Create a message for the agent
    message = Message(
        source_agent_id="source_agent",
        target_agent_id="test_agent",
        content="Hello, world!"
    )

    # Dispatch the message and run the kernel
    kernel.dispatch_message(message)
    kernel.run() # run() will exit after the queue is empty

    # Assert that the agent's handle_message was called
    mock_agent.handle_message.assert_called_once_with(message)

def test_kernel_handles_message_for_unknown_agent(mocker, caplog):
    """
    Tests that the kernel logs a warning and does not crash if a message
    is sent to a non-existent agent.
    """
    kernel = Kernel()
    message = Message(
        source_agent_id="source_agent",
        target_agent_id="non_existent_agent",
        content="This is a test."
    )

    kernel.dispatch_message(message)
    kernel.run()

    # Check that a warning was logged
    assert "No agent found with ID non_existent_agent" in caplog.text
    assert "Message dropped" in caplog.text
