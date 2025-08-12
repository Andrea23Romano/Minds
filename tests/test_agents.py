import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

from framework.core.agents import HumanInterfaceAgent, LLMAgent, FunctionAgent
from framework.core.kernel import MeshKernel

class TestAgents(unittest.TestCase):

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.kernel = AsyncMock(spec=MeshKernel)

    def tearDown(self):
        self.loop.close()

    def test_human_interface_agent(self):
        agent = HumanInterfaceAgent(self.kernel, "human")
        agent.send_message = AsyncMock() # Mock send_message
        message = {"content": {"new_goal_description": "test_goal"}}
        self.loop.run_until_complete(agent.think(message))
        agent.send_message.assert_called_once_with(
            "architect", {"type": "SET_NEW_GOAL", "goal": "test_goal"}
        )

    @patch('framework.core.agents.invoke_agent_llm', new_callable=AsyncMock)
    def test_llm_agent_sends_message(self, mock_invoke_llm):
        mock_invoke_llm.return_value = '{"actions": [{"type": "SEND_MESSAGE", "target": "test_target", "content": "test_content"}]}'

        agent = LLMAgent(self.kernel, "llm_agent", "test_prompt")
        agent.send_message = AsyncMock() # Mock the send_message method
        message = {"content": "test_message"}
        self.loop.run_until_complete(agent.think(message))

        agent.send_message.assert_called_once_with(
            "test_target", "test_content"
        )
        mock_invoke_llm.assert_awaited_once()

    @patch('framework.core.agents.invoke_agent_llm', new_callable=AsyncMock)
    def test_llm_agent_logs_metric(self, mock_invoke_llm):
        mock_invoke_llm.return_value = '{"actions": [{"type": "LOG_METRIC", "metric": {"name": "test_metric", "value": 1}}]}'

        agent = LLMAgent(self.kernel, "llm_agent", "test_prompt")
        # The agent calls kernel.log_metric directly
        message = {"content": "test_message"}
        self.loop.run_until_complete(agent.think(message))

        self.kernel.log_metric.assert_called_once_with(
            "llm_agent", {"name": "test_metric", "value": 1}
        )

    def test_function_agent(self):
        def test_func(content):
            return {"target": "test_target", "content": f"processed: {content}"}

        agent = FunctionAgent(self.kernel, "func_agent", test_func)
        agent.send_message = AsyncMock() # Mock the send_message method
        message = {"content": "test_input"}
        self.loop.run_until_complete(agent.think(message))

        agent.send_message.assert_called_once_with(
            "test_target", "processed: test_input"
        )

if __name__ == '__main__':
    unittest.main()
