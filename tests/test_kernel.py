import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock

from framework.core.kernel import MeshKernel
from framework.core.base_agent import BaseAgent

class MockAgent(BaseAgent):
    async def think(self, message: dict):
        pass

class TestMeshKernel(unittest.TestCase):

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.kernel = MeshKernel()

    def tearDown(self):
        self.loop.close()

    def test_register_agent(self):
        agent = MockAgent(self.kernel, "test_agent")
        self.kernel.register_agent(agent)
        self.assertIn("test_agent", self.kernel.agents)
        self.assertEqual(self.kernel.agents["test_agent"], agent)

    def test_register_duplicate_agent_raises_error(self):
        agent = MockAgent(self.kernel, "test_agent")
        self.kernel.register_agent(agent)
        with self.assertRaises(ValueError):
            self.kernel.register_agent(agent)

    def test_deregister_agent(self):
        agent = MockAgent(self.kernel, "test_agent")
        self.kernel.register_agent(agent)
        self.loop.run_until_complete(self.kernel.start_single_agent("test_agent"))
        self.assertIn("test_agent", self.kernel.agent_tasks)

        self.loop.run_until_complete(self.kernel.deregister_agent("test_agent"))
        self.assertNotIn("test_agent", self.kernel.agents)
        self.assertNotIn("test_agent", self.kernel.agent_tasks)

    def test_route_message(self):
        sender_agent = MockAgent(self.kernel, "sender")
        receiver_agent = MockAgent(self.kernel, "receiver")
        self.kernel.register_agent(sender_agent)
        self.kernel.register_agent(receiver_agent)

        content = {"test": "message"}
        self.loop.run_until_complete(self.kernel.route_message("sender", "receiver", content))

        message = self.loop.run_until_complete(receiver_agent.inbox.get())
        self.assertEqual(message["sender_id"], "sender")
        self.assertEqual(message["content"], content)

    def test_log_metric(self):
        metric = {"name": "test_metric", "value": 1}
        self.loop.run_until_complete(self.kernel.log_metric("test_agent", metric))
        logged_metric = self.loop.run_until_complete(self.kernel.metrics_bus.get())
        self.assertEqual(logged_metric["agent_id"], "test_agent")
        self.assertEqual(logged_metric["metric"], metric)

    def test_start_and_stop(self):
        agent1 = MockAgent(self.kernel, "agent1")
        agent2 = MockAgent(self.kernel, "agent2")
        self.kernel.register_agent(agent1)
        self.kernel.register_agent(agent2)

        self.loop.run_until_complete(self.kernel.start())
        tasks = list(self.kernel.agent_tasks.values())
        self.assertEqual(len(tasks), 2)
        self.assertIn("agent1", self.kernel.agent_tasks)
        self.assertIn("agent2", self.kernel.agent_tasks)

        self.kernel.stop()

        # Wait for all tasks to complete. Since they handle the cancellation
        # gracefully, they will just finish.
        async def wait_for_tasks():
            await asyncio.gather(*tasks, return_exceptions=True)

        self.loop.run_until_complete(wait_for_tasks())

        for task in tasks:
            self.assertTrue(task.done())


    def test_update_agent_prompt(self):
        from framework.core.agents import LLMAgent
        agent = LLMAgent(self.kernel, "llm_agent", "old_prompt")
        self.kernel.register_agent(agent)
        new_prompt = "new_prompt"
        self.kernel.update_agent_prompt("llm_agent", new_prompt)
        self.assertEqual(agent.prompt, new_prompt)

if __name__ == '__main__':
    unittest.main()
