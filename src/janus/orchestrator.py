# src/janus/orchestrator.py
import logging
from typing import Any, Dict

from janus.agent import BaseAgent
from janus.memory import BaseMemory
from janus.models import ChatMessage
from janus.tool import ToolRegistry

logger = logging.getLogger(__name__)


class AsyncLocalOrchestrator:
    """
    Manages the in-memory execution of an agentic graph.
    """

    def __init__(
        self,
        agent: BaseAgent,
        tools: ToolRegistry,
        memory: BaseMemory,
    ):
        self.agent = agent
        self.tools = tools
        self.memory = memory

    async def run(
        self, initial_state: Dict[str, Any], max_steps: int = 5
    ):
        """
        Runs the agentic graph.
        """
        state = initial_state.copy()

        # Add initial messages to memory
        if "messages" in state:
            for msg_data in state["messages"]:
                await self.memory.add(ChatMessage(**msg_data))

        for step in range(max_steps):
            logger.info(f"Orchestrator Step {step + 1}/{max_steps}")

            # Get current context from memory
            memory_context = await self.memory.get_context()
            current_state = {**state, **memory_context}

            # Execute the agent
            agent_output = await self.agent.execute(current_state, self.tools)

            # Update state and memory
            if "messages_to_add" in agent_output:
                for new_msg in agent_output["messages_to_add"]:
                    await self.memory.add(new_msg)

            yield await self.memory.get_context()


        logger.info("Orchestration finished.")
