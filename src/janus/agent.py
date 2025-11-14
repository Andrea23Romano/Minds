# src/janus/agent.py
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

from janus.memory import BaseMemory
from janus.models import Message
from janus.tool import ToolRegistry

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for agents.
    """

    @abstractmethod
    async def execute(
        self, state: Dict[str, Any], tools: "ToolRegistry"
    ) -> Dict[str, Any]:
        """
        Executes the agent's logic.
        """
        pass


class StandardPlannerAgent(BaseAgent):
    """
    A simple planner agent that uses a mock LLM to call tools.
    """

    async def _mock_llm_call(self, prompt: str) -> str:
        """
        A mock LLM call that returns a hard-coded tool call.
        """
        logger.info(f"Mock LLM call with prompt:\n{prompt}")
        # In a real scenario, this would be a JSON object or similar
        return '{"tool_name": "get_weather", "parameters": {"location": "SF"}}'

    async def execute(
        self, state: Dict[str, Any], tools: "ToolRegistry"
    ) -> Dict[str, Any]:
        """
        Executes a planning step: prompt -> LLM -> tool -> result.
        """
        # 1. Render prompt
        # A real implementation would use a templating engine
        prompt = f"""
Current state: {state}
Available tools: {tools.get_schemas()}
What is the next step?
"""
        # 2. Call mock LLM
        llm_response_str = await self._mock_llm_call(prompt)

        # 3. Parse LLM response
        try:
            llm_response = json.loads(llm_response_str)
            tool_name = llm_response["tool_name"]
            parameters = llm_response["parameters"]
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {"error": str(e)}

        # 4. Execute the tool
        try:
            tool_result = await tools.execute(tool_name, **parameters)
            new_message = Message(
                role="assistant",
                content=f"Tool {tool_name} executed successfully. Result: {tool_result}",
            )
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            new_message = Message(
                role="assistant", content=f"Error executing tool {tool_name}: {e}"
            )

        # 5. Return new state
        # In this simple case, we just append the result as a new message.
        # A more complex agent might update state in other ways.
        return {"new_message": new_message.model_dump()}
