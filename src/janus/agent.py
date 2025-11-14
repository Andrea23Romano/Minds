# src/janus/agent.py
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from janus.models import ChatMessage, Role
from janus.tool import ToolRegistry
from openai import AsyncOpenAI

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
    A simple planner agent that uses a live OpenAI client to call tools.
    """

    def __init__(self, api_key: str, system_prompt: str = "You are a helpful assistant."):
        self.client = AsyncOpenAI(api_key=api_key)
        self.system_prompt = ChatMessage(role=Role.SYSTEM, content=system_prompt)

    async def _call_openai_api(
        self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]
    ) -> ChatMessage:
        """
        Calls the OpenAI API with the given messages and tools.
        """
        formatted_messages = [self.system_prompt.model_dump()] + messages
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=formatted_messages,
            tools=tools,
        )
        response_message = response.choices[0].message
        if response_message.tool_calls:
            return ChatMessage(
                role=Role.ASSISTANT,
                tool_calls=response_message.tool_calls,
            )
        return ChatMessage(role=Role.ASSISTANT, content=response_message.content)

    async def execute(
        self, state: Dict[str, Any], tools: "ToolRegistry"
    ) -> Dict[str, Any]:
        """
        Executes a planning step: prompt -> LLM -> tool -> result.
        """
        messages = state.get("messages", [])
        tool_schemas = tools.get_schemas()
        llm_response_message = await self._call_openai_api(messages, tool_schemas)
        messages_to_add = [llm_response_message]
        if llm_response_message.tool_calls:
            for tool_call in llm_response_message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                    tool_result = await tools.execute(tool_name, **tool_args)
                    messages_to_add.append(
                        ChatMessage(role=Role.TOOL, content=str(tool_result))
                    )
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse tool arguments: {e}")
                    messages_to_add.append(
                        ChatMessage(
                            role=Role.TOOL,
                            content=f"Error parsing arguments for tool {tool_name}: {e}",
                        )
                    )
                except Exception as e:
                    logger.error(f"Tool execution failed: {e}")
                    messages_to_add.append(
                        ChatMessage(
                            role=Role.TOOL,
                            content=f"Error executing tool {tool_name}: {e}",
                        )
                    )
        return {"messages_to_add": messages_to_add}
