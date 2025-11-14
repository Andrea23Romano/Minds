# src/janus/tool.py
import logging
from typing import Any, Callable, Coroutine, Dict, List, Type

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def agent_tool(args_schema: Type[BaseModel]):
    """
    Decorator to register a function as an agent tool.
    Attaches the Pydantic schema to the function.
    """

    def decorator(
        func: Callable[..., Coroutine[Any, Any, Any]]
    ) -> Callable[..., Coroutine[Any, Any, Any]]:
        setattr(func, "_pydantic_schema", args_schema)
        return func

    return decorator


class ToolRegistry:
    """
    A simple registry for agent tools.
    """

    def __init__(self):
        self._tools: Dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}

    def register(self, tool_func: Callable[..., Coroutine[Any, Any, Any]]):
        """
        Registers a tool function.
        """
        tool_name = tool_func.__name__
        if tool_name in self._tools:
            raise ValueError(f"Tool '{tool_name}' is already registered.")
        logger.info(f"Registering tool: {tool_name}")
        self._tools[tool_name] = tool_func

    async def execute(self, tool_name: str, **kwargs: Any) -> Any:
        """
        Executes a tool with the given arguments.
        """
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' not found.")
        tool_func = self._tools[tool_name]
        logger.info(f"Executing tool: {tool_name} with args: {kwargs}")
        return await tool_func(**kwargs)

    def get_schemas(self) -> List[Dict[str, Any]]:
        """
        Returns a list of Pydantic schemas for all registered tools.
        """
        schemas = []
        for name, func in self._tools.items():
            schema = getattr(func, "_pydantic_schema", None)
            if schema:
                schemas.append(
                    {
                        "name": name,
                        "description": func.__doc__ or "",
                        "parameters": schema.model_json_schema(),
                    }
                )
        return schemas
