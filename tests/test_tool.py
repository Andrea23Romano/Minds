# tests/test_tool.py
import pytest
from pydantic import BaseModel

from janus.tool import ToolRegistry, agent_tool


class DummySchema(BaseModel):
    arg1: str
    arg2: int


@agent_tool(args_schema=DummySchema)
async def dummy_tool(arg1: str, arg2: int):
    """A dummy tool for testing."""
    return f"{arg1}-{arg2}"


@pytest.fixture
def tool_registry():
    return ToolRegistry()


def test_register_tool(tool_registry: ToolRegistry):
    tool_registry.register(dummy_tool)
    assert "dummy_tool" in tool_registry._tools


def test_register_duplicate_tool_raises_error(tool_registry: ToolRegistry):
    tool_registry.register(dummy_tool)
    with pytest.raises(ValueError):
        tool_registry.register(dummy_tool)


@pytest.mark.asyncio
async def test_execute_tool(tool_registry: ToolRegistry):
    tool_registry.register(dummy_tool)
    result = await tool_registry.execute("dummy_tool", arg1="test", arg2=123)
    assert result == "test-123"


@pytest.mark.asyncio
async def test_execute_nonexistent_tool_raises_error(tool_registry: ToolRegistry):
    with pytest.raises(ValueError):
        await tool_registry.execute("nonexistent_tool")


def test_get_schemas(tool_registry: ToolRegistry):
    tool_registry.register(dummy_tool)
    schemas = tool_registry.get_schemas()
    assert len(schemas) == 1
    schema = schemas[0]
    assert schema["type"] == "function"
    assert "function" in schema
    function_spec = schema["function"]
    assert function_spec["name"] == "dummy_tool"
    assert function_spec["description"] == "A dummy tool for testing."
    assert "arg1" in function_spec["parameters"]["properties"]
