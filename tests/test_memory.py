# tests/test_memory.py
import pytest
from pydantic import BaseModel

from janus.memory import InMemoryWorkingMemory


class DummyEntry(BaseModel):
    data: str


@pytest.fixture
def memory():
    return InMemoryWorkingMemory()


@pytest.mark.asyncio
async def test_add_entry(memory: InMemoryWorkingMemory):
    entry = DummyEntry(data="test")
    await memory.add(entry)
    assert len(memory.entries) == 1
    assert memory.entries[0] == entry


@pytest.mark.asyncio
async def test_get_context(memory: InMemoryWorkingMemory):
    entry1 = DummyEntry(data="test1")
    entry2 = DummyEntry(data="test2")
    await memory.add(entry1)
    await memory.add(entry2)
    context = await memory.get_context()
    assert context == {
        "messages": [
            {"data": "test1"},
            {"data": "test2"},
        ]
    }
