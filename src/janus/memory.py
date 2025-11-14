# src/janus/memory.py
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class BaseMemory(ABC):
    """
    Abstract base class for memory systems.
    """

    @abstractmethod
    async def add(self, entry: BaseModel):
        """
        Adds an entry to the memory.
        """
        pass

    @abstractmethod
    async def get_context(self) -> Dict[str, Any]:
        """
        Retrieves the current context from memory.
        """
        pass


class InMemoryWorkingMemory(BaseMemory):
    """
    An in-memory list-based working memory.
    """

    def __init__(self):
        self.entries: List[BaseModel] = []

    async def add(self, entry: BaseModel):
        """
        Adds an entry to the in-memory list.
        """
        logger.info(f"Adding to memory: {entry.model_dump()}")
        self.entries.append(entry)

    async def get_context(self) -> Dict[str, Any]:
        """
        Returns the list of entries.
        """
        return {"messages": [entry.model_dump() for entry in self.entries]}
