import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAgent(ABC):
    def __init__(self, kernel: 'MeshKernel', agent_id: str):
        self.agent_id = agent_id
        self.kernel = kernel
        self.inbox = asyncio.Queue()
        self.logger = logging.getLogger(f"Agent-{self.agent_id}")
        self.logger.info(f"Agent '{self.agent_id}' of type '{self.__class__.__name__}' initialized.")
    async def run(self):
        self.logger.info("Starting run loop.")
        while True:
            try:
                message = await self.inbox.get()
                sender_id = message.get("sender_id", "kernel")
                self.logger.info(f"Received message from '{sender_id}'. Processing...")
                await self.think(message)
                self.inbox.task_done()
            except asyncio.CancelledError:
                self.logger.info("Run loop cancelled.")
                break
    @abstractmethod
    async def think(self, message: Dict[str, Any]): pass
    async def send_message(self, target_agent_id: str, content: Any):
        await self.kernel.route_message(self.agent_id, target_agent_id, content)
