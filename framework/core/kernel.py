import asyncio
import logging
import json
from typing import Dict, Any

class MeshKernel:
    def __init__(self):
        self.agents: Dict[str, 'BaseAgent'] = {}
        self.agent_tasks: Dict[str, asyncio.Task] = {}
        self.logger = logging.getLogger("MeshKernel")
        self.metrics_bus = asyncio.Queue()
    def register_agent(self, agent: 'BaseAgent'):
        if agent.agent_id in self.agents: raise ValueError(f"Agent ID {agent.agent_id} exists.")
        self.agents[agent.agent_id] = agent
        self.logger.info(f"Registered agent '{agent.agent_id}'")
    async def start_single_agent(self, agent_id: str):
        if agent_id in self.agent_tasks: return
        agent = self.agents.get(agent_id)
        if agent:
            task = asyncio.create_task(agent.run())
            self.agent_tasks[agent_id] = task
            self.logger.info(f"Dynamically started agent '{agent_id}'.")
    async def deregister_agent(self, agent_id: str):
        if agent_id not in self.agents: return
        if agent_id in self.agent_tasks:
            task = self.agent_tasks.pop(agent_id)
            task.cancel()
            await asyncio.sleep(0)
        if agent_id in self.agents: del self.agents[agent_id]
        self.logger.info(f"Deregistered agent '{agent_id}'.")
    def update_agent_prompt(self, agent_id: str, new_prompt: str):
        agent = self.agents.get(agent_id)
        if agent and hasattr(agent, 'prompt'):
            self.logger.info(f"Updating prompt for agent '{agent_id}'.")
            agent.prompt = new_prompt
    async def route_message(self, sender_id: str, target_id: str, content: Any):
        self.logger.info(
            f"Routing message from '{sender_id}' to '{target_id}'. Content: {json.dumps(content)}"
        )
        target_agent = self.agents.get(target_id)
        if target_agent:
            await target_agent.inbox.put({"sender_id": sender_id, "content": content})
        else:
            self.logger.warning(f"Could not route message: Target agent '{target_id}' not found.")
    async def log_metric(self, agent_id: str, metric: Dict[str, Any]):
        await self.metrics_bus.put({"agent_id": agent_id, "metric": metric})
    async def start(self, kickstart_message: Dict[str, Any] = None):
        self.logger.info("Starting all initial agent tasks...")
        for agent_id in self.agents.keys(): await self.start_single_agent(agent_id)
        if kickstart_message: await self.route_message("kernel", kickstart_message['target_id'], kickstart_message['content'])
    def stop(self):
        for task in self.agent_tasks.values(): task.cancel()
