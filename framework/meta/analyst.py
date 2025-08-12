import asyncio
import logging
from typing import Callable, Dict
from ..core.base_agent import BaseAgent

class AnalystAgent(BaseAgent):
    def __init__(self, kernel: 'Kernel', agent_id: str, fitness_function: Callable, **kwargs):
        super().__init__(kernel, agent_id)
        self.fitness_function = fitness_function
        self.metrics_log = []
    async def think(self, message: Dict): pass
    async def run(self):
        self.logger.info("Starting to monitor metrics bus.")
        while True:
            try:
                metric_data = await self.kernel.metrics_bus.get()
                self.metrics_log.append(metric_data)
                fitness_score = self.fitness_function(self.metrics_log)
                self.logger.info(f"New fitness score: {fitness_score:.4f}")
                agent_context = {}
                for aid, agent in self.kernel.agents.items():
                    if aid in ["analyst", "architect", "developer", "deployment", "human_interface"]: continue
                    context = {"type": agent.__class__.__name__}
                    if hasattr(agent, 'prompt'): context['prompt'] = agent.prompt
                    agent_context[aid] = context
                await self.send_message("architect", {"type": "FITNESS_REPORT", "score": fitness_score, "agent_context": agent_context})
                self.kernel.metrics_bus.task_done()
            except asyncio.CancelledError: break
