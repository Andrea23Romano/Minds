import json
from typing import Dict
from ..core.base_agent import BaseAgent
from ..integrations.llm import invoke_architect_for_goal_change, invoke_architect_for_improvement

class ArchitectAgent(BaseAgent):
    def __init__(self, kernel: 'Kernel', agent_id: str, goal: str, **kwargs):
        super().__init__(kernel, agent_id)
        self.goal = goal
    async def think(self, message: Dict):
        content = message.get('content', {})
        msg_type = content.get('type')
        if msg_type == 'SET_NEW_GOAL':
            self.goal = content.get('goal')
            plan = await invoke_architect_for_goal_change(self.goal)
            await self.send_message("developer", {"type": "EXECUTE_PLAN", "plan": plan})
        elif msg_type == 'FITNESS_REPORT' and content['score'] < 0.99:
            plan = await invoke_architect_for_improvement(self.goal, content['score'], content['agent_context'])
            await self.send_message("developer", {"type": "EXECUTE_PLAN", "plan": plan})
