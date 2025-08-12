import json
from typing import Dict
from ..core.base_agent import BaseAgent

class DeveloperAgent(BaseAgent):
    async def think(self, message: Dict):
        content = message.get('content', {})
        if content.get('type') == 'EXECUTE_PLAN':
            plan = content['plan']
            self.logger.info(f"Received execution plan from architect: {json.dumps(plan)}")
            self.logger.info("Forwarding to deployment agent.")
            # The action is now more descriptive for deployment
            action_type = f"DEPLOY_{plan['action']}"
            await self.send_message("deployment", {"type": action_type, "plan": plan})
