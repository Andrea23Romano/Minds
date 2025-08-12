from ..core.base_agent import BaseAgent

class DeveloperAgent(BaseAgent):
    async def think(self, message: Dict):
        content = message.get('content', {})
        if content.get('type') == 'EXECUTE_PLAN':
            plan = content['plan']
            # The action is now more descriptive for deployment
            action_type = f"DEPLOY_{plan['action']}"
            await self.send_message("deployment", {"type": action_type, "plan": plan})
