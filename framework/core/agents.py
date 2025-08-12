import json
from typing import Dict, Any, Callable
from .base_agent import BaseAgent
from ..integrations.llm import invoke_agent_llm

class HumanInterfaceAgent(BaseAgent):
    async def think(self, message: Dict[str, Any]):
        content = message.get("content", {})
        new_goal = content.get("new_goal_description")

        if new_goal:
            self.logger.info(f"Received new goal from human user: '{new_goal}'")
            await self.send_message("architect", {"type": "SET_NEW_GOAL", "goal": new_goal})
        else:
            self.logger.warning(
                f"Received message I don't know how to handle: {json.dumps(content)}"
            )
class LLMAgent(BaseAgent):
    def __init__(self, kernel: 'Kernel', agent_id: str, prompt: str):
        super().__init__(kernel, agent_id)
        self.prompt = prompt
        self.state = {}
    async def think(self, message: Dict[str, Any]):
        try:
            action_plan_str = await invoke_agent_llm(self.prompt, self.agent_id, self.state, message)
            action_plan = json.loads(action_plan_str)
            for action in action_plan.get("actions", []):
                action_type = action.get("type")
                self.logger.info(f"Executing LLM action: {action_type}")
                if action_type == "SEND_MESSAGE": await self.send_message(action["target"], action["content"])
                elif action_type == "LOG_METRIC": await self.kernel.log_metric(self.agent_id, action["metric"])
                elif action_type == "UPDATE_STATE": self.state.update(action["new_state"])
        except Exception as e: self.logger.error(f"LLMAgent think cycle failed: {e}", exc_info=True)
class FunctionAgent(BaseAgent):
    def __init__(self, kernel: 'Kernel', agent_id: str, function: Callable):
        super().__init__(kernel, agent_id)
        self.function = function
    async def think(self, message: Dict[str, Any]):
        try:
            result = self.function(message.get('content'))
            if result and 'target' in result and 'content' in result:
                await self.send_message(result['target'], result['content'])
        except Exception as e: self.logger.error(f"FunctionAgent execution failed: {e}", exc_info=True)
