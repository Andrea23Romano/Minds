# demo.py
import asyncio
import json
import logging

from janus.agent import StandardPlannerAgent
from janus.memory import InMemoryWorkingMemory
from janus.models import WeatherArgs
from janus.orchestrator import AsyncLocalOrchestrator
from janus.tool import ToolRegistry, agent_tool

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@agent_tool(args_schema=WeatherArgs)
async def get_weather(location: str) -> str:
    """
    A mock tool to get the weather.
    """
    if location == "SF":
        return "Sunny"
    return "Cloudy"


async def run_demo():
    """
    Instantiates and runs a demo of the Janus MVP framework.
    """
    # 1. Instantiate components
    tool_registry = ToolRegistry()
    memory = InMemoryWorkingMemory()
    agent = StandardPlannerAgent()

    # 2. Register tools
    tool_registry.register(get_weather)

    # 3. Instantiate orchestrator
    orchestrator = AsyncLocalOrchestrator(
        agent=agent, tools=tool_registry, memory=memory
    )

    # 4. Define initial state
    initial_state = {
        "messages": [{"role": "user", "content": "What's the weather in SF?"}]
    }

    # 5. Run the orchestrator
    final_state = await orchestrator.run(initial_state)

    # 6. Print the final state
    print("--- Final State ---")
    print(json.dumps(final_state, indent=2))


if __name__ == "__main__":
    asyncio.run(run_demo())
