
import asyncio
import json
import logging
from typing import List

import streamlit as st
from pydantic import BaseModel

from janus.agent import StandardPlannerAgent
from janus.memory import InMemoryWorkingMemory
from janus.models import ChatMessage, Role
from janus.orchestrator import AsyncLocalOrchestrator
from janus.tool import ToolRegistry, agent_tool

# Configure logging
class ListLogHandler(logging.Handler):
    def __init__(self, log_list):
        super().__init__()
        self.log_list = log_list

    def emit(self, record):
        self.log_list.append(self.format(record))

log_messages: List[str] = []
list_handler = ListLogHandler(log_messages)
app_logger = logging.getLogger("janus")
app_logger.addHandler(list_handler)
app_logger.setLevel(logging.INFO)

# Define a demo tool
class GetWeatherArgs(BaseModel):
    location: str

@agent_tool(args_schema=GetWeatherArgs)
async def get_weather(location: str) -> str:
    """Gets the weather for a given location."""
    # In a real scenario, this would call a weather API
    return f"The weather in {location} is sunny."

# Streamlit UI
st.set_page_config(layout="wide")
st.title("Janus Agent Framework")

# Sidebar for configuration and observability
st.sidebar.title("Configuration & Observability")
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
log_expander = st.sidebar.expander("Live Event Log (Pillar 5)")
log_container = log_expander.empty()

# Main page with two columns
col1, col2 = st.columns(2)

with col1:
    st.header("Agent Interaction")
    user_prompt = st.text_input("Enter your message to the agent:")
    if st.button("Run Agent"):
        if not api_key:
            st.warning("Please enter your OpenAI API Key in the sidebar.")
        else:
            log_messages.clear()
            with st.spinner("Agent is thinking..."):
                # Initialize memory
                if "memory" not in st.session_state:
                    st.session_state.memory = InMemoryWorkingMemory()
                memory = st.session_state.memory

                # Initialize tools
                tool_registry = ToolRegistry()
                tool_registry.register(get_weather)

                # Initialize agent and orchestrator
                agent = StandardPlannerAgent(api_key=api_key)
                orchestrator = AsyncLocalOrchestrator(
                    agent=agent,
                    tools=tool_registry,
                    memory=memory,
                )

                # Add user message to memory
                async def run_orchestrator():
                    await memory.add(
                        ChatMessage(role=Role.USER, content=user_prompt)
                    )

                    # Run the orchestrator
                    initial_state = {"messages": [msg.model_dump() for msg in memory.entries]}
                    async for _ in orchestrator.run(initial_state):
                        pass

                asyncio.run(run_orchestrator())

                # Update UI
                log_container.text("\n".join(log_messages))

with col2:
    st.header("Agent Memory (Pillar 4)")
    if "memory" in st.session_state:
        for msg in st.session_state.memory.entries:
            with st.chat_message(msg.role.value):
                if msg.content:
                    st.write(msg.content)
                if msg.tool_calls:
                    st.code(
                        json.dumps(
                            [tool_call.dict() for tool_call in msg.tool_calls],
                            indent=2,
                        ),
                        language="json",
                    )
