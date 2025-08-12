"""
The main Streamlit UI for the Self-Improving Agentic Mesh, with a
live graph visualization and interactive mission control panel.
"""
import asyncio
import logging
import time
from typing import Dict, Callable

import streamlit as st

# --- Framework Imports ---
from framework.core.kernel import MeshKernel
from framework.core.agents import LLMAgent, FunctionAgent, HumanInterfaceAgent
from framework.meta.analyst import AnalystAgent
from framework.meta.architect import ArchitectAgent
from framework.meta.developer import DeveloperAgent
from framework.meta.deployment import DeploymentAgent

# --- Task Import ---
from tasks.dynamic_delegation_task import TASK_SPEC

# --- UI Configuration ---
st.set_page_config(page_title="Agentic Mesh Interface", layout="wide")
st.title("🤖 Human-Mesh Interface")
st.caption("Observe and interact with a live, self-improving agentic mesh.")

# --- UI Components ---
class StreamlitLogHandler(logging.Handler):
    """Custom logging handler to display logs in a Streamlit placeholder."""
    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder
        if "logs" not in st.session_state:
            st.session_state.logs = []

    def emit(self, record):
        st.session_state.logs.append(self.format(record))
        log_str = "\n".join(st.session_state.logs)
        self.placeholder.code(log_str, language="log")

def generate_graphviz(kernel: MeshKernel) -> str:
    """Generates a Graphviz DOT language string to visualize the mesh."""
    dot_lines = [
        'digraph Mesh {',
        '  rankdir=LR;',
        '  node [shape=box, style="rounded,filled", fontname="sans-serif"];',
        '  graph [bgcolor="#F0F2F6", fontname="sans-serif"];'
    ]
    styles = {
        "HumanInterfaceAgent": 'fillcolor="#cccccc"', # Grey
        "LLMAgent": 'fillcolor="#a9d1f7"',      # Light Blue
        "FunctionAgent": 'fillcolor="#a9f7c8"', # Light Green
        "AnalystAgent": 'fillcolor="#f7d1a9"',  # Orange
        "ArchitectAgent": 'fillcolor="#f7a9a9"',# Red
        "DeveloperAgent": 'fillcolor="#f7f1a9"',# Yellow
        "DeploymentAgent": 'fillcolor="#d1a9f7"'# Purple
    }
    for agent_id, agent in kernel.agents.items():
        agent_type = agent.__class__.__name__
        style = styles.get(agent_type, 'fillcolor="white"')
        dot_lines.append(f'  "{agent_id}" [label="{agent_id}\\n({agent_type})", {style}];')
    dot_lines.append('}')
    return "\n".join(dot_lines)

# --- Agent Factory ---
AGENT_CLASS_MAP = {
    "LLMAgent": LLMAgent,
    "FunctionAgent": FunctionAgent,
    "HumanInterfaceAgent": HumanInterfaceAgent
}
def create_agent_from_config(kernel: MeshKernel, agent_config: Dict) -> 'BaseAgent':
    """Factory function to instantiate the correct agent class from a config dict."""
    agent_type = agent_config.get("type")
    agent_class = AGENT_CLASS_MAP.get(agent_type)
    if not agent_class:
        raise ValueError(f"Unknown agent type: {agent_type}")
    config = agent_config.get("config", {})
    return agent_class(kernel=kernel, agent_id=agent_config["agent_id"], **config)

# --- Simulation Management ---
def initialize_simulation(task_spec: Dict):
    """Sets up the kernel and agents, storing them in the session state."""
    st.session_state.logs = []
    kernel = MeshKernel()

    # Add the human interface agent first
    kernel.register_agent(HumanInterfaceAgent(kernel, "human_interface"))

    # Instantiate operational agents
    for agent_config in task_spec["initial_agents"]:
        agent = create_agent_from_config(kernel, agent_config)
        kernel.register_agent(agent)

    # Instantiate the meta-plane
    kernel.register_agent(AnalystAgent(kernel, "analyst", fitness_function=task_spec["fitness_function"]))
    kernel.register_agent(ArchitectAgent(kernel, "architect", goal=task_spec["goal"]))
    kernel.register_agent(DeveloperAgent(kernel, "developer"))
    kernel.register_agent(DeploymentAgent(kernel, "deployment", agent_factory=create_agent_from_config))

    # Start all agent tasks
    asyncio.run(kernel.start(task_spec.get("kickstart_message")))

    st.session_state.kernel = kernel
    st.session_state.simulation_running = True
    logging.info(f"--- Task Started: {task_spec['goal']} ---")

def stop_simulation():
    """Stops all agent tasks and cleans up the session state."""
    if "kernel" in st.session_state and st.session_state.kernel:
        st.session_state.kernel.stop()
    st.session_state.simulation_running = False
    st.session_state.kernel = None
    logging.info("--- Simulation Stopped by User ---")

# --- Main UI Rendering ---
if "simulation_running" not in st.session_state:
    st.session_state.simulation_running = False
    st.session_state.kernel = None
    st.session_state.logs = []

# Top control bar
if not st.session_state.simulation_running:
    if st.button("🚀 Start Simulation", use_container_width=True):
        initialize_simulation(TASK_SPEC)
        st.rerun()
else:
    if st.button("🛑 Stop Simulation", use_container_width=True):
        stop_simulation()
        st.rerun()

# Main dashboard layout
if st.session_state.simulation_running:
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("Live Mesh Visualization")
        graph_placeholder = st.empty()
        st.subheader("Agent Inspector")
        agent_ids = list(st.session_state.kernel.agents.keys())
        selected_agent_id = st.selectbox("Select an agent to inspect:", agent_ids)
        if selected_agent_id:
            agent = st.session_state.kernel.agents[selected_agent_id]
            st.text(f"Agent ID: {agent.agent_id}")
            st.text(f"Agent Type: {agent.__class__.__name__}")
            if hasattr(agent, 'prompt'):
                st.text_area("Live Prompt:", value=agent.prompt, height=200, disabled=True)

    with col2:
        st.subheader("Mission Control")
        new_goal = st.text_input("Enter a new high-level goal for the mesh:", key="goal_input")
        if st.button("Submit New Goal"):
            if new_goal and "kernel" in st.session_state:
                kernel = st.session_state.kernel
                interface_agent = kernel.agents.get("human_interface")
                if interface_agent:
                    message = {"content": {"new_goal_description": new_goal}}
                    asyncio.run(interface_agent.inbox.put(message))
                    st.toast("New goal submitted to the mesh!")
            else:
                st.warning("Please enter a goal.")

        st.subheader("Real-time Logs")
        log_placeholder = st.empty()
        # Set up logging to go to both the Streamlit UI and the console
        logging.basicConfig(
            level=logging.INFO,
            handlers=[
                StreamlitLogHandler(log_placeholder),
                logging.StreamHandler() # This logs to the console
            ],
            force=True,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    # The main loop for live updates
    while st.session_state.simulation_running:
        dot_string = generate_graphviz(st.session_state.kernel)
        graph_placeholder.graphviz_chart(dot_string)
        time.sleep(1.5)
        st.rerun()
else:
    st.info("Simulation is stopped.")
