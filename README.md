# Self-Improving Agentic Mesh

This project is a Python framework for creating a "Self-Improving Agentic Mesh," a system of interconnected, autonomous agents that can dynamically adapt their structure and behavior to achieve high-level goals.

## Core Principles

The framework is built on the following principles:

*   **Heterogeneous Composition**: The mesh is a container for diverse computational units, including deterministic Python functions (`FunctionAgent`) and generative, language-driven agents (`LLMAgent`).
*   **Dynamic Topology**: A privileged "meta-plane" of agents has the authority to create and destroy operational agents at runtime, altering the mesh's structure to better achieve its goals.
*   **Goal-Driven Adaptation**: The system's behavior is guided by a high-level goal and a quantifiable `fitness_function`. The meta-plane's primary directive is to generate and execute plans that optimize this fitness score.
*   **Human-in-the-Loop Feedback**: A dedicated `HumanInterfaceAgent` allows a human operator to provide new goals in real-time, causing the mesh to adapt its objectives and success criteria dynamically.
*   **Prompt & Code Co-evolution**: The system can improve itself by modifying the natural language prompts of `LLMAgent`s or by rewriting the underlying Python source code of `FunctionAgent`s or even the task's `fitness_function` itself.

## Project Structure

The project is organized into a modular structure that separates the reusable framework from user-defined tasks.

```
/
|-- framework/
|   |-- core/             # Core components: BaseAgent, Kernel, Agent implementations
|   |-- meta/             # Self-improvement agents: Analyst, Architect, etc.
|   |-- integrations/     # Connectors to external services (e.g., LLMs)
|
|-- tasks/                # Definitions of specific problems for the mesh to solve
|-- tests/                # Unit tests for the framework
|
|-- app.py                # Main Streamlit user interface
|-- requirements.txt      # Project dependencies
|-- README.md             # This file
```

## Getting Started

### 1. Prerequisites

*   Python 3.8+
*   An OpenAI API key

### 2. Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd self-improving-mesh
    ```

2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

### 3. Configuration

This framework uses the OpenAI API to power its language-based agents. You must configure your API key as an environment variable.

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## How to Use

### Running the Application

The main user interface is a Streamlit application. To run it, execute the following command in your terminal:

```bash
streamlit run app.py
```

This will open a new tab in your browser with the "Human-Mesh Interface," where you can start the simulation, observe the agent mesh, and provide new goals.

### Running the Tests

The project includes a suite of unit tests for the core framework. To run them, use the following command:

```bash
python -m unittest discover tests
```

This will discover and run all tests located in the `tests/` directory.
