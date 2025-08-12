import os
import json
import logging
from openai import OpenAI, AsyncOpenAI

# Initialize the client as None
client: AsyncOpenAI | None = None
MODEL = "gpt-4-turbo-preview"

def get_client() -> AsyncOpenAI:
    """Initializes and returns the OpenAI client, creating it if it doesn't exist."""
    global client
    if client is None:
        # This will raise an error if the API key is not set, but only when it's first used.
        client = AsyncOpenAI()
    return client

async def invoke_agent_llm(prompt: str, agent_id: str, state: dict, message: dict) -> str:
    """
    Invokes an OpenAI model to get the next action plan for an agent.
    """
    logging.info(f"[OpenAI] Invoking agent '{agent_id}'")
    system_prompt = f"""
You are the AI model for an agent in a Self-Improving Agentic Mesh.
Your Agent ID is: {agent_id}
Your instructions are:
{prompt}

You must respond with a JSON object containing a list of actions.
The available action types are: "SEND_MESSAGE", "LOG_METRIC", "UPDATE_STATE", "DO_NOTHING".

Example response:
{json.dumps({"actions": [{"type": "SEND_MESSAGE", "target": "some_agent", "content": {"key": "value"}}]}, indent=2)}
"""
    user_prompt = f"""
Current state:
{json.dumps(state, indent=2)}

Incoming message:
{json.dumps(message, indent=2)}

Based on your instructions, the current state, and the incoming message, what is your action plan?
Return only the JSON object.
"""
    try:
        openai_client = get_client()
        response = await openai_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        action_plan_str = response.choices[0].message.content
        logging.info(f"[OpenAI] Raw response for '{agent_id}': {action_plan_str}")
        # Validate that the response is valid JSON
        json.loads(action_plan_str)
        return action_plan_str
    except Exception as e:
        logging.error(f"[OpenAI] Error invoking agent '{agent_id}': {e}", exc_info=True)
        return json.dumps({"actions": [{"type": "DO_NOTHING", "reason": str(e)}]})


async def invoke_architect_for_improvement(goal: str, fitness_score: float, agent_context: dict) -> dict:
    """
    Invokes the architect model to get a plan for improving the mesh.
    """
    logging.info("[OpenAI] Invoking Architect for improvement.")
    system_prompt = """
You are the Architect agent in a Self-Improving Agentic Mesh.
Your role is to analyze the system's performance and propose changes to improve it.
The system's current goal is: "{goal}"
The current fitness score is: {fitness_score} (where 1.0 is a perfect score).

You can propose one of the following actions:
- "CREATE_AGENT": Create a new agent to add capabilities.
- "DESTROY_AGENT": Remove a redundant or underperforming agent.
- "MODIFY_PROMPT": Change the instructions for an existing LLMAgent.
- "DO_NOTHING": If the system is performing well or no clear improvement is obvious.

You must return a JSON object describing the action and the specification for it.
"""
    user_prompt = f"""
The current high-level goal is: "{goal}"
The system's fitness score is {fitness_score:.4f}. This is below the target of 1.0.

Here is the context of the current operational agents:
{json.dumps(agent_context, indent=2)}

Based on this, what single improvement action do you propose?
Return only the JSON object.

Example for CREATE_AGENT:
{json.dumps({
    "action": "CREATE_AGENT",
    "reason": "The 'manager' agent needs a 'worker' to delegate the task to.",
    "spec": {
        "agent_id": "worker-01",
        "type": "LLMAgent",
        "config": {
            "prompt": "You are a worker. Your only job is to do tasks given to you."
        }
    }
}, indent=2)}

Example for MODIFY_PROMPT:
{json.dumps({
    "action": "MODIFY_PROMPT",
    "reason": "The 'manager' prompt is not explicit enough about waiting for a worker.",
    "spec": {
        "target_agent_id": "manager",
        "new_prompt": "You are a manager. Look for a worker. If you find one, delegate the task. If not, wait."
    }
}, indent=2)}
"""
    try:
        openai_client = get_client()
        response = await openai_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt.format(goal=goal, fitness_score=fitness_score)},
                {"role": "user", "content": user_prompt.format(goal=goal, fitness_score=fitness_score, agent_context=agent_context)},
            ],
            response_format={"type": "json_object"},
        )
        plan = json.loads(response.choices[0].message.content)
        logging.info(f"[OpenAI] Architect improvement plan: {plan}")
        return plan
    except Exception as e:
        logging.error(f"[OpenAI] Error invoking architect for improvement: {e}", exc_info=True)
        return {"action": "DO_NOTHING", "reason": f"LLM call failed: {e}"}


async def invoke_architect_for_goal_change(new_goal: str) -> dict:
    """
    Invokes the architect model to get a plan for adapting to a new goal.
    """
    logging.info(f"[OpenAI] Invoking Architect to adapt to new goal: {new_goal}")
    system_prompt = """
You are the Architect agent in a Self-Improving Agentic Mesh.
Your role is to devise a plan to adapt the system to a new high-level goal provided by a human.
A key way to adapt is to change the `fitness_function` in the task file, which is used to score the system's performance.

You must return a JSON object with the action "MODIFY_CODE".
The `spec` should contain the target file, the function name to be replaced, and the full, complete new Python code for that function.
The new code must be a single, valid Python function definition.
"""
    user_prompt = f"""
The system has been given a new high-level goal:
"{new_goal}"

To align with this goal, we need to modify the `fitness_function` in the file `tasks/dynamic_delegation_task.py`.
Please generate the Python code for a new `fitness_function(metrics_log: list) -> float` that correctly measures success for this new goal.

Return a JSON object with the action "MODIFY_CODE" and the new function's source code.
The `metrics_log` is a list of dictionaries, where each dict has an 'agent_id' and 'metric' key.
Example metric: `{{"agent_id": "worker-01", "metric": {{"name": "task_result", "value": "job_done"}}}}`

Return only the JSON object.
"""
    try:
        openai_client = get_client()
        response = await openai_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.format(new_goal=new_goal)},
            ],
            response_format={"type": "json_object"},
        )
        plan = json.loads(response.choices[0].message.content)
        logging.info(f"[OpenAI] Architect goal change plan: {plan}")
        return plan
    except Exception as e:
        logging.error(f"[OpenAI] Error invoking architect for goal change: {e}", exc_info=True)
        return {"action": "DO_NOTHING", "reason": f"LLM call failed: {e}"}
