import json
import logging

async def invoke_agent_llm(prompt: str, agent_id: str, state: dict, message: dict) -> str:
    # NOTE: This is a simulation. A real implementation would make an API call.
    logging.info(f"[LLM Sim] Invoking agent '{agent_id}'")
    if "manager" in agent_id:
        return json.dumps({"actions": [{"type": "SEND_MESSAGE", "target": "worker-01", "content": {"task": "job_done"}}]})
    if "worker" in agent_id:
        return json.dumps({"actions": [{"type": "LOG_METRIC", "metric": {"name": "task_result", "value": "job_done"}}, {"type": "SEND_MESSAGE", "target": "manager", "content": {"type": "complete"}}]})
    return json.dumps({"actions": [{"type": "DO_NOTHING"}]})

async def invoke_architect_for_improvement(goal: str, fitness_score: float, agent_context: dict) -> dict:
    logging.info("[LLM Sim] Invoking Architect for improvement.")
    return {
        "action": "CREATE_AGENT",
        "reason": "The 'manager' agent needs a 'worker' to delegate the task to.",
        "spec": {
            "agent_id": "worker-01",
            "type": "LLMAgent",
            "config": {
                "prompt": "You are a worker. When you receive a message with a 'task', log a metric 'task_result' with the value from the task, then send a 'complete' message back to the sender."
            }
        }
    }

async def invoke_architect_for_goal_change(new_goal: str) -> dict:
    logging.info(f"[LLM Sim] Invoking Architect to adapt to new goal: {new_goal}")
    new_function_code = """
def fitness_function(metrics_log: list) -> float:
    for record in reversed(metrics_log):
        metric = record.get("metric", {})
        if metric.get("name") == "manager_confirmed_completion" and metric.get("value") == True:
            return 1.0
    return 0.1
"""
    return {
        "action": "MODIFY_CODE",
        "reason": f"The system's goal has changed to '{new_goal}'. A new fitness function is required.",
        "spec": {
            "target_file": "tasks/dynamic_delegation_task.py",
            "target_function_name": "fitness_function",
            "new_function_code": new_function_code
        }
    }
