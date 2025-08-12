"""
Defines a task where the mesh must dynamically create a worker agent.
"""

def fitness_function(metrics_log: list) -> float:
    """Rewards the system if the worker completes the task."""
    for record in reversed(metrics_log):
        metric = record.get("metric", {})
        if metric.get("name") == "task_result" and metric.get("value") == "job_done":
            return 1.0 # Perfect score
    return 0.1 # Low score if the worker hasn't reported success

TASK_SPEC = {
    "goal": "The 'manager' agent must successfully delegate a task to a 'worker' agent and get a completion confirmation. If no worker exists, one must be created.",
    "initial_agents": [
        {
            "agent_id": "manager",
            "type": "LLMAgent",
            "config": {
                "prompt": "You are a manager. Your task is to get a job done. You cannot do it yourself. On startup, check for a 'worker-01' agent. If it exists, delegate a task to it by sending a message with content {'task': 'job_done'}. If it does not exist, do nothing and wait. When you receive a 'complete' message, log a metric 'manager_confirmed_completion' with value true."
            }
        }
    ],
    "fitness_function": fitness_function,
    "kickstart_message": {
        "target_id": "manager",
        "content": {"type": "START"}
    }
}
