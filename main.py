import os
import logging

from minds.kernel.kernel import Kernel
from minds.agents.system import InterfaceAgent, INTERFACE_AGENT_ID
from minds.agents.worker import WorkerAgent
from minds.core.models import Message

# Configure logging at the application entrypoint
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main entrypoint for the Minds application.
    """
    logger.info("Minds - Agentic Mesh Framework")
    logger.info("--------------------------------")

    # Before we start, let's check for the API key.
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("The OPENAI_API_KEY environment variable is not set.")
        logger.error("Please set it before running the application.")
        logger.error("export OPENAI_API_KEY='your_key_here'")
        return

    # 1. Initialize the Kernel
    kernel = Kernel()

    # 2. Instantiate agents
    interface_agent = InterfaceAgent(kernel)
    worker_agent = WorkerAgent(kernel) # It will read the API key from the environment

    # 3. Register agents with the kernel
    kernel.register_agent(interface_agent)
    kernel.register_agent(worker_agent)

    # 4. Create an initial user request message
    # This message simulates a user asking a question.
    # It originates from the interface agent and is directed to the worker.
    user_prompt = "Explain the theory of relativity in simple terms."
    logger.info(f"User Prompt: {user_prompt}")

    initial_message = Message(
        source_agent_id=INTERFACE_AGENT_ID,
        target_agent_id=worker_agent.agent_id,
        content=user_prompt
    )

    # 5. Dispatch the message
    kernel.dispatch_message(initial_message)

    # 6. Run the kernel's main loop
    # The kernel will process the message, the worker will generate a response,
    # send it to the interface agent, which will then print it.
    kernel.run()

    logger.info("Application finished.")


if __name__ == "__main__":
    main()
