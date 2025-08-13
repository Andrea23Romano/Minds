import logging
from typing import List, Optional

from minds.agents.base import BaseAgent
from minds.core.models import Message

# A well-known ID for the interface agent to be easily reachable.
INTERFACE_AGENT_ID = "interface_agent"

class InterfaceAgent(BaseAgent):
    """
    An agent that provides an interface to the user.
    For now, it prints final results to the console.
    """
    def __init__(self, kernel: 'Kernel'):
        super().__init__(kernel)
        # Override the random ID with a fixed, well-known ID.
        self.state.agent_id = INTERFACE_AGENT_ID
        self.logger = logging.getLogger(self.__class__.__name__)

    def handle_message(self, message: Message) -> Optional[List[Message]]:
        """
        Handles a message by printing its content to the console.
        This simulates showing a final result to the user.
        """
        self.logger.info(f"InterfaceAgent received message from {message.source_agent_id}.")
        print("--- Response from Agent Mesh ---")
        print(message.content)
        print("--------------------------------")
        # This agent does not generate new messages in this simple scenario.
        return None
