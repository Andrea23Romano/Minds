import logging
from collections import deque
from typing import Dict, TYPE_CHECKING

from minds.core.models import Message

if TYPE_CHECKING:
    from minds.agents.base import BaseAgent


class Kernel:
    """
    The Kernel is the core of the Minds framework.
    It manages the message queue and dispatches messages to the appropriate agents.
    """
    def __init__(self):
        self.message_queue: deque[Message] = deque()
        self.agents: Dict[str, 'BaseAgent'] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def register_agent(self, agent: 'BaseAgent'):
        """
        Registers a new agent with the kernel.
        """
        if agent.agent_id in self.agents:
            raise ValueError(f"Agent with ID {agent.agent_id} is already registered.")
        self.agents[agent.agent_id] = agent
        self.logger.info(f"Agent {agent.agent_id} ({agent.__class__.__name__}) registered.")

    def dispatch_message(self, message: Message):
        """
        Adds a message to the central message queue.
        """
        self.logger.info(f"Dispatching message {message.id} from {message.source_agent_id} to {message.target_agent_id}.")
        self.message_queue.append(message)

    def _process_next_message(self):
        """
        Processes the next message in the queue.
        """
        if not self.message_queue:
            return

        message = self.message_queue.popleft()
        self.logger.info(f"Processing message {message.id} for agent {message.target_agent_id}.")

        target_agent = self.agents.get(message.target_agent_id)
        if target_agent:
            try:
                new_messages = target_agent.handle_message(message)
                if new_messages:
                    for msg in new_messages:
                        self.dispatch_message(msg)
            except Exception as e:
                self.logger.error(f"Error processing message {message.id} for agent {message.target_agent_id}: {e}", exc_info=True)
        else:
            self.logger.warning(f"No agent found with ID {message.target_agent_id} for message {message.id}. Message dropped.")

    def run(self):
        """
        Starts the main loop of the kernel.
        It continuously processes messages from the queue.
        """
        self.logger.info("Kernel starting main loop.")
        while True:
            self._process_next_message()
            # In a real scenario, we'd have a condition to break the loop
            # and a sleep mechanism to prevent busy-waiting if the queue is empty.
            # For this initial version, this is sufficient.
            if not self.message_queue:
                break # For now, exit when queue is empty to prevent infinite loop in simple run
        self.logger.info("Kernel main loop finished.")
