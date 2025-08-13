from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

from minds.core.models import Message, AgentState

if TYPE_CHECKING:
    from minds.kernel.kernel import Kernel

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the mesh.
    """
    def __init__(self, kernel: 'Kernel'):
        self.state = AgentState()
        self.kernel = kernel

    @property
    def agent_id(self) -> str:
        """Convenience property to access the agent's ID."""
        return self.state.agent_id

    @abstractmethod
    def handle_message(self, message: Message) -> Optional[List[Message]]:
        """
        Process an incoming message.
        This method must be implemented by all concrete agent classes.
        It can optionally return a list of new messages to be dispatched.
        """
        pass
