import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

@dataclass
class AgentState:
    """
    Represents the state of an agent in the mesh.
    For now, it's a simple container for the agent's ID, but it can be expanded.
    """
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class Message:
    """
    Represents a message passed between agents in the mesh.
    """
    source_agent_id: str
    target_agent_id: str
    content: Any
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        # Basic validation
        if not self.source_agent_id or not self.target_agent_id:
            raise ValueError("source_agent_id and target_agent_id cannot be empty.")
