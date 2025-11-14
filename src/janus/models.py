# src/janus/models.py
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ChatMessage(BaseModel):
    role: Role
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class WeatherArgs(BaseModel):
    location: str
