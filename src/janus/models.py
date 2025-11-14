# src/janus/models.py
from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class WeatherArgs(BaseModel):
    location: str
