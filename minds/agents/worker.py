import logging
import os
from typing import List, Optional

import openai  # This will require 'pip install openai'

from minds.agents.base import BaseAgent
from minds.agents.system import INTERFACE_AGENT_ID
from minds.core.models import Message


class WorkerAgent(BaseAgent):
    """
    A worker agent that uses an LLM (OpenAI) to respond to prompts.
    """
    def __init__(self, kernel: 'Kernel'):
        super().__init__(kernel)
        self.logger = logging.getLogger(self.__class__.__name__)
        # It's crucial to have the API key.
        # The user should set the OPENAI_API_KEY environment variable.
        if not os.getenv("OPENAI_API_KEY"):
            self.logger.warning("OPENAI_API_KEY environment variable not set. The agent will not work.")
            # In a real app, you might want to raise an exception.
            # For now, a warning is fine.
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def handle_message(self, message: Message) -> Optional[List[Message]]:
        """
        Handles a prompt by calling the OpenAI API and sending the response
        to the InterfaceAgent.
        """
        self.logger.info(f"WorkerAgent received message from {message.source_agent_id}.")
        prompt = message.content

        if not isinstance(prompt, str):
            self.logger.warning(f"Received non-string content: {type(prompt)}. Ignoring.")
            return None

        try:
            self.logger.info("Calling OpenAI API...")
            # This is a deprecated API call style for openai<1.0
            # We will assume an older version for now for simplicity,
            # but this is a candidate for a future self-improvement task for the mesh.
            response = openai.Completion.create(
                model="text-davinci-003", # A standard, powerful model
                prompt=prompt,
                max_tokens=150
            )
            llm_response = response.choices[0].text.strip()
            self.logger.info("Received response from OpenAI API.")

            # Create a new message with the response for the interface agent
            response_message = Message(
                source_agent_id=self.agent_id,
                target_agent_id=INTERFACE_AGENT_ID,
                content=llm_response
            )
            return [response_message]

        except Exception as e:
            self.logger.error(f"An error occurred while calling OpenAI API: {e}", exc_info=True)
            # Optionally, inform the user about the error.
            error_response = Message(
                source_agent_id=self.agent_id,
                target_agent_id=INTERFACE_AGENT_ID,
                content=f"Sorry, I encountered an error: {e}"
            )
            return [error_response]
