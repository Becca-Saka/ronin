"""
LiteLLM wrapper client for communicating with LLMs (e.g. Gemini via OpenRouter).
Supports standard messaging and structured output via Pydantic schemas.
"""

import litellm
from pydantic import BaseModel

from .messages import Message


class LLMClient:
    """Client for making calls to the OpenRouter/LiteLLM completion API."""

    def __init__(self, model: str):
        """Initializes the LLM client.

        Args:
            model: The API model identifier to use for requests.
        """
        self._model = model

    def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
    ) -> str | BaseModel:
        """Sends a message sequence to the LLM and returns the completion.

        Args:
            messages: A list of Message objects (text, image, multipart) to send.
            response_model: Optional Pydantic BaseModel class for structured output.

        Returns:
            The raw text response from the LLM, or a validated instance of response_model
            if specified.
        """
        kwargs: dict = {
            "model": self._model,
            "messages": [m.to_dict() for m in messages],
            "extra_body": {"plugins": [{"id": "response-healing"}]},
        }
        if response_model:
            kwargs["response_format"] = response_model

        response = litellm.completion(**kwargs)
        content = response.choices[0].message.content

        if response_model:
            return response_model.model_validate_json(content)
        return content

