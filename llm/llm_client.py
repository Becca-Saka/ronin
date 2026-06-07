"""
LiteLLM wrapper client for communicating with LLMs (e.g. Gemini via OpenRouter).
Supports standard messaging and structured output via Pydantic schemas.
"""

import litellm
from pydantic import BaseModel

from .messages import Message


class LLMClient:
    """Client for making calls to the OpenRouter/LiteLLM completion API."""

    def __init__(self, model: str, usage_callback=None):
        """Initializes the LLM client.

        Args:
            model: The API model identifier to use for requests.
            usage_callback: Optional callable receiving usage dictionary records.
        """
        self._model = model
        self.usage_callback = usage_callback

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
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": response_model.model_json_schema(),
                    "strict": False,
                },
            }

        response = litellm.completion(**kwargs)

        if self.usage_callback:
            usage = getattr(response, "usage", None)
            if usage:
                cost = 0.0
                try:
                    cost = litellm.completion_cost(completion_response=response) or 0.0
                except Exception:
                    pass
                self.usage_callback(self._model, cost, usage)

        content = response.choices[0].message.content

        if response_model:
            stripped = content.strip()
            if stripped.startswith("```"):
                lines = stripped.splitlines()
                stripped = "\n".join(
                    lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
                )
            return response_model.model_validate_json(stripped)
        return content
