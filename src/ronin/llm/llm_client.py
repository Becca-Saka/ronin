"""
LiteLLM wrapper client for communicating with LLMs (e.g. Gemini via OpenRouter).
Supports standard messaging and structured output via Pydantic schemas.
"""

import json
import logging
from typing import TypeVar, overload

import litellm
from pydantic import BaseModel, ValidationError

from . import usage
from .messages import Message, TextMessage

logger = logging.getLogger(__name__)

STRUCTURED_ATTEMPTS = 2

Answer = TypeVar("Answer", bound=BaseModel)


class LLMClient:
    """Client for making calls to the OpenRouter/LiteLLM completion API."""

    def __init__(self, model: str, usage_callback=None):
        """Initializes the LLM client.

        Args:
            model: The API model identifier to use for requests.
            usage_callback: Optional callable receiving a usage.CallRecord per call.
        """
        self._model = model
        self.usage_callback = usage_callback

    @overload
    def generate_response(self, messages: list[Message]) -> str: ...

    @overload
    def generate_response(
        self, messages: list[Message], response_model: None
    ) -> str: ...

    @overload
    def generate_response(
        self, messages: list[Message], response_model: type[Answer]
    ) -> Answer: ...

    def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
    ) -> str | BaseModel:
        """Sends a message sequence to the LLM and returns the completion.

        Overloaded so a caller asking for a model is given that model, rather
        than a union it has to assert its way out of.

        Args:
            messages: A list of Message objects (text, image, multipart) to send.
            response_model: Optional Pydantic BaseModel class for structured output.

        Returns:
            The raw text response from the LLM, or a validated instance of response_model
            if specified.
        """
        attempts: list[Message] = list(messages)
        last_error: ValidationError | None = None

        # Schema enforcement is advisory on most providers, so a model may
        # answer in a shape the schema forbids. One correction is far cheaper
        # than losing the scenario the call belongs to.
        for attempt in range(STRUCTURED_ATTEMPTS if response_model else 1):
            try:
                return self._complete(attempts, response_model, attempt + 1)
            except ValidationError as exc:
                last_error = exc
                logger.warning(
                    "Model response did not match %s (attempt %d/%d; %d validation errors)",
                    response_model.__name__ if response_model else "schema",
                    attempt + 1,
                    STRUCTURED_ATTEMPTS,
                    len(exc.errors()),
                )
                safe_errors = exc.errors(include_url=False, include_input=False)
                attempts = list(messages) + [
                    TextMessage(
                        content=(
                            "Your previous reply did not match the required JSON schema:\n"
                            f"{json.dumps(safe_errors)}\n"
                            "Answer again as JSON matching the schema exactly."
                        )
                    )
                ]

        raise last_error

    def _complete(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        attempt: int = 1,
    ) -> str | BaseModel:
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
            self.usage_callback(usage.record(response, self._model, attempt))

        content = response.choices[0].message.content

        if not response_model:
            return content
        return self._parse(content, response_model)

    def _parse(self, content: str, response_model: type[BaseModel]) -> BaseModel:
        stripped = (content or "").strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            stripped = "\n".join(
                lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            )
        return response_model.model_validate_json(stripped)
