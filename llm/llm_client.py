import os

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel

from .messages import Message

load_dotenv()

_MODEL = "openrouter/google/gemini-2.5-flash"


class LLMClient:
    def __init__(self, model: str = _MODEL):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENROUTER_API_KEY is not set")
        os.environ["OPENROUTER_API_KEY"] = api_key
        self._model = model

    def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
    ) -> str | BaseModel:
        kwargs: dict = {
            "model": self._model,
            "messages": [message.to_dict() for message in messages],
            "extra_body": {"plugins": [{"id": "response-healing"}]},
        }

        if response_model:
            kwargs["response_format"] = response_model

        response = litellm.completion(**kwargs)
        content = response.choices[0].message.content

        if response_model:
            return response_model.model_validate_json(content)
        return content
