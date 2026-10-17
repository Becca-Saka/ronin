"""Checking that a model's provider credentials are present.

LiteLLM knows what each provider needs: one key for most, several for Bedrock,
a base URL for Ollama. So every caller defers to it rather than mapping model
prefixes to environment variables. It reports nothing missing for models it
cannot route, which keeps custom endpoints working.

A decision model is the exception, and is named below. It is checked here so
that one call still answers for every model a run needs.
"""

import io
import os
from contextlib import redirect_stderr, redirect_stdout
from typing import Optional

from ronin.errors import UsageError

DECISION_PREFIX = "typesafe/"
"""How a decision model names itself. LiteLLM's own prefix for the provider."""

DECISION_KEY = "TYPESAFE_API_KEY"
"""The name the decision SDK reads, and the one LiteLLM's provider expects."""


def missing_credentials(model: Optional[str]) -> list[str]:
    """Returns the environment variables a model needs but does not have."""
    if not model:
        return []
    if model.startswith(DECISION_PREFIX):
        return [] if os.environ.get(DECISION_KEY) else [DECISION_KEY]
    import litellm

    # validate_environment prints a provider banner for models it cannot route.
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return litellm.validate_environment(model).get("missing_keys") or []


def require_credentials(*models: Optional[str]) -> None:
    """Refuses to start when a model's credentials are absent."""
    for model in models:
        missing = missing_credentials(model)
        if missing:
            raise UsageError(
                f"Missing credentials for '{model}': {', '.join(missing)}.\n"
                "Add them to a .env file in this directory or export them."
            )
