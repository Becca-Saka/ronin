"""Checking that a model's provider credentials are present.

LiteLLM knows what each provider needs — one key for most, several for
Bedrock, a base URL for Ollama — so every caller defers to it rather than
mapping model prefixes to environment variables. It reports nothing missing
for models it cannot route, which keeps custom endpoints working.
"""

import io
from contextlib import redirect_stderr, redirect_stdout
from typing import Optional

from ronin.errors import UsageError


def missing_credentials(model: Optional[str]) -> list[str]:
    """Returns the environment variables a model needs but does not have."""
    if not model:
        return []
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
