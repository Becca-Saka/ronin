import logging
from pydantic import BaseModel, Field
from typing import List
from ronin.llm.llm_client import LLMClient
from ronin.llm.messages import TextMessage
from .models import UserJourney, GeneratedTestCase

logger = logging.getLogger(__name__)

GENERATION_ATTEMPTS = 3
MAX_GENERATED_SCENARIOS = 10

class GeneratedTestCasesList(BaseModel):
    # min_length reaches the model as minItems, so an empty suite is rejected
    # by the schema it was given rather than only after it has been paid for.
    scenarios: List[GeneratedTestCase] = Field(
        ..., min_length=1, max_length=MAX_GENERATED_SCENARIOS
    )

class ScenarioGenerator:
    """Uses LLM to decompose a UserJourney description into diverse testing scenarios."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def generate_scenarios(self, journey: UserJourney) -> List[GeneratedTestCase]:
        """Generates scenarios covering happy paths, edge cases, boundaries, and destructive monkeys."""
        logger.info("Decomposing journey into test scenarios...")
        
        prompt = f"""You are a senior QA Test Engineer. Your goal is to analyze the following natural-language User Journey and design a comprehensive suite of E2E test scenarios to thoroughly test the app and try to make it break.

### User Journey Description:
"{journey.description}"

### Custom Security Rules:
{chr(10).join(f"- {rule}" for rule in journey.security_rules) if journey.security_rules else "None specified."}

### Scenarios Requirement:
Generate a list of scenarios covering:
1. **happy_path**: The standard successful user story flow.
2. **edge_case**: Alternative execution flows, error handling, or empty states.
3. **boundary_test**: Inputting extreme, invalid, or very long text values to test limits.
4. **destructive_monkey**: Clicking elements out of order, double taps, or rapid actions.
5. **security_compliance**: Visible masked inputs and custom rules that can be verified from screenshots and detected UI text.
6. **accessibility_audit**: Visible clipping, touch-target clarity, and text alternatives present in detected UI text.
7. **performance_bottleneck**: Visible hangs, long-lived spinners, or unresponsive screens.

Ronin can observe screenshots, detected UI text, action outcomes, and elapsed
scenario time. It cannot inspect network traffic, HTTP headers, logs from the
app, semantic accessibility trees, screen readers, focus order, or measured
contrast ratios. Never generate an instruction or expected outcome that needs
one of those unavailable evidence sources.

Generate at most {MAX_GENERATED_SCENARIOS} scenarios. Prefer a small suite with
distinct evidence over several near-duplicates that repeat the full journey.

### How Scenarios Are Executed:
Each scenario runs on its own, starting from a **freshly launched app** — the
agent sees the app's first screen and nothing else. It has no memory of other
scenarios and cannot resume from where another left off.

So every scenario's instructions MUST start at app launch and include the full
navigation needed to reach the screen it cares about. A scenario about the
checkout screen must first sign in, reach a product, add it, and open checkout.
Never write instructions that assume the agent is already on a later screen.

Identify targets by their visible text, not by position or index: say "the Menu
button on the 'Sweet Treats' card", never "card 0" or "the first card".

Write one instruction per interaction, and never prescribe a separate focusing
tap before entering text — the agent focuses a field as part of typing into it.
Write "Enter 'test@example.com' in the Email field", not "Tap the Email field,
then type 'test@example.com'". Every wasted instruction costs the agent a step
it could have spent reaching the end of the scenario.

For each scenario, define:
- A unique test ID (e.g., TS_001_happy_path)
- A clear name description
- The classification category
- Step-by-step instructions for the executing agent, starting from app launch
- Expected outcomes (logical assertions) the agent must verify
"""
        message = TextMessage(content=prompt)

        last_error = None
        for attempt in range(GENERATION_ATTEMPTS):
            try:
                result: GeneratedTestCasesList = self.llm.generate_response(
                    messages=[message],
                    response_model=GeneratedTestCasesList
                )
                logger.info("Successfully generated %d test scenarios.", len(result.scenarios))
                return result.scenarios
            except Exception as e:
                last_error = e
                logger.warning(
                    "Scenario generation attempt %d/%d failed: %s",
                    attempt + 1,
                    GENERATION_ATTEMPTS,
                    e,
                )

        logger.error(
            "Scenario generation failed after %d attempts: %s", GENERATION_ATTEMPTS, last_error
        )
        raise RuntimeError(f"Scenario generation failed — aborting session: {last_error}") from last_error
