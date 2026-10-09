import logging
from pydantic import BaseModel
from typing import List
from llm.llm_client import LLMClient
from llm.messages import TextMessage
from .models import UserJourney, GeneratedTestCase

logger = logging.getLogger(__name__)

class GeneratedTestCasesList(BaseModel):
    scenarios: List[GeneratedTestCase]

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
5. **security_compliance**: Masked inputs, API token check instructions, and validating custom security rules.
6. **accessibility_audit**: Scans for clear visual indicators, contrast, or unlabeled widgets.
7. **performance_bottleneck**: Spotting hangs, spinner durations, or unresponsive screens.

For each scenario, define:
- A unique test ID (e.g., TS_001_happy_path)
- A clear name description
- The classification category
- Step-by-step instructions for the executing agent
- Expected outcomes (logical assertions) the agent must verify
"""
        message = TextMessage(content=prompt)

        last_error = None
        for attempt in range(2):
            try:
                result: GeneratedTestCasesList = self.llm.generate_response(
                    messages=[message],
                    response_model=GeneratedTestCasesList
                )
                logger.info("Successfully generated %d test scenarios.", len(result.scenarios))
                return result.scenarios
            except Exception as e:
                last_error = e
                logger.warning("Scenario generation attempt %d/2 failed: %s", attempt + 1, e)

        logger.error("Scenario generation failed after 2 attempts: %s", last_error)
        raise RuntimeError(f"Scenario generation failed — aborting session: {last_error}") from last_error
