from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional

# Trail models live in agent/trail.py — re-exported here for backward compatibility.
from ronin.agent.trail import SessionTrail, TrailElement, ScreenEntry  # noqa: F401


class UserJourney(BaseModel):
    """Represents a high-level natural language user journey to test."""
    id: str
    name: str
    description: str
    security_rules: List[str] = Field(default_factory=list)
    secrets: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Name-to-value pairs the journey declared sensitive. Masked by name "
            "in written artifacts, never withheld from the planner."
        ),
    )


class GeneratedTestCase(BaseModel):
    """Represents a generated test scenario targeting a specific classification."""
    test_id: str
    scenario_name: str
    classification: Literal[
        "happy_path",
        "edge_case",
        "boundary_test",
        "destructive_monkey",
        "security_compliance",
        "accessibility_audit",
        "performance_bottleneck"
    ]
    steps_instructions: List[str]
    expected_outcomes: List[str] = Field(default_factory=list)


class BoundingBox(BaseModel):
    """Represents coordinates highlighting a visual issue."""
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class QAIssue(BaseModel):
    """Represents a visual, security, or logical issue found during testing."""
    issue_type: str = Field(
        ...,
        description=(
            "Category of the issue. Prefer: render_overflow, text_clipping, overlapping_elements, "
            "empty_image_error, crash_screen, sensitive_data_exposure, unmasked_input, "
            "insecure_connection, broken_navigation, accessibility_violation, performance_hang."
        ),
    )
    description: str
    bounding_box: Optional[BoundingBox] = None
    severity: Literal["low", "medium", "high", "critical"]
    current_screen: Optional[str] = None


class RuleVerification(BaseModel):
    """Represents the status of verifying a specific journey expectation or custom rule."""
    rule: str
    rule_id: Optional[str] = Field(
        default=None,
        description=(
            "Stable ID supplied in the prompt, such as expected:0 or security:1. "
            "Dynamic assertions may omit it."
        ),
    )
    status: Literal["passed", "failed", "untested"]
    applicable: bool = Field(
        default=True,
        description=(
            "False when this screen has nothing the rule could apply to, so a "
            "'passed' here would mean nothing. A card-masking rule is not "
            "applicable on a screen with no card field."
        ),
    )
    reasoning: str
    screen_name: Optional[str] = None


class ScenarioReport(BaseModel):
    """Represents the results of running an individual test scenario."""
    test_id: str
    scenario_name: str
    classification: str
    status: Literal[
        "PASSED",
        "FAILED",
        "SKIPPED",
        "ERROR",
        "ABORTED",
        "UNVERIFIED",
        "UNSUPPORTED",
    ]
    duration: float = 0.0
    steps_executed: int = 0
    rule_verifications: List[RuleVerification] = Field(default_factory=list)
    detected_issues: List[QAIssue] = Field(default_factory=list)
    screenshots: List[str] = Field(default_factory=list)
    failure_reason: Optional[str] = None


class JourneyReport(BaseModel):
    """Represents the aggregated results of all scenarios generated for a journey."""
    journey_id: str
    journey_name: str
    description: str
    status: Literal["PASSED", "FAILED", "RUNNING"]
    total_duration: float = 0.0
    scenarios: List[ScenarioReport] = Field(default_factory=list)
    unevidenced_security_rules: List[str] = Field(
        default_factory=list,
        description=(
            "Security rules no scenario was able to evidence. A rule nothing "
            "checked is not a rule that held."
        ),
    )
    passed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
