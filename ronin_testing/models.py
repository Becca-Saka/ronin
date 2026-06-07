from pydantic import BaseModel, Field
from typing import List, Literal, Optional

# Trail models live in agent/trail.py — re-exported here for backward compatibility.
from agent.trail import SessionTrail, TrailElement, ScreenEntry  # noqa: F401


class UserJourney(BaseModel):
    """Represents a high-level natural language user journey to test."""
    id: str
    name: str
    description: str
    security_rules: List[str] = Field(default_factory=list)


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
    expected_outcomes: List[str]


class BoundingBox(BaseModel):
    """Represents coordinates highlighting a visual issue."""
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class QAIssue(BaseModel):
    """Represents a visual, security, or logical issue found during testing."""
    issue_type: Literal[
        "render_overflow",
        "text_clipping",
        "overlapping_elements",
        "empty_image_error",
        "crash_screen",
        "sensitive_data_exposure",
        "unmasked_input",
        "insecure_connection",
        "broken_navigation",
        "accessibility_violation",
        "performance_hang"
    ]
    description: str
    bounding_box: Optional[BoundingBox] = None
    severity: Literal["low", "medium", "high", "critical"]
    current_screen: Optional[str] = None


class RuleVerification(BaseModel):
    """Represents the status of verifying a specific journey expectation or custom rule."""
    rule: str
    status: Literal["passed", "failed", "untested"]
    reasoning: str
    screen_name: Optional[str] = None


class ScenarioReport(BaseModel):
    """Represents the results of running an individual test scenario."""
    test_id: str
    scenario_name: str
    classification: str
    status: Literal["PASSED", "FAILED", "SKIPPED"]
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
    status: Literal["PASSED", "FAILED"]
    total_duration: float = 0.0
    scenarios: List[ScenarioReport] = Field(default_factory=list)
    passed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
