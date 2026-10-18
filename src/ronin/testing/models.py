from enum import Enum

from pydantic import BaseModel, Field, model_validator
from typing import Dict, List, Literal, Optional

# Trail models live in agent/trail.py — re-exported here for backward compatibility.
from ronin.agent.actuation.actions import CapabilityKind
from ronin.agent.device.launch import SetupMode  # noqa: F401
from ronin.agent.memory.trail import SessionTrail, TrailElement, ScreenEntry  # noqa: F401


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


class AssertionScope(str, Enum):
    """What kind of evidence an expected outcome needs.

    The auditor is handed one screenshot. Asked whether an app hung, or
    whether it navigated somewhere, it answers from pixels that cannot carry
    either fact. Both a false pass and a false missing-evidence follow. The
    scope says which evidence settles the claim, so a claim nothing can settle
    is visible instead of silently wrong.
    """

    CURRENT_SCREEN = "current_screen"
    """True or false about what is on one screen. The auditor can see this."""

    TRANSITION = "transition"
    """The run reached a screen, or moved between two. The screens visited
    carry this; a single screenshot does not."""

    RUN_TIMING = "run_timing"
    """How long the app took, or whether it stayed responsive. Nothing Ronin
    records measures this today, so an outcome scoped here is reported as
    unproven rather than counted either way."""


class ExpectedOutcome(BaseModel):
    """One thing a scenario must show, and what could show it."""

    statement: str
    scope: AssertionScope = Field(
        default=AssertionScope.CURRENT_SCREEN,
        description=(
            "Which evidence settles this. 'current_screen' for anything visible "
            "on a single screen. 'transition' for having reached or moved "
            "between screens. 'run_timing' for speed or responsiveness."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_a_bare_statement(cls, value):
        """A plain string is a claim about one screen, which is the common case."""
        if isinstance(value, str):
            return {"statement": value}
        return value


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
    # Required, and min_length reaches the model as minItems, so the schema
    # the model is given says that one is needed. A scenario that declares no
    # outcome can only report as unverified.
    expected_outcomes: List[ExpectedOutcome] = Field(
        ...,
        min_length=1,
        description=(
            "What must be true for this scenario to pass. Give at least one. "
            "A separate component judges these, so a scenario that declares "
            "none cannot be judged at all."
        ),
    )
    required_capabilities: List[CapabilityKind] = Field(
        default_factory=list,
        description=(
            "Device states this scenario cannot run without, named by kind. "
            "Declare one only where the scenario is about that state, such as "
            "'network' for a scenario about losing connectivity. A target that "
            "cannot apply one skips the scenario instead of testing something "
            "else."
        ),
    )
    setup: SetupMode = Field(
        default=SetupMode.CLEAR_DATA,
        description=(
            "The state this scenario needs before its first step. "
            "'clear_data' wipes the app so it starts as a first install would. "
            "'relaunch' restarts it and keeps any session. "
            "'preserve' leaves the app where the previous scenario left it."
        ),
    )


# The model writes this one to point at what it reports. The detector's
# perception.BoundingBox stays separate, so neither schema moves when the other
# changes.
class BoundingBox(BaseModel):
    """Represents coordinates highlighting a visual issue."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int


class Rendering(str, Enum):
    """How a sensitive value appears on screen.

    An exposure claim is only as good as what was seen. Asked for a sentence,
    a model can call a bare "CVV:" label a plain-text CVV and pass the rule
    forbidding one in the same answer. Asked which of these it saw, it has to
    pick, and the three are not confusable.
    """

    VALUE = "value"
    """The real characters are legible. This is the exposure."""

    MASKED = "masked"
    """Dots, asterisks or bullets stand in for the characters."""

    LABEL_ONLY = "label_only"
    """A caption such as "CVV:" with no value beside it."""

    ABSENT = "absent"
    """Nothing of the kind is on this screen."""


class SensitiveFinding(BaseModel):
    """What was seen where a sensitive value could be, and what it looked like."""

    field: str = Field(..., description="What the value is, e.g. 'CVV' or 'card number'.")

    @model_validator(mode="before")
    @classmethod
    def _accept_the_names_a_finding_arrives_under(cls, value):
        """Reads a finding under the names its two fields arrive with.

        How a value was rendered comes back as `rendering`, `state`, `status`
        or `type`. `type` is the ambiguous one: it names the value in some
        answers and its rendering in others. The renderings are a closed set,
        so which one it carries can be read off rather than guessed.

        A missing name falls back to where the value was seen, which is what
        the model gave instead of naming it.
        """
        if not isinstance(value, dict):
            return value
        finding = dict(value)
        named = finding.pop("type", None)
        if named in {rendering.value for rendering in Rendering}:
            finding.setdefault("rendering", named)
        elif named is not None:
            finding.setdefault("field", named)
        for alias in ("state", "status"):
            if "rendering" not in finding and alias in finding:
                finding["rendering"] = finding.pop(alias)
        if "field" not in finding and "location" in finding:
            finding["field"] = finding["location"]
        if finding.get("quoted") is None:
            finding["quoted"] = ""
        return finding
    rendering: Rendering = Field(
        ...,
        description=(
            "Exactly what is on screen. 'value' only when the real characters "
            "are legible. A caption with nothing beside it is 'label_only'. "
            "Dots or bullets are 'masked'."
        ),
    )
    quoted: str = Field(
        default="",
        description="The characters on screen, copied exactly. Empty unless rendering is 'value'.",
    )
    screen_name: Optional[str] = Field(
        default=None,
        description="Filled in by Ronin, not by the model, from the screen audited.",
    )

    def is_exposure(self) -> bool:
        """Whether this is a leak, decided by the rendering rather than by prose."""
        return self.rendering is Rendering.VALUE


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


STATUS_SPELLINGS = {
    "pass": "passed",
    "passed": "passed",
    "fail": "failed",
    "failed": "failed",
    "untested": "untested",
    "not_tested": "untested",
}
"""How a verdict's status is spelled, mapped to the three that are accepted."""


class RuleVerification(BaseModel):
    """Represents the status of verifying a specific journey expectation or custom rule."""
    rule: str = Field(
        ...,
        description="The words of the rule being answered, as they were listed.",
    )
    rule_id: Optional[str] = Field(
        default=None,
        description=(
            "Stable ID supplied in the prompt, such as expected:0 or security:1, "
            "and nothing else. Dynamic assertions may omit it."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_the_shapes_a_verdict_arrives_in(cls, value):
        """Reads a verdict however the model chose to write it.

        One rejected verdict takes the whole assessment with it, and the
        verdicts beside it were answers to other rules.

        The id is the part the gate needs, and the gate matches it whole, so
        it is read out of whichever field holds it and out of whatever else
        was written around it. It stands in for `rule` when only the id was
        given: an id names its rule as well as the words do.

        An unrecognised status is left alone to be refused. A verdict is the
        thing being recorded, and guessing at one invents evidence.
        """
        if not isinstance(value, dict):
            return value
        from ronin.testing import rules

        verdict = dict(value)
        found = rules.id_in(verdict.get("rule_id")) or rules.id_in(verdict.get("rule"))
        if found:
            verdict["rule_id"] = found
        if not verdict.get("rule") and verdict.get("rule_id"):
            verdict["rule"] = verdict["rule_id"]
        verdict["status"] = STATUS_SPELLINGS.get(
            str(verdict.get("status", "")).strip().lower().replace(" ", "_"),
            verdict.get("status"),
        )
        return verdict
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
    cause: Optional[str] = Field(
        default=None,
        description=(
            "Whose problem a scenario that reached no verdict is: the device, "
            "a model provider, Ronin, the evidence or the scenario itself. "
            "Absent on PASSED and on FAILED, where the app is the answer."
        ),
    )
    duration: float = 0.0
    steps_executed: int = 0
    rule_verifications: List[RuleVerification] = Field(default_factory=list)
    detected_issues: List[QAIssue] = Field(default_factory=list)
    screenshots: List[str] = Field(default_factory=list)
    failure_reason: Optional[str] = None


class RunEnvironment(BaseModel):
    """What produced a report, for when the same journey behaves differently."""
    ronin_version: str
    model: str
    fast_model: Optional[str] = None
    platform: str
    grounding: str
    detector: Optional[str] = Field(
        default=None, description="Detection weights, as repository@revision."
    )


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
    environment: Optional[RunEnvironment] = None
