"""What a run leaves behind: the evidence, and the reports built from it.

Split from `models.py`, which holds what a run is given. These are what it
produces. One module answered both questions and grew past the file ceiling.

Every name here is re-exported by `models.py`, so no caller has to know
which of the two it came from.
"""

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from ronin.agent.run.record import StepRecord
from ronin.identity import AppIdentity, CiIdentity, DeviceIdentity, GitIdentity


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
    a model can call a bare caption a legible value and pass the rule
    forbidding one in the same answer. Asked which of these it saw, it has to
    pick, and the four are not confusable.
    """

    VALUE = "value"
    """The real characters are legible. This is the exposure."""

    MASKED = "masked"
    """Dots, asterisks or bullets stand in for the characters."""

    LABEL_ONLY = "label_only"
    """A caption naming the field, with no value beside it."""

    ABSENT = "absent"
    """Nothing of the kind is on this screen."""


class SensitiveFinding(BaseModel):
    """What was seen where a sensitive value could be, and what it looked like."""

    field: str = Field(
        ..., description="What the value is, named as this screen names it."
    )

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


Status = Literal[
    "PASSED",
    "FAILED",
    "SKIPPED",
    "ERROR",
    "ABORTED",
    "UNVERIFIED",
    "UNSUPPORTED",
]
"""The words a scenario can end on.

Named because two places hold one of them: the judge that decides it, and the
report that carries it. A second copy of the list would let one grow a word
the other refuses.
"""


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
        from ronin.testing.judge import rules

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
            "'passed' here would mean nothing. A rule about a field is not "
            "applicable on a screen without that field."
        ),
    )
    reasoning: str
    screen_name: Optional[str] = None


class ScenarioReport(BaseModel):
    """Represents the results of running an individual test scenario."""
    scenario_uuid: Optional[str] = Field(
        default=None,
        description=(
            "Which scenario this reports on. Absent on a report Ronin writes "
            "about itself, which answered no scenario."
        ),
    )
    test_id: str
    scenario_name: str
    cell: str = Field(
        description=(
            "The cell this scenario answered, such as 'security.negative.0'. "
            "A report Ronin writes about itself rather than a scenario carries "
            "'infrastructure'."
        )
    )
    status: Status
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
    steps: List[StepRecord] = Field(
        default_factory=list,
        description="What each step of this scenario did, in order.",
    )
    failure_reason: Optional[str] = None


class RunEnvironment(BaseModel):
    """What produced a report, for when the same journey behaves differently.

    Ronin's own settings, and the four things around them that decide what a
    run means: the build, the target, the commit and the job. A scenario that
    failed, with no record of which build it failed on, is not evidence of a
    defect.
    """
    ronin_version: str
    model: str
    fast_model: Optional[str] = None
    platform: str
    grounding: str
    detector: Optional[str] = Field(
        default=None, description="Detection weights, as repository@revision."
    )
    app: AppIdentity = Field(default_factory=AppIdentity)
    device: DeviceIdentity = Field(default_factory=DeviceIdentity)
    git: GitIdentity = Field(default_factory=GitIdentity)
    ci: CiIdentity = Field(default_factory=CiIdentity)


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
            "Security rules no scenario was able to evidence. A rule no "
            "scenario reached is not a rule that held."
        ),
    )
    passed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    environment: Optional[RunEnvironment] = None
