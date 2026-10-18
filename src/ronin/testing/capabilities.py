"""What a scenario needs, and whether this target can give it.

Two kinds of refusal live here, and both happen before a scenario spends a
step finding out.

The first reads the scenario's prose for evidence Ronin does not collect. The second reads the capabilities the scenario declared against the target in
front of it. Support is not a property of the platform alone. `adb emu geo fix`
needs an emulator console, `cmd locale` arrived in API 33, and a screen-reader
toggle needs the screen reader installed.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

from ronin.agent.actuation.actions import CapabilityKind
from ronin.agent.device.receipts import Receipt
from ronin.agent.device.capability_maps import TALKBACK_PACKAGE
from ronin.agent.device.platform import Platform
from ronin.testing.models import GeneratedTestCase

logger = logging.getLogger(__name__)




_UNSUPPORTED_REQUIREMENTS = {
    "network inspection": "network traffic capture is not configured",
    "https": "HTTP transport inspection is not configured",
    "http header": "HTTP request inspection is not configured",
    "authorization header": "HTTP request inspection is not configured",
    "accessibility tree": "semantic accessibility-tree inspection is not configured",
    "accessible label": "semantic accessibility-tree inspection is not configured",
    "accessible name": "semantic accessibility-tree inspection is not configured",
    "screen reader": "screen-reader verification is not configured",
    "focus order": "keyboard/screen-reader focus traversal is not configured",
    "contrast ratio": "deterministic contrast measurement is not configured",
    "wcag": "deterministic contrast measurement is not configured",
    "colour contrast": "deterministic contrast measurement is not configured",
    "color contrast": "deterministic contrast measurement is not configured",
}


@dataclass(frozen=True)
class Target:
    """The device a scenario would run on, as far as support is concerned.

    `api_level` and `packages` are optional because a caller may not know
    them. Unknown never refuses a scenario. A requirement nothing can confirm is
    reported as met, so a missing read fails closed on the refusal and not
    on the run.
    """

    platform: Platform
    emulator: bool
    api_level: Optional[int] = None
    packages: Optional[frozenset[str]] = None

    def describe(self) -> str:
        kind = "emulator" if self.emulator else "physical device"
        api = f" API {self.api_level}" if self.api_level else ""
        return f"{self.platform.value} {kind}{api}"


@dataclass(frozen=True)
class Support:
    """What one capability needs of a target on one platform.

    `available` is false for a capability the platform offers no working path
    to at all. Spelled out rather than left out of the table, so the reason
    travels with the refusal.
    """

    available: bool = True
    emulator_only: bool = False
    min_api: Optional[int] = None
    needs_package: Optional[str] = None
    detail: str = ""


# A platform absent from a capability's entry cannot apply it at all. The
# controller refuses the same cases at the device. This is that knowledge
# early enough to skip a scenario instead of failing a step inside it.
CAPABILITY_SUPPORT: dict[CapabilityKind, dict[Platform, Support]] = {
    CapabilityKind.LOCATION: {
        Platform.ANDROID: Support(
            emulator_only=True, detail="'adb emu geo fix' needs an emulator console"
        ),
        Platform.IOS: Support(
            emulator_only=True, detail="'simctl location' needs a simulator"
        ),
    },
    CapabilityKind.NETWORK: {
        Platform.ANDROID: Support(),
    },
    CapabilityKind.BATTERY: {
        Platform.ANDROID: Support(),
        Platform.IOS: Support(
            emulator_only=True, detail="'simctl status_bar' needs a simulator"
        ),
    },
    CapabilityKind.TIMEZONE: {
        Platform.ANDROID: Support(
            available=False,
            detail=(
                "the system zone is persist.sys.timezone, which the shell user "
                "cannot set, and a write to 'settings global timezone' changes "
                "nothing the device reads"
            ),
        ),
    },
    CapabilityKind.LOCALE: {
        Platform.ANDROID: Support(
            min_api=33, detail="per-app locales arrived in Android 13"
        ),
    },
    CapabilityKind.PERMISSION: {
        Platform.ANDROID: Support(),
        Platform.IOS: Support(
            emulator_only=True, detail="'simctl privacy' needs a simulator"
        ),
    },
    CapabilityKind.INTERRUPT: {
        Platform.ANDROID: Support(),
        Platform.IOS: Support(
            emulator_only=True, detail="'simctl notify_post' needs a simulator"
        ),
    },
    CapabilityKind.BIOMETRIC: {
        Platform.ANDROID: Support(
            emulator_only=True, detail="a fingerprint can only be injected into an emulator"
        ),
        Platform.IOS: Support(
            emulator_only=True, detail="biometric enrolment needs a simulator"
        ),
    },
    CapabilityKind.CLIPBOARD: {
        Platform.ANDROID: Support(),
        Platform.IOS: Support(),
    },
    CapabilityKind.DEEPLINK: {
        Platform.ANDROID: Support(),
        Platform.IOS: Support(
            emulator_only=True, detail="'simctl openurl' needs a simulator"
        ),
    },
    CapabilityKind.APP_STATE: {
        Platform.ANDROID: Support(),
        Platform.IOS: Support(
            emulator_only=True, detail="'simctl terminate' needs a simulator"
        ),
    },
    CapabilityKind.MEDIA_INJECT: {
        Platform.ANDROID: Support(),
        Platform.IOS: Support(
            emulator_only=True, detail="'simctl addmedia' needs a simulator"
        ),
    },
    CapabilityKind.ACCESSIBILITY: {
        Platform.ANDROID: Support(
            needs_package=TALKBACK_PACKAGE,
            detail="the screen reader must be installed to be switched on",
        ),
    },
}


def unsupported_capability(kind: CapabilityKind, target: Target) -> Optional[str]:
    """Why this target cannot apply this capability, or None when it can."""
    support = CAPABILITY_SUPPORT.get(kind, {}).get(target.platform)
    if support is None:
        return f"{kind.value} is not available on {target.platform.value}"

    if not support.available:
        return f"{kind.value} cannot be set on {target.platform.value}: {support.detail}"
    if support.emulator_only and not target.emulator:
        return f"{kind.value} needs an emulator: {support.detail}"
    if (
        support.min_api is not None
        and target.api_level is not None
        and target.api_level < support.min_api
    ):
        return (
            f"{kind.value} needs API {support.min_api} and "
            f"{target.describe()} is below it: {support.detail}"
        )
    if (
        support.needs_package
        and target.packages is not None
        and support.needs_package not in target.packages
    ):
        return (
            f"{kind.value} needs {support.needs_package} installed: {support.detail}"
        )
    return None


def unsupported_reason(
    test_case: GeneratedTestCase, target: Optional[Target] = None
) -> str | None:
    """Why a scenario cannot be evidenced here, or None when it can.

    `target` is optional and defaults to reading nothing about the device. A
    caller that does not know the target then keeps the prose rules it had,
    rather than refusing every scenario that declares a capability.
    """
    text = "\n".join(
        [
            *test_case.steps_instructions,
            *(outcome.statement for outcome in test_case.expected_outcomes),
        ]
    ).lower()
    reasons = {
        reason for phrase, reason in _UNSUPPORTED_REQUIREMENTS.items() if phrase in text
    }

    if target is not None:
        for kind in test_case.required_capabilities:
            refusal = unsupported_capability(kind, target)
            if refusal:
                reasons.add(f"{refusal} (target: {target.describe()})")

    if not reasons:
        return None
    return "Unsupported evidence requirement(s): " + "; ".join(sorted(reasons))


def target_of(device_control) -> Target:
    """The device in front of us, as far as capability support is concerned.

    Reads are best-effort. A property this cannot establish stays None, which reports every
    requirement about it as met. A scenario refused on a read that failed
    would skip work the target can do.
    """
    from ronin.agent.device.adb_control import AdbControl

    platform = device_control.platform
    serial = getattr(device_control, "serial", None) or ""

    if platform != Platform.ANDROID:
        # simctl lists simulators and nothing else, so a udid it does not list
        # is a real iPhone, which can have almost none of these capabilities.
        # Unknown reads as a simulator, because that is the reading that
        # refuses nothing, the same way an unread API level does.
        from ronin.agent.device.simctl_control import SimctlControl

        simulated = SimctlControl.is_simulator(serial) if serial else None
        target = Target(platform=platform, emulator=simulated is not False)
        logger.info("Capability target: %s", target.describe())
        return target

    if not serial:
        return Target(platform=platform, emulator=True)

    emulator = bool(AdbControl.avd_name(serial))
    api = AdbControl.ask(serial, "getprop", "ro.build.version.sdk")
    try:
        api_level = int(api) if api else None
    except ValueError:
        api_level = None

    packages = AdbControl.installed_packages(serial)
    target = Target(
        platform=platform,
        emulator=emulator,
        api_level=api_level,
        packages=frozenset(packages) if packages else None,
    )
    logger.info("Capability target: %s", target.describe())
    return target


def unconfirmed_capabilities(
    required: Sequence[CapabilityKind], receipts: Sequence[Receipt]
) -> list[str]:
    """The declared capabilities a run never had confirmed on the device.

    A scenario declares what state it needs. Preflight answers whether this
    target could give it, which is a different question from whether the run
    did. Without this a scenario can declare `network`, stay online for every
    step, and pass on outcomes that were never about the state it asked for.

    A capability no receipt mentions was never applied. One whose receipts are all unreadable was applied and never confirmed. That
    is the same gap from the other side, because nothing establishes the
    state existed.
    """
    confirmed = {receipt.capability for receipt in receipts if receipt.confirmed}
    applied = {receipt.capability for receipt in receipts}
    missing = []
    for kind in required:
        name = kind.value
        if name in confirmed:
            continue
        missing.append(
            f"{name} (applied but never confirmed)" if name in applied
            else f"{name} (never applied)"
        )
    return missing
