"""Deciding whether a target can give a scenario what it declared.

Support is not a property of the platform alone. The emulator console, the API
level and the installed tools all decide it, and a scenario that needs one this
target lacks is skipped before it spends a step finding out.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from ronin.agent.actuation.actions import CapabilityKind
from ronin.agent.device.platform import Platform

SERIAL = "emulator-5554"


class PreflightTests(unittest.TestCase):
    """What a scenario declares, read against the target in front of it."""

    def _case(self, *required, steps=("open the app",)):
        from ronin.testing.models import GeneratedTestCase

        return GeneratedTestCase(
            test_id="TS_1",
            scenario_name="S",
            classification="edge_case",
            steps_instructions=list(steps),
            expected_outcomes=["a banner appears"],
            required_capabilities=list(required),
        )

    def _target(self, platform=Platform.ANDROID, **kwargs):
        from ronin.testing.capabilities import Target

        return Target(platform=platform, **{"emulator": True, **kwargs})

    def _reason(self, case, target):
        from ronin.testing.capabilities import unsupported_reason

        return unsupported_reason(case, target)

    def test_a_capability_the_target_can_apply_is_no_obstacle(self):
        self.assertIsNone(
            self._reason(self._case(CapabilityKind.NETWORK), self._target())
        )

    def test_a_capability_the_platform_lacks_skips_the_scenario(self):
        reason = self._reason(
            self._case(CapabilityKind.NETWORK), self._target(Platform.IOS)
        )

        self.assertIn("network", reason)
        self.assertIn("ios", reason)

    def test_a_capability_needing_an_emulator_names_the_target(self):
        """Mock location goes through the emulator console, which a phone lacks."""
        reason = self._reason(
            self._case(CapabilityKind.LOCATION), self._target(emulator=False)
        )

        self.assertIn("emulator", reason)
        self.assertIn("physical device", reason)

    def test_an_api_below_the_floor_skips_the_scenario(self):
        reason = self._reason(
            self._case(CapabilityKind.LOCALE), self._target(api_level=31)
        )

        self.assertIn("API 33", reason)

    def test_an_api_at_the_floor_is_enough(self):
        self.assertIsNone(
            self._reason(self._case(CapabilityKind.LOCALE), self._target(api_level=33))
        )

    def test_an_unknown_api_refuses_nothing(self):
        """A read that failed must not skip work the target can do."""
        self.assertIsNone(
            self._reason(self._case(CapabilityKind.LOCALE), self._target(api_level=None))
        )

    def test_a_missing_tool_skips_the_scenario(self):
        reason = self._reason(
            self._case(CapabilityKind.ACCESSIBILITY),
            self._target(packages=frozenset({"com.example.app"})),
        )

        self.assertIn("talkback", reason.lower())

    def test_an_unknown_package_list_refuses_nothing(self):
        self.assertIsNone(
            self._reason(
                self._case(CapabilityKind.ACCESSIBILITY), self._target(packages=None)
            )
        )

    def test_declaring_nothing_leaves_the_prose_check_alone(self):
        reason = self._reason(
            self._case(steps=("Perform network inspection",)), self._target()
        )

        self.assertIn("network traffic capture", reason)

    def test_a_caller_with_no_target_checks_only_the_prose(self):
        """The default cannot refuse a scenario it knows nothing about."""
        from ronin.testing.capabilities import unsupported_reason

        self.assertIsNone(unsupported_reason(self._case(CapabilityKind.NETWORK)))

    def test_every_capability_is_declared_in_the_support_table(self):
        """A capability with no entry would be refused on every target."""
        from ronin.testing.capabilities import CAPABILITY_SUPPORT

        self.assertEqual(set(CAPABILITY_SUPPORT), set(CapabilityKind))


class TargetReadingTests(unittest.TestCase):
    """Identifying the target without a driver session."""

    def _read(self, avd, api="33", packages=frozenset()):
        device = SimpleNamespace(platform=Platform.ANDROID, serial=SERIAL)
        adb = mock.patch(
            "ronin.agent.device.adb_control.AdbControl.avd_name", return_value=avd
        )
        installed = mock.patch(
            "ronin.agent.device.adb_control.AdbControl.installed_packages",
            return_value=packages,
        )
        sdk = mock.patch(
            "ronin.agent.device.adb_control.AdbControl.ask", return_value=api
        )
        from ronin.testing.capabilities import target_of

        with adb, installed, sdk:
            return target_of(device)

    def test_a_device_started_from_an_avd_is_an_emulator(self):
        self.assertTrue(self._read(avd="Pixel_8").emulator)

    def test_a_device_with_no_avd_name_is_physical(self):
        self.assertFalse(self._read(avd=None).emulator)

    def test_the_api_level_is_read_as_a_number(self):
        self.assertEqual(self._read(avd="Pixel_8", api="36").api_level, 36)

    def test_an_api_level_that_is_not_a_number_is_unknown(self):
        self.assertIsNone(self._read(avd="Pixel_8", api="null").api_level)

    def test_a_target_with_no_serial_reads_nothing_off_it(self):
        from ronin.testing.capabilities import target_of

        target = target_of(SimpleNamespace(platform=Platform.ANDROID, serial=""))

        self.assertIsNone(target.api_level)
        self.assertIsNone(target.packages)


class PhysicalDeviceTests(unittest.TestCase):
    """A real iPhone can have almost none of these, and said so when asked.

    Before this, every iOS target was read as a simulator, so eight
    capabilities a physical device cannot apply preflighted as supported. A
    scenario declaring one was run rather than skipped, and the capability
    failed inside it.
    """

    PHYSICAL = "00008140-00096DC43A52801C"
    SIMULATED = "8B59D40E-33B7-40EC-AB5D-50FFA0AB306E"

    def _target(self, listed):
        from ronin.testing.capabilities import target_of

        with mock.patch(
            "ronin.agent.device.simctl_control.SimctlControl.ask",
            return_value=listed,
        ):
            return target_of(
                SimpleNamespace(platform=Platform.IOS, serial=self.PHYSICAL)
            )

    def _allowed(self, target):
        from ronin.testing.capabilities import unsupported_capability

        return {
            kind.value
            for kind in CapabilityKind
            if unsupported_capability(kind, target) is None
        }

    def test_a_udid_simctl_does_not_list_is_a_physical_device(self):
        """simctl enumerates simulators and nothing else."""
        target = self._target(f"-- iOS 18.2 --\n  iPhone 15 ({self.SIMULATED}) (Booted)")

        self.assertFalse(target.emulator)
        self.assertIn("physical", target.describe())

    def test_a_udid_simctl_lists_is_a_simulator(self):
        target = self._target(f"-- iOS 18.2 --\n  iPhone 15 ({self.PHYSICAL}) (Booted)")

        self.assertTrue(target.emulator)

    def test_a_physical_iphone_allows_almost_nothing(self):
        target = self._target("-- iOS 18.2 --\n  (none)")

        self.assertEqual(self._allowed(target), {"clipboard"})

    def test_a_simulator_still_allows_what_it_can_do(self):
        target = self._target(f"  iPhone ({self.PHYSICAL}) (Booted)")

        self.assertIn("location", self._allowed(target))
        self.assertIn("biometric", self._allowed(target))

    def test_a_simctl_that_will_not_answer_refuses_nothing(self):
        """Unknown reads as a simulator, as an unread API level does."""
        target = self._target(None)

        self.assertTrue(target.emulator)


class DeclaredCapabilityTests(unittest.TestCase):
    """A scenario declares the state it needs, and the run has to reach it.

    Preflight answers whether this target could give a capability. Whether the
    run did is a different question, and without it a scenario can declare
    `network`, stay online for every step, and pass on outcomes that were never
    about the state it asked for.
    """

    def _receipt(self, capability, observed, requested="on"):
        from ronin.agent.device.receipts import Receipt

        return Receipt(
            capability=capability,
            requested=requested,
            observed=observed,
            target=SERIAL,
            source="mobile: getConnectivity",
        )

    def _missing(self, required, receipts):
        from ronin.testing.capabilities import unconfirmed_capabilities

        return unconfirmed_capabilities(required, receipts)

    def test_a_capability_the_run_never_applied_is_named(self):
        missing = self._missing([CapabilityKind.NETWORK], [])

        self.assertEqual(missing, ["network (never applied)"])

    def test_a_capability_applied_and_confirmed_is_not_named(self):
        missing = self._missing(
            [CapabilityKind.NETWORK], [self._receipt("network", "on")]
        )

        self.assertEqual(missing, [])

    def test_a_capability_applied_but_never_confirmed_is_named(self):
        """Unreadable is safe only while something later notices it."""
        missing = self._missing(
            [CapabilityKind.NETWORK], [self._receipt("network", None)]
        )

        self.assertEqual(missing, ["network (applied but never confirmed)"])

    def test_another_capability_s_receipt_does_not_answer_for_this_one(self):
        missing = self._missing(
            [CapabilityKind.NETWORK], [self._receipt("battery", "42", "42")]
        )

        self.assertEqual(missing, ["network (never applied)"])

    def test_declaring_nothing_requires_nothing(self):
        self.assertEqual(self._missing([], []), [])


class DeclaredCapabilityGateTests(unittest.TestCase):
    """The gate refuses a scenario whose declared state was never confirmed."""

    def _status(self, required, receipts):
        from ronin.agent.device.receipts import Receipt
        from ronin.testing.qa_agent import QAAgent

        made = [
            Receipt(
                capability=name,
                requested="on",
                observed=observed,
                target=SERIAL,
                source="s",
            )
            for name, observed in receipts
        ]
        return QAAgent.__new__(QAAgent)._determine_status(
            run_result=SimpleNamespace(
                status="goal_reached",
                failure_reason=None,
                capability_receipts=made,
            ),
            stop_reason=None,
            all_verifications=[],
            all_issues=[],
            qa_outputs=[
                SimpleNamespace(is_scenario_feasible=True, infeasibility_reason=None)
            ],
            assessments=[SimpleNamespace(rule_verifications=[], detected_issues=[])],
            required_outcomes=[],
            security_rules=[],
            required_capabilities=required,
        )

    def test_a_scenario_that_never_went_offline_cannot_pass(self):
        status, reason, cause = self._status([CapabilityKind.NETWORK], [])

        self.assertEqual(status, "UNVERIFIED")
        self.assertIn("network", reason)
        self.assertIn("never applied", reason)

    def test_a_scenario_whose_state_was_confirmed_is_not_held_back_by_it(self):
        """Whatever else the scenario still needs, the capability is settled."""
        _, reason, _ = self._status([CapabilityKind.NETWORK], [("network", "on")])

        self.assertNotIn("network", reason or "")
