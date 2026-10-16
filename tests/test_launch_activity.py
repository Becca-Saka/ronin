"""Naming the activity a launch intent has to start.

An intent with no component matches every launcher on the device, and Android
refuses it rather than choosing. aapt reports no launchable activity for apps
that put the MAIN/LAUNCHER filter on an <activity-alias>, which is how apps
offer a choice of icon, so the name has to come from somewhere else.
"""

import unittest
from unittest.mock import patch

from ronin.agent.device.adb_control import AdbControl
from ronin.agent.device.manifest import enabled_launcher, launcher_entries
from ronin.errors import DeviceUnavailable

NS = "http://schemas.android.com/apk/res/android"


def alias(name, enabled=None, launcher=True, tag="activity-alias"):
    """One manifest node, shaped like an aapt2 xmltree dump."""
    lines = [f"          E: {tag} (line=77)"]
    lines.append(f'            A: {NS}:name(0x01010003)="{name}" (Raw: "{name}")')
    if enabled is not None:
        lines.append(f"            A: {NS}:enabled(0x0101000e)={str(enabled).lower()}")
    if launcher:
        lines.append("              E: intent-filter (line=82)")
        lines.append("                E: action (line=83)")
        lines.append(f'                    A: {NS}:name(0x01010003)="android.intent.action.MAIN" (Raw: "x")')
        lines.append("                E: category (line=85)")
        lines.append(
            f'                    A: {NS}:name(0x01010003)="android.intent.category.LAUNCHER" (Raw: "x")'
        )
    return "\n".join(lines)


class ManifestLauncherTests(unittest.TestCase):
    """Reading launcher entries out of the manifest when aapt reports none."""

    def test_the_one_enabled_icon_is_the_launcher(self):
        """Fossify ships 19 aliases, one per icon colour, and disables all but one."""
        dump = "\n".join(
            [
                alias("org.fossify.calendar.activities.SplashActivity.Grey_black", enabled=False),
                alias("org.fossify.calendar.activities.SplashActivity.Green", enabled=True),
                alias("org.fossify.calendar.activities.SplashActivity.Teal", enabled=False),
            ]
        )

        self.assertEqual(
            enabled_launcher(dump, "org.fossify.calendar"),
            "org.fossify.calendar.activities.SplashActivity.Green",
        )

    def test_an_absent_enabled_attribute_counts_as_enabled(self):
        """Wikipedia's DefaultIcon alias omits it; Android treats that as true."""
        dump = "\n".join(
            [alias("org.wikipedia.Alpha", enabled=False), alias("org.wikipedia.DefaultIcon")]
        )

        self.assertEqual(enabled_launcher(dump, "org.wikipedia"), "org.wikipedia.DefaultIcon")

    def test_a_disabled_alias_is_never_chosen(self):
        """Starting one fails in a way that reads like the app is broken."""
        dump = alias("org.fossify.notes.activities.SplashActivity.Red", enabled=False)

        self.assertIsNone(enabled_launcher(dump, "org.fossify.notes"))

    def test_two_enabled_launchers_are_left_alone(self):
        """Nothing in the APK says which one the device would show."""
        dump = "\n".join([alias("com.example.One"), alias("com.example.Two")])

        self.assertIsNone(enabled_launcher(dump, "com.example"))

    def test_a_plain_activity_counts_too(self):
        dump = alias("com.example.MainActivity", tag="activity")

        self.assertEqual(enabled_launcher(dump, "com.example"), "com.example.MainActivity")

    def test_an_activity_without_the_launcher_category_is_ignored(self):
        """Most activities declare neither MAIN nor LAUNCHER."""
        dump = alias("com.example.SettingsActivity", tag="activity", launcher=False)

        self.assertEqual(launcher_entries(dump, "com.example"), [])

    def test_the_actions_own_name_is_not_mistaken_for_the_activity(self):
        """android:name appears on the node and on every action and category."""
        dump = alias("com.example.MainActivity")

        self.assertEqual(
            [entry.name for entry in launcher_entries(dump, "com.example")],
            ["com.example.MainActivity"],
        )

    def test_a_relative_class_name_is_expanded_against_the_package(self):
        dump = alias(".ui.MainActivity")

        self.assertEqual(enabled_launcher(dump, "com.example"), "com.example.ui.MainActivity")


class DeviceResolutionTests(unittest.TestCase):
    """Asking the device what the launcher would start."""

    BRIEF = (
        "priority=0 preferredOrder=0 match=0x108000 specificIndex=-1 isDefault=false\n"
        "org.fossify.calendar/.activities.SplashActivity.Green\n"
    )

    def _resolve(self, stdout):
        with patch(
            "ronin.agent.device.adb_control.run_command",
            return_value=type("R", (), {"stdout": stdout})(),
        ):
            return AdbControl.launch_activity_on_device("emulator-5554", "org.fossify.calendar")

    def test_the_last_line_names_the_component(self):
        """The first line is match metadata, not an activity."""
        self.assertEqual(
            self._resolve(self.BRIEF),
            "org.fossify.calendar.activities.SplashActivity.Green",
        )

    def test_no_activity_found_resolves_to_nothing(self):
        self.assertIsNone(self._resolve("No activity found\n"))

    def test_a_component_for_another_package_is_refused(self):
        """A chooser or a resolver activity would launch the wrong app."""
        self.assertIsNone(self._resolve("android/com.android.internal.app.ResolverActivity\n"))

    def test_empty_output_resolves_to_nothing(self):
        self.assertIsNone(self._resolve(""))


class ResolutionOrderTests(unittest.TestCase):
    """Which source gets to name the activity, and what happens when none can."""

    APK = "/tmp/app.apk"
    PACKAGE = "org.fossify.calendar"

    def _resolve(self, on_device, xmltree, badged, serial="emulator-5554"):
        with (
            patch.object(AdbControl, "launch_activity_on_device", return_value=on_device),
            patch.object(AdbControl, "dump_manifest", return_value=xmltree),
            patch.object(AdbControl, "get_apk_info", return_value=(self.PACKAGE, badged)),
        ):
            return AdbControl.resolve_launch_activity(self.APK, self.PACKAGE, serial)

    def test_the_device_outranks_the_apk(self):
        """A user who changed the app icon made the manifest's default wrong."""
        result = self._resolve(
            on_device="org.fossify.calendar.activities.SplashActivity.Red",
            xmltree=alias("org.fossify.calendar.activities.SplashActivity.Green", enabled=True),
            badged="org.fossify.calendar.Badged",
        )

        self.assertEqual(result, "org.fossify.calendar.activities.SplashActivity.Red")

    def test_the_manifest_answers_when_there_is_no_device_yet(self):
        """Nothing can be asked before the emulator is up."""
        result = self._resolve(
            on_device=None,
            xmltree=alias("org.fossify.calendar.activities.SplashActivity.Green", enabled=True),
            badged=None,
            serial=None,
        )

        self.assertEqual(result, "org.fossify.calendar.activities.SplashActivity.Green")

    def test_badging_answers_when_the_manifest_cannot_choose(self):
        result = self._resolve(
            on_device=None,
            xmltree="\n".join([alias("com.example.One"), alias("com.example.Two")]),
            badged="org.fossify.calendar.Badged",
        )

        self.assertEqual(result, "org.fossify.calendar.Badged")

    def test_naming_no_activity_stops_the_run(self):
        """Passing None on produces an adb error about intents three layers down."""
        with self.assertRaises(DeviceUnavailable) as caught:
            self._resolve(on_device=None, xmltree=None, badged=None)

        self.assertIn(self.PACKAGE, str(caught.exception))
