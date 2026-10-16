"""The audit is only worth its call if it could have disagreed."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock


def _assessment(**overrides):
    from ronin.testing.qa_auditor import Assessment

    values = {"screen_name": "Login Screen"}
    values.update(overrides)
    return Assessment(**values)


class AuditorInputTests(unittest.TestCase):
    """What the auditor is not told is the point of it."""

    def _auditor(self, llm):
        from ronin.testing.qa_auditor import QAAuditor

        return QAAuditor(llm, ["Order confirmed"], ["Card masked"])

    def _llm(self):
        return SimpleNamespace(generate_response=Mock(return_value=_assessment()))

    def _sent(self, llm):
        (messages, _model), _ = llm.generate_response.call_args
        return messages[0].to_dict()

    def test_the_screenshot_is_sent(self):
        llm = self._llm()
        self._auditor(llm).audit(b"png-bytes", screen_id="abc")

        kinds = [part["type"] for part in self._sent(llm)["content"]]
        self.assertIn("image_url", kinds)

    def test_the_rules_are_sent(self):
        llm = self._llm()
        self._auditor(llm).audit(b"png", screen_id="abc")

        text = " ".join(
            part.get("text", "") for part in self._sent(llm)["content"]
        )
        self.assertIn("[security:0] Card masked", text)

    def test_the_plan_is_not_sent(self):
        """A verdict from something that knows the intent is not independent."""
        llm = self._llm()
        self._auditor(llm).audit(b"png", screen_id="abc")

        text = " ".join(
            part.get("text", "") for part in self._sent(llm)["content"]
        ).lower()
        for leak in ("action", "history", "previous step", "next step", "tap "):
            self.assertNotIn(leak, text, f"the auditor was told about {leak!r}")


class ScreenCachingTests(unittest.TestCase):
    """A screen that has not changed cannot have changed its mind."""

    def _auditor(self, llm):
        from ronin.testing.qa_auditor import QAAuditor

        return QAAuditor(llm, [], [])

    def test_the_same_screen_is_audited_once(self):
        llm = SimpleNamespace(generate_response=Mock(return_value=_assessment()))
        auditor = self._auditor(llm)

        auditor.audit(b"png", screen_id="same")
        auditor.audit(b"png", screen_id="same")

        self.assertEqual(llm.generate_response.call_count, 1)

    def test_a_new_screen_is_audited_again(self):
        llm = SimpleNamespace(generate_response=Mock(return_value=_assessment()))
        auditor = self._auditor(llm)

        auditor.audit(b"png", screen_id="one")
        auditor.audit(b"png", screen_id="two")

        self.assertEqual(llm.generate_response.call_count, 2)

    def test_an_unidentified_screen_is_never_cached(self):
        llm = SimpleNamespace(generate_response=Mock(return_value=_assessment()))
        auditor = self._auditor(llm)

        auditor.audit(b"png", screen_id=None)
        auditor.audit(b"png", screen_id=None)

        self.assertEqual(llm.generate_response.call_count, 2)


class AssessmentScrubbingTests(unittest.TestCase):
    """The auditor writes in its own words, which quote the journey's values."""

    def test_declared_secrets_are_masked_in_what_it_wrote(self):
        from ronin.testing.qa_auditor import QAAuditor
        from ronin.testing.models import QAIssue, RuleVerification
        from ronin.utils.secrets import SecretScrubber

        scrubber = SecretScrubber()
        scrubber.learn_secrets({"password": "hunter2"})
        written = _assessment(
            goal_evidence="the field held hunter2",
            rule_verifications=[
                RuleVerification(
                    rule="Password masked",
                    rule_id="security:0",
                    status="failed",
                    reasoning="hunter2 was visible",
                )
            ],
            detected_issues=[
                QAIssue(
                    issue_type="sensitive_data_exposure",
                    severity="critical",
                    description="hunter2 shown in plain text",
                )
            ],
        )
        llm = SimpleNamespace(generate_response=Mock(return_value=written))

        assessed = QAAuditor(llm, [], ["Password masked"], scrubber=scrubber).audit(b"p", "s")

        self.assertNotIn("hunter2", assessed.goal_evidence)
        self.assertNotIn("hunter2", assessed.rule_verifications[0].reasoning)
        self.assertNotIn("hunter2", assessed.detected_issues[0].description)
        self.assertIn("{{password}}", assessed.goal_evidence)


class CacheAliasingTests(unittest.TestCase):
    """Callers withdraw findings from what they are given, so give them a copy."""

    def _auditor(self):
        from ronin.testing.models import QAIssue
        from ronin.testing.qa_auditor import QAAuditor

        crash = QAIssue(
            issue_type="crash_screen", severity="critical", description="Blank screen"
        )
        llm = SimpleNamespace(
            generate_response=Mock(return_value=_assessment(detected_issues=[crash]))
        )
        return QAAuditor(llm, [], []), llm

    def test_withdrawing_a_finding_does_not_erase_it_from_the_cache(self):
        """A crash dropped on the first look must still be there on the second."""
        auditor, _ = self._auditor()

        first = auditor.audit(b"png", screen_id="same")
        first.detected_issues = []
        second = auditor.audit(b"png", screen_id="same")

        self.assertEqual(len(second.detected_issues), 1)

    def test_the_screen_is_still_only_audited_once(self):
        auditor, llm = self._auditor()

        auditor.audit(b"png", screen_id="same")
        auditor.audit(b"png", screen_id="same")

        self.assertEqual(llm.generate_response.call_count, 1)
