"""What the eval gate tolerates, what it fails, and what it reports."""

import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from tools.eval import baseline, corpus, tiers
import tools.eval_gate as gate


DIGEST = "0f1e2d3c4b5a6978"


def _answer(item: str, state: str = tiers.WRONG, digest: str = DIGEST) -> tiers.Answer:
    return tiers.Answer(item=item, state=state, detail="because", digest=digest)


def _excused(*items: str, fixture: str = DIGEST) -> dict:
    return {
        item: baseline.Exemption(reason="TICKET-1, the prompt is too long", fixture=fixture)
        for item in items
    }


VERDICTS = frozenset({corpus.VERDICT_TIER})


def _baseline(known_wrong=None, known_flaky=None) -> baseline.Baseline:
    return baseline.Baseline(
        known_wrong=known_wrong or {}, known_flaky=known_flaky or {}
    )


class RatchetTests(unittest.TestCase):
    """A new wrong answer fails. One already accounted for does not."""

    def test_a_wrong_item_the_baseline_does_not_name_is_a_regression(self):
        reading = baseline.Reading(tiers=VERDICTS, answers=[_answer("a-cell")], baseline=_baseline())

        self.assertEqual([found.item for found in reading.regressions], ["a-cell"])

    def test_a_wrong_item_the_baseline_names_is_tolerated(self):
        reading = baseline.Reading(
            tiers=VERDICTS,
            answers=[_answer("a-cell")], baseline=_baseline(known_wrong=_excused("a-cell"))
        )

        self.assertEqual(reading.regressions, [])
        self.assertTrue(reading.holds)

    def test_a_lapsed_exemption_fails_even_where_the_item_now_answers_correctly(self):
        """Otherwise a label edited to match the code clears the item.

        The entry would be left standing, excusing whatever takes its place
        next, and the build would be green for the change that weakened it.
        """
        reading = baseline.Reading(
            tiers=VERDICTS,
            answers=[_answer("a-cell", tiers.CORRECT, digest="aaaaaaaaaaaaaaaa")],
            baseline=_baseline(known_wrong=_excused("a-cell")),
        )

        self.assertEqual(reading.regressions, [])
        self.assertEqual([found.item for found in reading.lapsed], ["a-cell"])
        self.assertEqual(reading.repaired, [])
        self.assertFalse(reading.holds)

    def test_a_baselined_item_of_a_tier_this_sweep_did_not_run_is_left_alone(self):
        """The free run reads the verdict corpus. Every screen entry would
        otherwise read as a fixture somebody deleted."""
        reading = baseline.Reading(
            tiers=VERDICTS,
            answers=[_answer("a-verdict-case", tiers.CORRECT)],
            baseline=_baseline(known_wrong=_excused("a-screen/expected:1")),
        )

        self.assertEqual(reading.absent, [])
        self.assertTrue(reading.holds)

    def test_an_exemption_lapses_when_the_fixture_it_covers_changes(self):
        """An exemption is granted to a case, not to a name.

        Without this, editing the evidence or the label under a name already
        excused would keep the exemption, and the corpus could be weakened
        without the gate saying so.
        """
        reading = baseline.Reading(
            tiers=VERDICTS,
            answers=[_answer("a-cell", digest="aaaaaaaaaaaaaaaa")],
            baseline=_baseline(known_wrong=_excused("a-cell")),
        )

        self.assertEqual([found.item for found in reading.regressions], ["a-cell"])
        self.assertEqual([found.item for found in reading.lapsed], ["a-cell"])
        self.assertIn("covers an older fixture", reading.render())

    def test_a_flaky_item_needs_its_own_entry(self):
        """Wrong every time and disagreeing with itself are different findings."""
        flaky = [_answer("a-cell", tiers.FLAKY)]

        self.assertEqual(
            len(baseline.Reading(answers=flaky, baseline=_baseline()).regressions), 1
        )
        self.assertEqual(
            baseline.Reading(
                tiers=VERDICTS,
                answers=flaky, baseline=_baseline(known_flaky=_excused("a-cell"))
            ).regressions,
            [],
        )

    def test_an_exemption_covers_the_state_it_was_written_for_and_no_other(self):
        """`known_wrong` says a rule is answered wrongly.

        It does not say the auditor may stop answering it. Silence covered by
        a wrong-answer entry would undo the reason silence is classed apart.
        """
        wrong = _baseline(known_wrong=_excused("a-cell"))

        self.assertEqual(
            baseline.Reading(
                tiers=VERDICTS, answers=[_answer("a-cell", tiers.WRONG)], baseline=wrong
            ).regressions,
            [],
        )
        self.assertEqual(
            [
                found.item
                for found in baseline.Reading(
                    tiers=VERDICTS,
                    answers=[_answer("a-cell", tiers.UNANSWERED)],
                    baseline=wrong,
                ).regressions
            ],
            ["a-cell"],
        )
        self.assertEqual(
            [
                found.item
                for found in baseline.Reading(
                    tiers=VERDICTS,
                    answers=[_answer("a-cell", tiers.FLAKY)],
                    baseline=wrong,
                ).regressions
            ],
            ["a-cell"],
        )

    def test_an_unanswered_item_is_a_regression(self):
        reading = baseline.Reading(
            tiers=VERDICTS,
            answers=[_answer("a-cell", tiers.UNANSWERED)], baseline=_baseline()
        )

        self.assertEqual(len(reading.regressions), 1)

    def test_a_baselined_item_that_is_now_correct_is_reported_and_does_not_fail(self):
        """The line comes out in the commit that earned it, not automatically."""
        reading = baseline.Reading(
            tiers=VERDICTS,
            answers=[_answer("a-cell", tiers.CORRECT)],
            baseline=_baseline(known_wrong=_excused("a-cell")),
        )

        self.assertEqual(reading.regressions, [])
        self.assertEqual(reading.repaired, ["a-cell"])
        self.assertIn("take it out of the baseline", reading.render())

    def test_a_baselined_item_the_corpus_no_longer_holds_is_reported(self):
        """A line naming nothing tolerates nothing, and hides what takes the name."""
        reading = baseline.Reading(
            tiers=VERDICTS,
            answers=[_answer("still-here", tiers.CORRECT)],
            baseline=_baseline(known_wrong=_excused("renamed-away")),
        )

        self.assertEqual(reading.absent, ["renamed-away"])
        self.assertFalse(reading.holds)
        self.assertIn("no longer in the corpus", reading.render())

    def test_the_rendering_counts_the_correct_and_names_the_rest(self):
        reading = baseline.Reading(
            tiers=VERDICTS,
            answers=[_answer("good", tiers.CORRECT), _answer("bad")],
            baseline=_baseline(),
        )
        rendered = reading.render()

        self.assertIn("1 of 2 corpus items answered correctly", rendered)
        self.assertIn("wrong: bad", rendered)

    def test_a_tolerated_item_is_rendered_as_baselined(self):
        reading = baseline.Reading(
            tiers=VERDICTS,
            answers=[_answer("bad")], baseline=_baseline(known_wrong=_excused("bad"))
        )

        self.assertIn("(baselined)", reading.render())


class BaselineFileTests(unittest.TestCase):
    """The file on disk, which the gate reads and never writes."""

    def test_the_repositorys_own_baseline_reads(self):
        recorded = baseline.Baseline.read()

        self.assertIsInstance(recorded.known_wrong, dict)
        self.assertIsInstance(recorded.known_flaky, dict)

    def test_an_entry_carries_a_reason_and_the_fixture_it_covers(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.toml"
            path.write_text(
                '[known_wrong."cart/expected:1"]\n'
                'reason = "TICKET-7"\n'
                'fixture = "0f1e2d3c4b5a6978"\n',
                encoding="utf-8",
            )

            entry = baseline.Baseline.read(path).known_wrong["cart/expected:1"]

        self.assertEqual(entry.reason, "TICKET-7")
        self.assertEqual(entry.fixture, "0f1e2d3c4b5a6978")

    def test_an_entry_without_the_fixture_it_covers_is_refused(self):
        """A loose exemption would outlive the case it was granted to."""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.toml"
            path.write_text(
                '[known_wrong."cart/expected:1"]\nreason = "TICKET-7"\n', encoding="utf-8"
            )

            with self.assertRaises(baseline.BaselineError) as refused:
                baseline.Baseline.read(path)

        self.assertIn("cart/expected:1", str(refused.exception))

    def test_a_baseline_that_cannot_be_read_exits_two(self):
        with mock.patch.object(corpus, "verdict_fixtures", return_value=[]), \
             mock.patch.object(tiers, "judge_verdicts", return_value=[]), \
             mock.patch.object(
                 baseline.Baseline, "read",
                 side_effect=baseline.BaselineError("baseline.toml: 'x' needs a reason"),
             ):
            self.assertEqual(gate.main([]), 2)


class ExitCodeTests(unittest.TestCase):
    """What CI reads: a pass, a regression, an unreadable corpus, no credentials."""

    def _run(self, answers, argv=(), recorded=None):
        with mock.patch.object(tiers, "judge_verdicts", return_value=list(answers)), \
             mock.patch.object(corpus, "verdict_fixtures", return_value=[]), \
             mock.patch.object(
                 baseline.Baseline, "read", return_value=recorded or _baseline()
             ):
            return gate.main(list(argv))

    def test_a_corpus_that_answers_as_it_should_passes(self):
        self.assertEqual(self._run([_answer("a-cell", tiers.CORRECT)]), 0)

    def test_an_unaccounted_wrong_answer_fails(self):
        self.assertEqual(self._run([_answer("a-cell")]), 1)

    def test_a_baselined_wrong_answer_passes(self):
        self.assertEqual(
            self._run([_answer("a-cell")], recorded=_baseline(known_wrong=_excused("a-cell"))),
            0,
        )

    def test_a_baselined_item_the_corpus_lost_fails_the_build(self):
        """Deleting a fixture is how a failing case stops failing unfixed."""
        self.assertEqual(
            self._run(
                [_answer("still-here", tiers.CORRECT)],
                recorded=_baseline(known_wrong=_excused("deleted-away")),
            ),
            1,
        )

    def test_a_corpus_that_cannot_be_read_is_not_a_pass(self):
        with mock.patch.object(
            corpus, "verdict_fixtures", side_effect=corpus.CorpusError("broken.json: bad")
        ):
            self.assertEqual(gate.main([]), 2)

    def test_the_screen_corpus_without_credentials_says_so_rather_than_passing(self):
        """Exit 3 separates a misconfigured job from a real regression."""
        with mock.patch.object(corpus, "verdict_fixtures", return_value=[]), \
             mock.patch.object(tiers, "judge_verdicts", return_value=[]), \
             mock.patch.object(gate, "_model", return_value="a/model"), \
             mock.patch(
                 "ronin.llm.credentials.missing_credentials", return_value=["A_KEY"]
             ):
            self.assertEqual(gate.main(["--screens"]), 3)

    def _screens_raising(self, error):
        return [
            mock.patch.object(corpus, "verdict_fixtures", return_value=[]),
            mock.patch.object(tiers, "judge_verdicts", return_value=[]),
            mock.patch.object(gate, "_model", return_value="a/model"),
            mock.patch("ronin.llm.credentials.missing_credentials", return_value=[]),
            mock.patch.object(tiers, "audit_screens", side_effect=error),
        ]

    def test_a_provider_that_cannot_answer_is_not_read_as_a_regression(self):
        """An outage must not send somebody looking for a prompt they kept."""
        from ronin.errors import ProviderUnavailable

        with ExitStack() as patches:
            for patch in self._screens_raising(
                ProviderUnavailable("no endpoints found after 3 attempts")
            ):
                patches.enter_context(patch)

            self.assertEqual(gate.main(["--screens"]), 4)

    def test_a_failure_that_is_not_the_providers_is_not_called_an_outage(self):
        """A defect in this harness exiting 4 would read as weather."""
        with ExitStack() as patches:
            for patch in self._screens_raising(KeyError("expected:0")):
                patches.enter_context(patch)

            with self.assertRaises(KeyError):
                gate.main(["--screens"])

    def test_a_corpus_error_in_the_screen_sweep_stays_a_corpus_error(self):
        """A fixture nobody can read is ours to fix, not the provider's."""
        with mock.patch.object(corpus, "verdict_fixtures", return_value=[]), \
             mock.patch.object(tiers, "judge_verdicts", return_value=[]), \
             mock.patch.object(gate, "_model", return_value="a/model"), \
             mock.patch(
                 "ronin.llm.credentials.missing_credentials", return_value=[]
             ), \
             mock.patch.object(
                 tiers, "audit_screens",
                 side_effect=corpus.CorpusError("cart.json: no screenshot"),
             ):
            self.assertEqual(gate.main(["--screens"]), 2)

    def test_the_screen_corpus_is_not_asked_for_unless_it_is_asked_for(self):
        """The default run spends nothing. A gate that charges per push is not run."""
        with mock.patch.object(corpus, "verdict_fixtures", return_value=[]), \
             mock.patch.object(tiers, "judge_verdicts", return_value=[]), \
             mock.patch.object(corpus, "screen_fixtures") as screens:
            gate.main([])

        screens.assert_not_called()


class TheCommittedCorpusTests(unittest.TestCase):
    """The corpus in the repository, against the code in the repository."""

    def test_every_fixture_loads(self):
        self.assertTrue(corpus.verdict_fixtures())
        self.assertTrue(corpus.screen_fixtures())

    def test_the_deterministic_corpus_answers_as_the_corpus_says(self):
        """Run here as well as in CI, so a change to the judge fails at once.

        Nothing is mocked. A fixture that stops agreeing with the ranking is
        the signal this corpus exists to give, and waiting for a build to give
        it would be waiting for no reason: it needs no model and no device.
        """
        answers = tiers.judge_verdicts(corpus.verdict_fixtures())
        reading = baseline.Reading(answers=answers, baseline=baseline.Baseline.read())

        self.assertEqual(
            [found.item for found in reading.regressions], [], reading.render()
        )


if __name__ == "__main__":
    unittest.main()
