#!/usr/bin/env python3
"""The eval gate: what Ronin's own judging answers on a labelled corpus.

Run it as a module from the repository root:

    python -m tools.eval_gate              the deterministic corpus
    python -m tools.eval_gate --screens    and the auditor, which needs a model

This is a gate on Ronin, not a command a user runs against an app. It answers
one question: did a change to a prompt, a rule or the verdict ranking make
Ronin answer a labelled case differently? It does not answer whether Ronin is
good. Nothing in a corpus this size could.

Exit codes:
    0  every item answered as the corpus says, or was already baselined
    1  an item answered wrongly and the baseline does not account for it
    2  the corpus or the baseline could not be read
    3  the screen corpus was asked for and no model credentials are present
    4  the screen corpus was asked and the provider could not answer
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):  # pragma: no cover - module form is what CI runs
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.eval import baseline, corpus, tiers  # noqa: E402 - after the path fix


def _prompt_store(record: Optional[Path]):
    """Where the question behind each look is kept.

    With a record to write beside, the bodies are kept: an image reaches the
    store as a hash rather than as bytes, so a body is the words that were
    asked and little else. Without one, only the hash is taken, and nothing
    is written.
    """
    from ronin.llm.prompts import PromptStore

    if record is None:
        return PromptStore(Path("prompts"), keep_bodies=False)
    return PromptStore(record.parent / "prompts", keep_bodies=True)


def _auditor_factory(model: str, record: Optional[Path]):
    """Builds one auditor per look, over one client, and keeps each call.

    A fresh auditor per look, because the cache lives on the auditor and a
    cached answer is not a second sample. One client, because the route and
    its credentials are the same question every time.

    The client records the prompt it sent, with no body kept. The hash is
    what a later sweep is compared against, and the body would be a copy of
    a screen already in the corpus.
    """
    from ronin.llm.llm_client import LLMClient
    from ronin.testing.judge.qa_auditor import QAAuditor

    calls: list = []
    client = LLMClient(
        model=model, usage_callback=calls.append, prompt_store=_prompt_store(record)
    )

    def build(fixture: corpus.ScreenFixture) -> QAAuditor:
        return QAAuditor(client, fixture.outcomes, fixture.security_rules)

    def last():
        """The call the auditor just made, or None where none was reported."""
        return calls.pop() if calls else None

    return build, last


def _at_least_one(given: str) -> int:
    """A replicate count a sweep can actually use.

    Zero asks nothing and reports every rule unanswered, which reads as the
    auditor having gone silent.
    """
    count = int(given)
    if count < 1:
        raise argparse.ArgumentTypeError(f"replicates must be at least 1, not {count}")
    return count


def _model(named: str | None) -> str:
    from ronin.config import load_config

    return named or load_config().resolve("model")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--screens",
        action="store_true",
        help="Also put the screen corpus to the auditor. Spends tokens.",
    )
    parser.add_argument(
        "--replicates",
        type=_at_least_one,
        default=tiers.REPLICATES,
        help="How many times each screen rule is asked.",
    )
    parser.add_argument("--model", default=None, help="Override the auditor's model.")
    parser.add_argument(
        "--record",
        type=Path,
        default=None,
        help=(
            "Write what each call asked and answered to this file, with the "
            "prompts beside it. A sweep without it reports states alone, and "
            "a state cannot say which call produced it."
        ),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    audits: list = []

    ran = {corpus.VERDICT_TIER} | ({corpus.SCREEN_TIER} if args.screens else set())
    try:
        recorded = baseline.Baseline.read()
        answers = tiers.judge_verdicts(corpus.verdict_fixtures())
        if args.screens:
            answers += _screen_answers(args, audits)
    except corpus.CorpusError as exc:
        print(f"The corpus could not be read: {exc}")
        return 2
    except (OSError, baseline.BaselineError) as exc:
        print(f"The baseline could not be read: {exc}")
        return 2
    except _NoCredentials as exc:
        print(str(exc))
        return 3
    except _ProviderRefused as exc:
        # The calls before the failure are still evidence of what answered.
        _keep(args, audits)
        print(str(exc))
        return 4

    _keep(args, audits)
    reading = baseline.Reading(
        answers=answers, baseline=recorded, tiers=frozenset(ran)
    )
    print(reading.render())
    if reading.holds:
        print("\nEval gate passed.")
        return 0
    if reading.regressions:
        print(
            f"\nEval gate failed. {len(reading.regressions)} item(s) answer wrongly "
            "and tools/eval-baseline.toml does not account for them."
        )
    if reading.absent:
        print(
            f"Eval gate failed. {len(reading.absent)} baselined item(s) are no longer "
            "in the corpus. A fixture that was deleted takes its failure with it."
        )
    if reading.lapsed:
        print(
            f"Eval gate failed. {len(reading.lapsed)} exemption(s) cover an older "
            "version of the fixture they name. Read the case again before renewing one."
        )
    return 1


def _keep(args, audits: list) -> None:
    """Writes the sweep where one was asked for and there is one to write."""
    if args.record and audits:
        _written(args.record, audits, args)


def _written(path: Path, audits: list, args) -> None:
    """The sweep, as one file a later sweep is compared against."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "requested_model": _model(args.model),
                "replicates": args.replicates,
                "audits": [audit.as_dict() for audit in audits],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(audits)} audit(s) to {path}.")


class _NoCredentials(Exception):
    """The screen corpus was asked for and nothing can answer it."""


class _ProviderRefused(Exception):
    """The provider could not answer, which is not a regression in Ronin.

    Kept apart from exit 1 on purpose. A build that reads an outage as a
    worse auditor would have somebody looking for a prompt they did not
    change. It is kept apart from exit 0 for the same reason in reverse: the
    screen corpus was asked for and nothing answered it.
    """


def _screen_answers(args, audits: list) -> list[tiers.Answer]:
    from ronin.llm.credentials import missing_credentials

    model = _model(args.model)
    missing = missing_credentials(model)
    if missing:
        raise _NoCredentials(
            f"The screen corpus needs '{model}', and {', '.join(missing)} is not set. "
            "The deterministic corpus ran. Nothing was asked of a model."
        )
    try:
        build, last = _auditor_factory(model, args.record)
        return tiers.audit_screens(
            corpus.screen_fixtures(),
            build,
            args.replicates,
            audits=audits,
            call_for=last,
        )
    except corpus.CorpusError:
        raise
    except Exception as exc:
        _only_a_provider_failure(exc)
        from ronin.llm.failures import reportable

        raise _ProviderRefused(
            f"The screen corpus could not be asked of '{model}': {reportable(exc)}\n"
            "The deterministic corpus ran and is reported above."
        ) from exc


def _only_a_provider_failure(exc: Exception) -> None:
    """Lets anything that is not the provider's doing through as itself.

    Read as an outage, a defect in this harness exits 4 and reads as weather.
    The cause boundary is the one Ronin already classifies provider failures
    with, so the gate does not invent a second opinion about them.
    """
    from ronin.errors import Cause, cause_of

    if cause_of(exc) is not Cause.PROVIDER:
        raise exc


if __name__ == "__main__":
    raise SystemExit(main())
