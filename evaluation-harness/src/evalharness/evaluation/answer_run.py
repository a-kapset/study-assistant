"""An answer run: ask every question of a Q&A set, score what the app answers, and record the model calls each
answer took.

For every question a search with the app's own settings runs first: it returns the passages the answer step saw,
so a case that went wrong can be placed at the search or after it. A run whose gold sections are not sections of
the stored course is refused before the first question.

    poetry run python -m evalharness.evaluation.answer_run datasets/<course>/qa_v0.toml

Run it from evaluation-harness/, on the stack started with the proxy (evaluation-harness/compose.proxy.yaml). The
run is written to runs/answers/ as a new JSON file and its summary is printed. Exit code 0 when the run completed,
whatever its numbers; 2 when it was refused.
"""

import argparse
import asyncio
import hashlib
import logging
import subprocess
import sys
import uuid
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import structlog
from pydantic import BaseModel, ConfigDict, ValidationError

from evalharness.corpus_db import CorpusReader
from evalharness.datasets.qa_set import QaSet, QaTargets, load_qa_set
from evalharness.evaluation.answer_record import AnswerCaseRecord, AnswerRunRecord, write_answer_record
from evalharness.evaluation.retrieval import Corpus, RunRefused, read_app_settings
from evalharness.observation import Observation
from evalharness.oracles.answer import (
    AbstainReason,
    AnswerSummary,
    AnswerVerdict,
    AskReply,
    flipped_cases,
    score_case,
    summarise,
)
from evalharness.proxy.keys import repetition_suffix
from evalharness.proxy.ledger import LedgerLine
from evalharness.proxy.ledger import read_ledger as read_ledger_file
from evalharness.settings import HarnessSettings, StackSettings, load_stack_settings
from evalharness.stats.intervals import TargetVerdict, share_interval, share_verdict
from evalharness.transport import SutTransport

# Answer records only, apart from the retrieval records, whose folder is counted for held-out uses.
ANSWERS_DIR = Path("runs/answers")


class _Section(BaseModel):
    """The one field of a hit or a citation a run needs: the section of its passage."""

    model_config = ConfigDict(extra="ignore")
    section_id: str | None


class _Searched(BaseModel):
    """The hits of a search response, in the order the app returned them."""

    model_config = ConfigDict(extra="ignore")
    hits: list[_Section]


class _Answered(BaseModel):
    """The fields of an ask response a run scores and records."""

    model_config = ConfigDict(extra="ignore")
    abstain_reason: AbstainReason | None
    citations: list[_Section]
    invalid_citations: int


class _ErrorBody(BaseModel):
    """The code of the app's error envelope."""

    model_config = ConfigDict(extra="ignore")
    code: str


class _ErrorEnvelope(BaseModel):
    """The app's error envelope."""

    model_config = ConfigDict(extra="ignore")
    error: _ErrorBody


def check_gold_sections(qa: QaSet, section_ids: set[str]) -> None:
    """Refuse the run when a gold section of the set is not a section of the stored course.

    No answer could ever cite such a section, so its question would score "cited_other" in every run, and a changed
    section numbering of the course would look like worse answers.
    """
    gold = {case.gold_section for case in qa.cases if case.gold_section is not None}
    missing = sorted(gold - section_ids)
    if missing:
        raise RunRefused("gold_section_missing", f"gold sections {missing} are not sections of the stored course")


async def run(
    qa: QaSet,
    sut: SutTransport,
    corpus: Corpus,
    read_ledger: Callable[[], Sequence[LedgerLine]],
    *,
    qa_file: str,
    qa_bytes: bytes,
    harness_commit: str,
    harness_dirty: bool,
    repetitions: int = 1,
) -> AnswerRunRecord:
    """Check the gold sections, search every question once, ask every question ``repetitions`` times one at a time,
    and return the run's record.

    Search is asked once: it does not sample, so asking again would only repeat its hits. Each ask carries its
    repetition in the request id, so the proxy calls the model anew instead of serving the first answer again.
    ``read_ledger`` returns the proxy's ledger; it is read once, after the last question.
    """
    if repetitions < 1:
        raise ValueError(f"a run asks every question at least once, got {repetitions}")
    started_at = datetime.now(UTC)
    check_gold_sections(qa, corpus.section_ids(qa.course_id))
    snapshot = corpus.snapshot(qa.course_id)
    app_settings = await read_app_settings(sut)
    searched = {}
    for case in qa.cases:
        searched[case.id] = await sut.request(
            "POST", "/v1/search", json={"course_id": qa.course_id, "query": case.question}
        )
    asked = []
    for repetition in range(1, repetitions + 1):
        for case in qa.cases:
            answered = await sut.request(
                "POST",
                "/v1/ask",
                json={"course_id": qa.course_id, "question": case.question},
                request_id_suffix=repetition_suffix(repetition),
            )
            asked.append((repetition, case, answered))
    calls: defaultdict[str, list[LedgerLine]] = defaultdict(list)
    for line in read_ledger():
        if line.request_id is not None:
            calls[line.request_id].append(line)
    cases = tuple(
        _case_record(case.id, repetition, case.gold_section, searched[case.id], answered, calls[answered.request_id])
        for repetition, case, answered in asked
    )
    # The ledger holds every call made on this machine; only the calls of this run's questions count.
    asked_ids = {answered.request_id for _, _, answered in asked}
    models = {
        line.model_returned for request_id in asked_ids for line in calls[request_id] if line.model_returned is not None
    }
    summaries = tuple(
        summarise(
            [case.gold_section for case in cases if case.repetition == repetition],
            [case.score for case in cases if case.repetition == repetition],
        )
        for repetition in range(1, repetitions + 1)
    )
    return AnswerRunRecord(
        run_id=uuid.uuid4().hex,
        started_at=started_at,
        harness_commit=harness_commit,
        harness_dirty=harness_dirty,
        qa_file=qa_file,
        qa_sha256=hashlib.sha256(qa_bytes).hexdigest(),
        course_id=qa.course_id,
        corpus_snapshot=snapshot,
        app_settings=app_settings,
        models_returned=tuple(sorted(models)),
        repetitions=repetitions,
        summaries=summaries,
        cases=cases,
    )


def _case_record(
    case_id: str,
    repetition: int,
    gold_section: str | None,
    searched: Observation,
    answered: Observation,
    calls: Sequence[LedgerLine],
) -> AnswerCaseRecord:
    """Record and score one question in one repetition from its search response, its ask response and the ledger
    lines of the ask."""
    search_sections = (
        tuple(hit.section_id for hit in _Searched.model_validate(searched.body_json).hits)
        if _error(searched) is None
        else None
    )
    ask_error = _error(answered)
    reply = _Answered.model_validate(answered.body_json) if ask_error is None else None
    ask_reply = (
        AskReply(abstain_reason=reply.abstain_reason, cited_sections=tuple(c.section_id for c in reply.citations))
        if reply is not None
        else None
    )
    # A miss got no reply and was not paid for; only answered calls have a cost.
    costs = [line.cost_usd for line in calls if line.source != "miss"]
    return AnswerCaseRecord(
        case_id=case_id,
        repetition=repetition,
        gold_section=gold_section,
        search_request_id=searched.request_id,
        ask_request_id=answered.request_id,
        ask_error=ask_error,
        abstain_reason=reply.abstain_reason if reply is not None else None,
        cited_sections=ask_reply.cited_sections if ask_reply is not None else (),
        invalid_citations=reply.invalid_citations if reply is not None else None,
        search_sections=search_sections,
        model_calls=tuple(line.source for line in calls),
        cost_usd=None if any(cost is None for cost in costs) else sum(cost for cost in costs if cost is not None),
        score=score_case(gold_section, ask_reply, search_sections),
    )


def _error(observation: Observation) -> str | None:
    """Say why a response holds no result: the transport error, the app's error code or "http_<status>"; None
    when the request succeeded. A failed question never stops the run."""
    if observation.transport_error is not None:
        return observation.transport_error
    if observation.status_code == 200:
        return None
    try:
        return _ErrorEnvelope.model_validate(observation.body_json).error.code
    except ValidationError:
        return f"http_{observation.status_code}"


def main(argv: Sequence[str]) -> int:
    """Run the command and return its exit code: 0 when the run completed, whatever its numbers; 2 when refused."""
    parser = argparse.ArgumentParser(prog="python -m evalharness.evaluation.answer_run")
    parser.add_argument("qa_file", type=Path, help="the Q&A set file")
    parser.add_argument("--reps", type=int, default=1, help="how many times to ask every question (default 1)")
    args = parser.parse_args(argv)
    if args.reps < 1:
        parser.error("--reps must be at least 1")
    # One log line per request would bury the summary; failed requests still show as warnings.
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))
    settings = HarnessSettings()
    stack = load_stack_settings(settings.stack_env_file)
    qa_bytes = args.qa_file.read_bytes()
    qa = load_qa_set(args.qa_file)
    commit, dirty = _harness_version()
    try:
        record = asyncio.run(_run(qa, qa_bytes, args.qa_file, settings, stack, commit, dirty, args.reps))
    except RunRefused as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    path = write_answer_record(record, ANSWERS_DIR)
    _print_summary(record, qa.targets, path)
    return 0


async def _run(
    qa: QaSet,
    qa_bytes: bytes,
    qa_file: Path,
    settings: HarnessSettings,
    stack: StackSettings,
    commit: str,
    dirty: bool,
    repetitions: int,
) -> AnswerRunRecord:
    """Make the run against the stack: requests over HTTP, the corpus through the read-only role, the proxy's
    ledger from this machine."""
    with CorpusReader(settings, stack) as corpus:
        async with SutTransport(settings) as sut:
            return await run(
                qa,
                sut,
                corpus,
                lambda: read_ledger_file(settings.proxy_ledger_file),
                qa_file=qa_file.as_posix(),
                qa_bytes=qa_bytes,
                harness_commit=commit,
                harness_dirty=dirty,
                repetitions=repetitions,
            )


def _harness_version() -> tuple[str, bool]:
    """Return the git commit of the checkout and whether the harness folder has changes not yet committed."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--", "."], capture_output=True, text=True, check=True)
    return commit, bool(status.stdout.strip())


def target_verdict(summary: AnswerSummary, targets: QaTargets) -> dict[str, TargetVerdict] | None:
    """Compare the run with each target; None when a case failed, so the run measured less than it claims.

    A share is judged by its 95 % interval: "met" only when even the lower end reaches the minimum. A share that
    could not be computed (no case to compute it over) does not meet its target. Unsupported answers are judged by
    their count, not by an interval: each one is a blocker.
    """
    if summary.failed:
        return None
    answered = _answered(summary)
    return {
        "answered": (
            share_verdict(answered, summary.answerable, targets.answered_min) if summary.answerable else "below"
        ),
        "gold_cited": (
            share_verdict(summary.verdicts.get("cited_gold", 0), answered, targets.gold_cited_min)
            if answered
            else "below"
        ),
        "unsupported_answers": "met" if summary.unsupported_answers <= targets.unsupported_answers_max else "below",
    }


def _answered(summary: AnswerSummary) -> int:
    """Count the answerable questions the app answered, with or without the gold section among the citations."""
    return summary.verdicts.get("cited_gold", 0) + summary.verdicts.get("cited_other", 0)


def _range(successes: int, n: int) -> str:
    """Write the Wilson interval of ``successes`` out of ``n`` as "low-high", or "n/a" when there is no question."""
    if n == 0:
        return "n/a"
    interval = share_interval(successes, n)
    return f"{interval.low:.3f}-{interval.high:.3f}"


def _print_summary(record: AnswerRunRecord, targets: QaTargets, path: Path) -> None:
    """Print the run's identity and cost, where answerable questions went wrong and how unanswerable ones were
    declined (first repetition), the results against their targets (each repetition), and the flip rate."""
    dirty = " + uncommitted changes" if record.harness_dirty else ""
    costs = [case.cost_usd for case in record.cases]
    cost = "unknown" if any(c is None for c in costs) else f"${sum(c for c in costs if c is not None):.4f}"
    models = " ".join(record.models_returned) or "none called"
    print(f"run {record.run_id[:8]}  {record.course_id}  harness {record.harness_commit[:8]}{dirty}")
    print(f"model {models}  cost {cost}  repetitions {record.repetitions}  record: {path}")
    print(f"invalid citations dropped by the app: {sum(c.invalid_citations or 0 for c in record.cases)}")

    first = [case for case in record.cases if case.repetition == 1]
    label = ", repetition 1" if record.repetitions > 1 else ""
    answerable = [case for case in first if case.gold_section is not None]
    print(f"\nanswerable ({len(answerable)}{label}): verdict by whether search found the gold section")
    print(f"  {'verdict':<22}{'found':>7}{'not found':>11}")
    for verdict in ("cited_gold", "cited_other", "declined_no_passages", "declined_by_model", "failed"):
        found = sum(c.score.verdict == verdict and c.score.gold_retrieved is True for c in answerable)
        lost = sum(c.score.verdict == verdict and c.score.gold_retrieved is not True for c in answerable)
        print(f"  {verdict:<22}{found:>7}{lost:>11}")

    unanswerable = [case for case in first if case.gold_section is None]
    print(f"\nunanswerable ({len(unanswerable)}{label}):")
    for verdict in ("declined_no_passages", "declined_by_model", "answered_unsupported", "failed"):
        print(f"  {verdict:<22}{sum(c.score.verdict == verdict for c in unanswerable):>7}")

    for repetition, summary in enumerate(record.summaries, start=1):
        errors = sorted(
            {
                c.ask_error or "unknown"
                for c in record.cases
                if c.repetition == repetition and c.score.verdict == "failed"
            }
        )
        _print_targets(summary, targets, f"repetition {repetition}" if record.repetitions > 1 else "", errors)
    if record.repetitions > 1:
        _print_flips(record)


def _print_targets(s: AnswerSummary, targets: QaTargets, title: str, errors: list[str]) -> None:
    """Print one repetition's results with their intervals against the targets, or why it has no verdict."""
    verdicts = target_verdict(s, targets)
    heading = f"{title}: " if title else ""
    if verdicts is None:
        print(f"\n{heading}invalid: {s.failed} question(s) got no answer ({', '.join(errors)}); no target verdict")
        return
    answered = "n/a" if s.answered is None else f"{s.answered:.3f}"
    gold_cited = "n/a" if s.gold_cited is None else f"{s.gold_cited:.3f}"
    answered_ci = _range(_answered(s), s.answerable)
    gold_cited_ci = _range(s.verdicts.get("cited_gold", 0), _answered(s))
    unsupported_ci = _range(s.unsupported_answers, s.unanswerable)
    print(f"\n{heading:<13}{'value':>6}{'95% CI':>14}  target")
    print(f"answered     {answered:>6}{answered_ci:>14}  >= {targets.answered_min}: {verdicts['answered']}")
    print(f"gold cited   {gold_cited:>6}{gold_cited_ci:>14}  >= {targets.gold_cited_min}: {verdicts['gold_cited']}")
    print(
        f"unsupported  {s.unsupported_answers:>6}{unsupported_ci:>14}  <= {targets.unsupported_answers_max}: "
        f"{verdicts['unsupported_answers']}  (count; the interval is of their share of {s.unanswerable})"
    )
    values = verdicts.values()
    overall = "below" if "below" in values else ("not_confirmed" if "not_confirmed" in values else "met")
    print(f"all targets: {overall}")


def _print_flips(record: AnswerRunRecord) -> None:
    """Print the share of questions whose verdict changed between the repetitions, and each such question with its
    verdicts in order: these are the questions where one answer says little about the next."""
    verdicts: defaultdict[str, list[AnswerVerdict]] = defaultdict(list)
    for case in record.cases:
        verdicts[case.case_id].append(case.score.verdict)
    counted, flipped = flipped_cases(verdicts)
    if counted == 0:
        print("\nflip rate: n/a (every question failed in some repetition)")
        return
    print(
        f"\nflip rate {len(flipped)}/{counted} = {len(flipped) / counted:.3f}, 95% CI {_range(len(flipped), counted)}"
        f"  (questions whose verdict was not the same in all {record.repetitions} repetitions)"
    )
    for case_id in flipped:
        print(f"  {case_id}: {' > '.join(verdicts[case_id])}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
