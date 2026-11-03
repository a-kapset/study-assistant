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
from typing import Literal

import structlog
from pydantic import BaseModel, ConfigDict, ValidationError

from evalharness.corpus_db import CorpusReader
from evalharness.datasets.qa_set import QaSet, QaTargets, load_qa_set
from evalharness.evaluation.answer_record import AnswerCaseRecord, AnswerRunRecord, write_answer_record
from evalharness.evaluation.retrieval import Corpus, RunRefused
from evalharness.observation import Observation
from evalharness.oracles.answer import AbstainReason, AnswerSummary, AskReply, score_case, summarise
from evalharness.proxy.ledger import LedgerLine
from evalharness.proxy.ledger import read_ledger as read_ledger_file
from evalharness.settings import HarnessSettings, StackSettings, load_stack_settings
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
) -> AnswerRunRecord:
    """Check the gold sections, search and ask every question one at a time, and return the run's record.

    ``read_ledger`` returns the proxy's ledger; it is read once, after the last question.
    """
    started_at = datetime.now(UTC)
    check_gold_sections(qa, corpus.section_ids(qa.course_id))
    snapshot = corpus.snapshot(qa.course_id)
    asked = []
    for case in qa.cases:
        body = {"course_id": qa.course_id}
        searched = await sut.request("POST", "/v1/search", json={**body, "query": case.question})
        answered = await sut.request("POST", "/v1/ask", json={**body, "question": case.question})
        asked.append((case, searched, answered))
    calls: defaultdict[str, list[LedgerLine]] = defaultdict(list)
    for line in read_ledger():
        if line.request_id is not None:
            calls[line.request_id].append(line)
    cases = tuple(
        _case_record(case.id, case.gold_section, searched, answered, calls[answered.request_id])
        for case, searched, answered in asked
    )
    # The ledger holds every call made on this machine; only the calls of this run's questions count.
    asked_ids = {answered.request_id for _, _, answered in asked}
    models = {
        line.model_returned for request_id in asked_ids for line in calls[request_id] if line.model_returned is not None
    }
    return AnswerRunRecord(
        run_id=uuid.uuid4().hex,
        started_at=started_at,
        harness_commit=harness_commit,
        harness_dirty=harness_dirty,
        qa_file=qa_file,
        qa_sha256=hashlib.sha256(qa_bytes).hexdigest(),
        course_id=qa.course_id,
        corpus_snapshot=snapshot,
        models_returned=tuple(sorted(models)),
        summary=summarise([case.gold_section for case in cases], [case.score for case in cases]),
        cases=cases,
    )


def _case_record(
    case_id: str, gold_section: str | None, searched: Observation, answered: Observation, calls: Sequence[LedgerLine]
) -> AnswerCaseRecord:
    """Record and score one question from its search response, its ask response and the ledger lines of the ask."""
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
    args = parser.parse_args(argv)
    # One log line per request would bury the summary; failed requests still show as warnings.
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))
    settings = HarnessSettings()
    stack = load_stack_settings(settings.stack_env_file)
    qa_bytes = args.qa_file.read_bytes()
    qa = load_qa_set(args.qa_file)
    commit, dirty = _harness_version()
    try:
        record = asyncio.run(_run(qa, qa_bytes, args.qa_file, settings, stack, commit, dirty))
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
            )


def _harness_version() -> tuple[str, bool]:
    """Return the git commit of the checkout and whether the harness folder has changes not yet committed."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--", "."], capture_output=True, text=True, check=True)
    return commit, bool(status.stdout.strip())


def target_verdict(summary: AnswerSummary, targets: QaTargets) -> dict[str, Literal["met", "below"]] | None:
    """Compare the run with each target; None when a case failed, so the run measured less than it claims.

    A share that could not be computed (no case to compute it over) does not meet its target.
    """
    if summary.failed:
        return None

    def at_least(value: float | None, minimum: float) -> Literal["met", "below"]:
        """Say whether a share reaches its minimum."""
        return "met" if value is not None and value >= minimum else "below"

    return {
        "answered": at_least(summary.answered, targets.answered_min),
        "gold_cited": at_least(summary.gold_cited, targets.gold_cited_min),
        "unsupported_answers": "met" if summary.unsupported_answers <= targets.unsupported_answers_max else "below",
    }


def _print_summary(record: AnswerRunRecord, targets: QaTargets, path: Path) -> None:
    """Print the run's identity and cost, where answerable questions went wrong, how unanswerable ones were
    declined, and the results against their targets."""
    s = record.summary
    dirty = " + uncommitted changes" if record.harness_dirty else ""
    costs = [case.cost_usd for case in record.cases]
    cost = "unknown" if any(c is None for c in costs) else f"${sum(c for c in costs if c is not None):.4f}"
    models = " ".join(record.models_returned) or "none called"
    print(f"run {record.run_id[:8]}  {record.course_id}  harness {record.harness_commit[:8]}{dirty}")
    print(f"model {models}  cost {cost}  record: {path}")
    print(f"invalid citations dropped by the app: {sum(c.invalid_citations or 0 for c in record.cases)}")

    print(f"\nanswerable ({s.answerable}): verdict by whether search found the gold section")
    print(f"  {'verdict':<22}{'found':>7}{'not found':>11}")
    answerable = [case for case in record.cases if case.gold_section is not None]
    for verdict in ("cited_gold", "cited_other", "declined_no_passages", "declined_by_model", "failed"):
        found = sum(c.score.verdict == verdict and c.score.gold_retrieved is True for c in answerable)
        lost = sum(c.score.verdict == verdict and c.score.gold_retrieved is not True for c in answerable)
        print(f"  {verdict:<22}{found:>7}{lost:>11}")

    print(f"\nunanswerable ({s.unanswerable}):")
    for verdict in ("declined_no_passages", "declined_by_model", "answered_unsupported", "failed"):
        count = sum(c.score.verdict == verdict for c in record.cases if c.gold_section is None)
        print(f"  {verdict:<22}{count:>7}")

    verdicts = target_verdict(s, targets)
    if verdicts is None:
        failed = sorted({c.ask_error or "unknown" for c in record.cases if c.score.verdict == "failed"})
        print(f"\ninvalid run: {s.failed} question(s) got no answer ({', '.join(failed)}); no target verdict")
        return
    answered = "n/a" if s.answered is None else f"{s.answered:.3f}"
    gold_cited = "n/a" if s.gold_cited is None else f"{s.gold_cited:.3f}"
    print(f"\nanswered     {answered:>6}  target >= {targets.answered_min}: {verdicts['answered']}")
    print(f"gold cited   {gold_cited:>6}  target >= {targets.gold_cited_min}: {verdicts['gold_cited']}")
    print(
        f"unsupported  {s.unsupported_answers:>6}  target <= {targets.unsupported_answers_max}: "
        f"{verdicts['unsupported_answers']}"
    )
    overall = "met" if all(v == "met" for v in verdicts.values()) else "below"
    print(f"all targets: {overall}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
