"""The record of one answer run over a Q&A set: what every question got from the app in each repetition, how it
scored, and the values needed to decide later whether two runs may be compared.

It holds no question or answer text: answers quote the course, and a case id is enough to find its question in the
set's file. Answer records go to their own folder, apart from the retrieval records the retrieval runner counts.
"""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, JsonValue

from evalharness.failures import FailureLabel
from evalharness.oracles.answer import AbstainReason, AnswerCaseScore, AnswerSummary
from evalharness.proxy.ledger import Source


class _Strict(BaseModel):
    """Base for the record: an unknown key is an error, and a record never changes after it is made."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class AnswerCaseRecord(_Strict):
    """One question in one repetition: what search found for it, what the app answered, which model calls that
    took, and the score."""

    case_id: str
    # Counted from 1; a run without repetitions has only the first.
    repetition: int
    # None for a question the course does not answer.
    gold_section: str | None
    # The same ids are in the app's log lines and, for the ask, in the proxy's ledger.
    search_request_id: str
    ask_request_id: str
    # Why the ask gave no answer: the code of the app's error response, "http_<status>" or the transport error.
    ask_error: str | None
    # Why the ask failed, from its response and the ledger: decides whether the case counts for quality. None when it
    # did not fail.
    failure_label: FailureLabel | None
    abstain_reason: AbstainReason | None
    cited_sections: tuple[str | None, ...]
    # Citations the model gave to passages it was not shown; None when the ask failed.
    invalid_citations: int | None
    # The sections of the passages a search for the question found, in order; None when that search failed.
    search_sections: tuple[str | None, ...] | None
    # Who answered each model call of the ask, from the proxy's ledger; empty when the model was not called.
    model_calls: tuple[Source, ...]
    # What the model calls cost at the listed prices, whoever answered; None when a call has no price.
    cost_usd: float | None
    score: AnswerCaseScore


class AnswerRunRecord(_Strict):
    """Everything one answer run measured, and the values two runs must share, or differ in on purpose."""

    run_id: str
    # In UTC.
    started_at: datetime
    # The harness's git commit, and whether its files had changes not yet committed (the harness folder only).
    harness_commit: str
    harness_dirty: bool
    qa_file: str
    # The SHA-256 of the set's file: the same value means the same questions, gold sections and targets.
    qa_sha256: str
    course_id: str
    # Taken once, before the first question.
    corpus_snapshot: str
    # The settings the app reported at GET /v1/config before the first question, secrets masked. Their "git_commit"
    # names the app's code, with "-dirty" when app/ had uncommitted changes at build time.
    app_settings: dict[str, JsonValue]
    # The model versions the provider reported answering with, from the ledger; empty when no model was called.
    models_returned: tuple[str, ...]
    # How many times every question was asked; runs asked a different number of times are not compared directly.
    repetitions: int
    # One per repetition, in order.
    summaries: tuple[AnswerSummary, ...]
    cases: tuple[AnswerCaseRecord, ...]


def write_answer_record(record: AnswerRunRecord, answers_dir: Path) -> Path:
    """Write the record as a new JSON file in ``answers_dir`` and return its path; never replace an existing file."""
    answers_dir.mkdir(parents=True, exist_ok=True)
    path = answers_dir / f"{record.started_at:%Y%m%dT%H%M%SZ}-{record.course_id}-{record.run_id[:8]}.json"
    with path.open("x", encoding="utf-8") as file:
        file.write(record.model_dump_json(indent=2))
    return path
