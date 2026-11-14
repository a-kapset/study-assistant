"""The record of one paraphrase relation run: the settings it ran under, the counts it was judged by, and what both
searches of every pair returned.

Like a retrieval record it holds no question text: the case id finds the question and its rewording in the
committed pair file, whose hash the record carries.
"""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, JsonValue

from evalharness.evaluation.record import PassName, PassResult
from evalharness.relations.paraphrase import PairOutcome, RelationSummary


class _Strict(BaseModel):
    """Base for the record: an unknown key is an error, and a record never changes after it is made."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PairPass(_Strict):
    """Both searches of one pair in one pass, and how the pair came out."""

    question: PassResult
    rewording: PassResult
    outcome: PairOutcome


class PairRecord(_Strict):
    """One pair: its question's id, the section that answers it, and the result of each pass."""

    case_id: str
    gold_section: str
    passes: dict[PassName, PairPass]


class ParaphraseRunRecord(_Strict):
    """Everything one relation run measured, and the values two runs must share, or differ in on purpose, to be
    compared."""

    run_id: str
    # In UTC.
    started_at: datetime
    # The harness's git commit, and whether its files had changes not yet committed (the harness folder only).
    harness_commit: str
    harness_dirty: bool
    pair_file: str
    # Names the texts searched: the pair file and its Q&A set file.
    set_sha256: str
    course_id: str
    # Taken once, before the first search.
    corpus_snapshot: str
    # The settings the app reported at GET /v1/config before the first search, secrets masked. Their "git_commit"
    # names the app's code, with "-dirty" when app/ had uncommitted changes at build time.
    app_settings: dict[str, JsonValue]
    # The values the app reported in the primary pass; None when no search of that pass succeeded.
    embedding_space: str | None
    k: int | None
    min_score: float | None
    # The search path the app reported with them: its strategy, and whether a reranker ordered the hits. None when
    # no search of the primary pass succeeded, and in records written before runs read them.
    strategy: str | None = None
    reranked: bool | None = None
    # Per pass: "all", "key term removed" and "no key term in question".
    summaries: dict[PassName, dict[str, RelationSummary]]
    pairs: tuple[PairRecord, ...]


def write_record(record: ParaphraseRunRecord, runs_dir: Path) -> Path:
    """Write the record as a new JSON file in ``runs_dir`` and return its path; an existing file is never replaced."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{record.started_at:%Y%m%dT%H%M%SZ}-{record.course_id}-paraphrase-{record.run_id[:8]}.json"
    with path.open("x", encoding="utf-8") as file:
        file.write(record.model_dump_json(indent=2))
    return path
