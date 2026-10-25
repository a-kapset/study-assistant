"""The record of one evaluation run: what was measured, under which settings, and what every question scored.

A record is the only lasting result of a run. It holds every value needed to decide later whether two runs may be
compared (gold set, corpus, settings, harness version) and the scores of every question, so a summary can be
recomputed or cut another way. It holds no question or passage text: that is course material, and the case id is
enough to find a question in the local question bank.
"""

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from evalharness.oracles.retrieval import CaseScore, Summary

# "primary": the search as a user sends it, with the app's own settings. "diagnostic": the same number of hits with
# no lowest score, which shows how many misses the lowest score causes.
type PassName = Literal["primary", "diagnostic"]


class _Strict(BaseModel):
    """Base for the record: an unknown key is an error, and a record never changes after it is made."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class HitRecord(_Strict):
    """One returned passage, named by its id, with its section and its score."""

    chunk_id: str
    section_id: str | None
    score: float


class PassResult(_Strict):
    """What one search of one question returned, and how it scored."""

    # The same id is in the app's log lines for this search.
    request_id: str
    # Why the search gave no result: the code of the app's error response, "http_<status>" for an error response in
    # another format, or the transport error; None when the search ran.
    error: str | None
    # In the order the app returned them; empty when the search found nothing or failed.
    hits: tuple[HitRecord, ...]
    score: CaseScore


class CaseRecord(_Strict):
    """One question of the gold set: its tags, its answer section and the result of each pass."""

    case_id: str
    exam_set: str
    chapter: str
    knowledge_level: str | None
    gold_section: str
    passes: dict[PassName, PassResult]


class RunRecord(_Strict):
    """Everything one run measured, and every value two runs must share, or differ in on purpose, to be compared."""

    run_id: str
    # In UTC.
    started_at: datetime
    # The harness's git commit, and whether its files had changes not yet committed (the harness folder only).
    harness_commit: str
    harness_dirty: bool
    declaration_file: str
    # Names the questions, their answer sections and the scoring rules; it also covers the question bank's hash.
    gold_sha256: str
    course_id: str
    sets: tuple[str, ...]
    # Which use of the course's held-out sets this run is, counting the uses before records were kept;
    # None when the run includes no held-out set.
    held_out_use: int | None
    # Taken once, before the first search.
    corpus_snapshot: str
    # The values the app reported in the primary pass; None when no search of that pass succeeded.
    embedding_space: str | None
    k: int | None
    min_score: float | None
    # Per pass: "all", then one entry per exam set, chapter and knowledge level, e.g. "set A", "chapter 1", "level K2".
    summaries: dict[PassName, dict[str, Summary]]
    cases: tuple[CaseRecord, ...]


class _HeldOutMark(BaseModel):
    """The two fields of any record, also of one written by an older harness, that say whether it used held-out sets."""

    model_config = ConfigDict(extra="ignore")

    course_id: str
    held_out_use: int | None


def write_record(record: RunRecord, runs_dir: Path) -> Path:
    """Write the record as a new JSON file in ``runs_dir`` and return its path; an existing file is never replaced."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{record.started_at:%Y%m%dT%H%M%SZ}-{record.course_id}-{record.run_id[:8]}.json"
    with path.open("x", encoding="utf-8") as file:
        file.write(record.model_dump_json(indent=2))
    return path


def recorded_held_out_uses(runs_dir: Path, course_id: str) -> int:
    """Count the recorded runs of a course that included a held-out set; a file that is not a record is an error."""
    if not runs_dir.is_dir():
        return 0
    marks = (_HeldOutMark.model_validate_json(path.read_bytes()) for path in runs_dir.glob("*.json"))
    return sum(mark.course_id == course_id and mark.held_out_use is not None for mark in marks)
