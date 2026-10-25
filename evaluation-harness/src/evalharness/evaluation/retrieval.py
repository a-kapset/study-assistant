"""A retrieval run: send every chosen question of a gold set to the course search and score what comes back.

Every question is searched twice. The primary pass sends it as a user does, so the app's own number of hits and
lowest score apply. The diagnostic pass asks for the same number of hits with no lowest score, so the misses the
lowest score causes can be told apart from the misses of the search itself. A run that could not measure what it
claims is refused before the first search.
"""

import uuid
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from evalharness.datasets.retrieval_gold import GoldCase, GoldDeclaration, GoldSet
from evalharness.evaluation.record import CaseRecord, HitRecord, PassResult, RunRecord
from evalharness.observation import Observation
from evalharness.oracles.retrieval import CaseScore, Summary, score_case, summarise
from evalharness.transport import SutTransport

# Why a run was refused; a caller can act on the code without parsing the message.
type RunRefusedCode = Literal["course_mismatch", "unknown_set", "held_out_not_allowed", "gold_section_missing"]


class RunRefused(ValueError):
    """The run could not measure what it claims, so it is refused before the first search; says why with a code."""

    def __init__(self, code: RunRefusedCode, detail: str) -> None:
        """Keep the code apart from the message, so tests and callers need not parse text."""
        super().__init__(f"{code}: {detail}")
        self.code = code


class Corpus(Protocol):
    """What a run reads from the app's stored corpus: the database reader in a real run, a fake in tests."""

    def section_ids(self, course_id: str) -> set[str]:
        """Return the ids of the course's headed sections."""
        ...

    def snapshot(self, course_id: str) -> str:
        """Name the course's current corpus."""
        ...


class _Hit(BaseModel):
    """The fields of a search hit a run records; the passage text is not read."""

    model_config = ConfigDict(extra="ignore")

    chunk_id: str
    section_id: str | None
    score: float


class _SearchResult(BaseModel):
    """The fields of a successful search response a run reads."""

    model_config = ConfigDict(extra="ignore")

    embedding_space: str
    k: int
    min_score: float
    hits: list[_Hit]


class _ErrorBody(BaseModel):
    """The code of the app's error response."""

    model_config = ConfigDict(extra="ignore")

    code: str


class _ErrorEnvelope(BaseModel):
    """The app's error response, read only for its code."""

    model_config = ConfigDict(extra="ignore")

    error: _ErrorBody


# One search as a run sees it: the request id, why the search gave no result (None when it ran), and the result.
type _Searched = tuple[str, str | None, _SearchResult | None]


def bank_folder(declaration: GoldDeclaration, local_course_id: str, local_sources: Path, env_file: Path) -> Path:
    """Return the folder of the course's local materials; refused when the stack mounts another course's materials.

    A relative folder starts from the folder of the stack's .env file, as Docker Compose reads it.
    """
    if local_course_id != declaration.course_id:
        raise RunRefused(
            "course_mismatch",
            f"the gold set is for {declaration.course_id!r}, the stack's local course is {local_course_id!r}",
        )
    return env_file.parent / local_sources


def select_cases(gold: GoldSet, sets: Sequence[str], include_held_out: bool) -> tuple[GoldCase, ...]:
    """Choose the questions of ``sets``; an unknown set, or a held-out set without ``include_held_out``, is refused."""
    known = {case.exam_set for case in gold.cases}
    unknown = sorted(set(sets) - known)
    if unknown:
        raise RunRefused("unknown_set", f"sets {unknown} are not in the gold set, which has {sorted(known)}")
    held_out = sorted(set(sets) & set(gold.declaration.held_out_sets))
    if held_out and not include_held_out:
        raise RunRefused("held_out_not_allowed", f"sets {held_out} are held out; ask for them explicitly")
    return tuple(case for case in gold.cases if case.exam_set in sets)


def check_sections(gold: GoldSet, section_ids: set[str]) -> None:
    """Refuse the run when an answer section of the gold set is not a section of the stored course.

    No search could ever find such a section, so its questions would be misses in every run, and a changed section
    numbering of the course would look like worse retrieval.
    """
    missing = sorted({case.gold_section for case in gold.cases} - section_ids)
    if missing:
        raise RunRefused("gold_section_missing", f"answer sections {missing} are not sections of the stored course")


async def run(
    gold: GoldSet,
    cases: Sequence[GoldCase],
    sut: SutTransport,
    corpus: Corpus,
    *,
    declaration_file: str,
    held_out_use: int | None,
    harness_commit: str,
    harness_dirty: bool,
) -> RunRecord:
    """Check the answer sections, make both passes over ``cases`` one search at a time, and return the run's record.

    The diagnostic pass asks for the number of hits the app used in the primary pass, so both passes are scored over
    the same places; when no primary search succeeded, it asks for the declared number.
    """
    course_id = gold.declaration.course_id
    started_at = datetime.now(UTC)
    check_sections(gold, corpus.section_ids(course_id))
    snapshot = corpus.snapshot(course_id)

    primary = [await _search(sut, course_id, case.query, {}) for case in cases]
    reported = next((result for _, _, result in primary if result is not None), None)
    k = reported.k if reported is not None else gold.declaration.primary.k
    diagnostic = [await _search(sut, course_id, case.query, {"k": k, "min_score": -1}) for case in cases]

    primary_results = [_pass_result(case, searched, k) for case, searched in zip(cases, primary, strict=True)]
    diagnostic_results = [_pass_result(case, searched, k) for case, searched in zip(cases, diagnostic, strict=True)]
    return RunRecord(
        run_id=uuid.uuid4().hex,
        started_at=started_at,
        harness_commit=harness_commit,
        harness_dirty=harness_dirty,
        declaration_file=declaration_file,
        gold_sha256=gold.sha256,
        course_id=course_id,
        sets=tuple(sorted({case.exam_set for case in cases})),
        held_out_use=held_out_use,
        corpus_snapshot=snapshot,
        embedding_space=reported.embedding_space if reported is not None else None,
        k=reported.k if reported is not None else None,
        min_score=reported.min_score if reported is not None else None,
        summaries={
            "primary": summaries(cases, [result.score for result in primary_results]),
            "diagnostic": summaries(cases, [result.score for result in diagnostic_results]),
        },
        cases=tuple(
            CaseRecord(
                case_id=case.case_id,
                exam_set=case.exam_set,
                chapter=case.chapter,
                knowledge_level=case.knowledge_level,
                gold_section=case.gold_section,
                passes={"primary": first, "diagnostic": second},
            )
            for case, first, second in zip(cases, primary_results, diagnostic_results, strict=True)
        ),
    )


def summaries(cases: Sequence[GoldCase], scores: Sequence[CaseScore]) -> dict[str, Summary]:
    """Summarise one pass over all questions, then per exam set, chapter and knowledge level."""
    groups: defaultdict[str, list[CaseScore]] = defaultdict(list)
    for case, score in zip(cases, scores, strict=True):
        level = case.knowledge_level or "unknown"
        for key in ("all", f"set {case.exam_set}", f"chapter {case.chapter}", f"level {level}"):
            groups[key].append(score)
    return {key: summarise(group) for key, group in groups.items()}


async def _search(sut: SutTransport, course_id: str, query: str, extra: dict[str, JsonValue]) -> _Searched:
    """Send one search of the course; the question goes as it is, with ``extra`` fields added to the request."""
    observation = await sut.request("POST", "/v1/search", json={"course_id": course_id, "query": query, **extra})
    error, result = _read(observation)
    return observation.request_id, error, result


def _read(observation: Observation) -> tuple[str | None, _SearchResult | None]:
    """Read a search response: the result, or why there is none.

    An error response is a failed search of this question only, so one failing case never stops the run.
    """
    if observation.transport_error is not None:
        return observation.transport_error, None
    if observation.status_code != 200:
        try:
            return _ErrorEnvelope.model_validate(observation.body_json).error.code, None
        except ValidationError:
            return f"http_{observation.status_code}", None
    return None, _SearchResult.model_validate(observation.body_json)


def _pass_result(case: GoldCase, searched: _Searched, k: int) -> PassResult:
    """Record one search of one question and score it against the question's answer section over ``k`` places."""
    request_id, error, result = searched
    hits = tuple(
        HitRecord(chunk_id=hit.chunk_id, section_id=hit.section_id, score=hit.score)
        for hit in (result.hits if result is not None else [])
    )
    sections = [hit.section_id for hit in hits] if result is not None else None
    return PassResult(request_id=request_id, error=error, hits=hits, score=score_case(case.gold_section, sections, k))
