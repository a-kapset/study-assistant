"""A retrieval run: send every chosen question of a gold set to the course search and score what comes back.

Every question is searched twice. The primary pass sends it as a user does, so the app's own number of hits and
lowest score apply. The diagnostic pass asks for the same number of hits with no lowest score, so the misses the
lowest score causes can be told apart from the misses of the search itself. A run that could not measure what it
claims is refused before the first search, also when the app reports that a server its search uses is down. Every
search keeps the path that served it, so a result from a weaker path is never read as the configured one.
"""

import re
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, NamedTuple, Protocol

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from evalharness.datasets.retrieval_gold import GoldCase, GoldDeclaration, GoldSet
from evalharness.evaluation.record import CaseRecord, HitRecord, PassResult, RunRecord
from evalharness.observation import Observation
from evalharness.oracles.retrieval import CaseScore, Summary, score_case, summarise
from evalharness.transport import SutTransport

# Why a run was refused; a caller can act on the code without parsing the message.
type RunRefusedCode = Literal[
    "course_mismatch",
    "unknown_set",
    "held_out_not_allowed",
    "gold_section_missing",
    "gold_question_in_corpus",
    "app_config_unreadable",
    "stack_not_ready",
]


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

    def chunk_texts(self, course_id: str) -> list[str]:
        """Return the text of every stored chunk of the course."""
        ...


class _Hit(BaseModel):
    """The fields of a search hit a run records; the passage text is not read."""

    model_config = ConfigDict(extra="ignore")

    chunk_id: str
    section_id: str | None
    score: float


class SearchResult(BaseModel):
    """The fields of a successful search response a run reads."""

    model_config = ConfigDict(extra="ignore")

    embedding_space: str
    k: int
    min_score: float
    # The search path that served the result: its strategy, and whether a reranker ordered the hits.
    strategy: str
    reranked: bool
    # Set when a model server was down and a weaker path than the configured one answered.
    degraded_reason: str | None = None
    hits: list[_Hit]


class _Ready(BaseModel):
    """The part of the app's GET /ready response a run reads: the overall state and the state of each check."""

    model_config = ConfigDict(extra="ignore")

    status: str
    checks: dict[str, str] = {}


class _ErrorBody(BaseModel):
    """The code of the app's error response."""

    model_config = ConfigDict(extra="ignore")

    code: str


class _ErrorEnvelope(BaseModel):
    """The app's error response, read only for its code."""

    model_config = ConfigDict(extra="ignore")

    error: _ErrorBody


class Searched(NamedTuple):
    """One search as a run sees it."""

    request_id: str
    # Why the search gave no result; None when it ran.
    error: str | None
    result: SearchResult | None


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


# How much of a question's start is looked for in the corpus: long enough to be that question and not a common
# phrase ("Which of the following"), short enough to survive a different ending or answer options printed after it.
_QUESTION_START_WORDS = 12
# A question shorter than this says too little to tell a reprint from ordinary course text, so it is not checked.
_SHORTEST_CHECKED_QUESTION = 6


def check_questions_not_in_corpus(gold: GoldSet, chunk_texts: Sequence[str]) -> None:
    """Refuse the run when the start of a gold question appears word for word in a stored chunk of the course.

    A course source may reprint the very questions the gold set measures with, e.g. a textbook with the official
    sample exam and its answers. The search would then find the question itself, not the passage that answers it,
    and recall would rise without any better retrieval. Text is compared in lower case with punctuation and line
    breaks reduced to single spaces, because the same words come out of a PDF with other breaks and capitals.
    Every question is checked, held-out ones too: a reprint spoils them as well.
    """
    corpus = "\n".join(_words(text) for text in chunk_texts)
    found = []
    for case in gold.cases:
        words = _words(case.query).split()
        if len(words) >= _SHORTEST_CHECKED_QUESTION and " ".join(words[:_QUESTION_START_WORDS]) in corpus:
            found.append(case.case_id)
    if found:
        raise RunRefused("gold_question_in_corpus", f"the stored chunks contain the questions {found}")


def _words(text: str) -> str:
    """Reduce text to its words in lower case, separated by single spaces."""
    return " ".join(re.findall(r"\w+", text.lower()))


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
    """Check the corpus, make both passes over ``cases`` one search at a time, and return the run's record.

    The diagnostic pass asks for the number of hits the app used in the primary pass, so both passes are scored over
    the same places; when no primary search succeeded, it asks for the declared number.
    """
    course_id = gold.declaration.course_id
    started_at = datetime.now(UTC)
    check_sections(gold, corpus.section_ids(course_id))
    check_questions_not_in_corpus(gold, corpus.chunk_texts(course_id))
    snapshot = corpus.snapshot(course_id)
    app_settings = await read_app_settings(sut)
    await check_ready(sut)

    primary = [await search(sut, course_id, case.query, {}) for case in cases]
    reported = next((one.result for one in primary if one.result is not None), None)
    k = reported.k if reported is not None else gold.declaration.primary.k
    diagnostic = [await search(sut, course_id, case.query, {"k": k, "min_score": -1}) for case in cases]

    primary_results = [
        pass_result(case.gold_section, searched, k) for case, searched in zip(cases, primary, strict=True)
    ]
    diagnostic_results = [
        pass_result(case.gold_section, searched, k) for case, searched in zip(cases, diagnostic, strict=True)
    ]
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
        app_settings=app_settings,
        embedding_space=reported.embedding_space if reported is not None else None,
        k=reported.k if reported is not None else None,
        min_score=reported.min_score if reported is not None else None,
        strategy=reported.strategy if reported is not None else None,
        reranked=reported.reranked if reported is not None else None,
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


class _AppConfig(BaseModel):
    """The part of the app's GET /v1/config response a run records."""

    model_config = ConfigDict(extra="ignore")

    settings: dict[str, JsonValue]


async def read_app_settings(sut: SutTransport) -> dict[str, JsonValue]:
    """Read the app's effective settings, so the record says which code and configuration answered.

    Refused when they cannot be read: a record without them could never be compared with another one, since a
    change of the app's code would not be seen.
    """
    observation = await sut.request("GET", "/v1/config")
    if observation.transport_error is not None or observation.status_code != 200:
        reason = observation.transport_error or f"status {observation.status_code}"
        raise RunRefused("app_config_unreadable", f"GET /v1/config failed: {reason}")
    try:
        return _AppConfig.model_validate(observation.body_json).settings
    except ValidationError as error:
        raise RunRefused("app_config_unreadable", f"GET /v1/config has no settings object: {error}") from error


async def check_ready(sut: SutTransport) -> None:
    """Refuse the run unless the app reports that every server its search uses answers.

    The app stays ready when a model server is down and then answers searches on a weaker path: the strategy's
    order without the reranker, or by words only. A run started on such a stack would measure that path under the
    name of the configured one, and only after every search had waited for the missing server.
    """
    observation = await sut.request("GET", "/ready")
    if observation.transport_error is not None or observation.status_code != 200:
        reason = observation.transport_error or f"status {observation.status_code}: {observation.body_text[:200]}"
        raise RunRefused("stack_not_ready", f"GET /ready failed: {reason}")
    try:
        ready = _Ready.model_validate(observation.body_json)
    except ValidationError as error:
        raise RunRefused("stack_not_ready", f"GET /ready has no status: {error}") from error
    if ready.status != "ok":
        down = sorted(check for check, state in ready.checks.items() if state != "ok")
        raise RunRefused(
            "stack_not_ready", f"the app reports {ready.status!r}, not answering: {', '.join(down) or 'not named'}"
        )


def search_path(strategy: str | None, reranked: bool | None) -> str:
    """Write the search path a run recorded, for the first lines of a summary."""
    if strategy is None or reranked is None:
        return "search path not recorded"
    return f"strategy {strategy}, rerank {'on' if reranked else 'off'}"


def weaker_path_counts(results: Iterable[PassResult]) -> str | None:
    """Count the searches answered on a weaker path by the app's reason, e.g. "reranker_unavailable 3"; None when
    the configured path answered every one."""
    reasons = Counter(result.degraded_reason for result in results if result.degraded_reason is not None)
    if not reasons:
        return None
    return ", ".join(f"{reason} {n}" for reason, n in sorted(reasons.items()))


async def search(sut: SutTransport, course_id: str, query: str, extra: dict[str, JsonValue]) -> Searched:
    """Send one search of the course; the question goes as it is, with ``extra`` fields added to the request."""
    observation = await sut.request("POST", "/v1/search", json={"course_id": course_id, "query": query, **extra})
    error, result = _read(observation)
    return Searched(observation.request_id, error, result)


def _read(observation: Observation) -> tuple[str | None, SearchResult | None]:
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
    return None, SearchResult.model_validate(observation.body_json)


def pass_result(gold_section: str, searched: Searched, k: int) -> PassResult:
    """Record one search of one question and score it against the question's answer section over ``k`` places."""
    result = searched.result
    hits = tuple(
        HitRecord(chunk_id=hit.chunk_id, section_id=hit.section_id, score=hit.score)
        for hit in (result.hits if result is not None else [])
    )
    sections = [hit.section_id for hit in hits] if result is not None else None
    return PassResult(
        request_id=searched.request_id,
        error=searched.error,
        degraded_reason=result.degraded_reason if result is not None else None,
        hits=hits,
        score=score_case(gold_section, sections, k),
    )
