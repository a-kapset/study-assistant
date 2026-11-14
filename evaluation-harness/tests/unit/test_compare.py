"""Unit tests of the check that decides whether two recorded runs may be compared, and of what they are compared
on."""

from datetime import UTC, datetime

import pytest

from evalharness.evaluation.compare import AnyRecord, Factor, NotComparable, check_comparable, found_by_id
from evalharness.evaluation.paraphrase_record import ParaphraseRunRecord
from evalharness.evaluation.record import RunRecord

_SNAPSHOT_BEFORE = "a" * 64
_SNAPSHOT_AFTER = "b" * 64
_APP_BEFORE = "e" * 40
_APP_AFTER = "f" * 40


def _app(commit: str) -> dict[str, object]:
    """Build the recorded app settings of an app built from ``commit``."""
    return {"app_settings": {"log_level": "INFO", "git_commit": commit}}


def _record(**changes: object) -> RunRecord:
    """Build a record of a run with no questions; ``changes`` replace the recorded values of the baseline run."""
    values: dict[str, object] = {
        "run_id": "0" * 32,
        "started_at": datetime(2026, 10, 25, tzinfo=UTC),
        "harness_commit": "0" * 40,
        "harness_dirty": False,
        "declaration_file": "datasets/evalharness_course/retrieval_gold_v0.toml",
        "gold_sha256": "c" * 64,
        "course_id": "evalharness_course",
        "sets": ("A", "B"),
        "held_out_use": None,
        "corpus_snapshot": _SNAPSHOT_BEFORE,
        **_app(_APP_BEFORE),
        "embedding_space": "evalharness_space",
        "k": 5,
        "min_score": 0.83,
        "strategy": "vector",
        "reranked": False,
        "summaries": {},
        "cases": (),
    }
    return RunRecord.model_validate(values | changes)


_HIT = {"verdict": "hit", "first_relevant_rank": 1, "reciprocal_rank": 1.0, "precision": 0.2, "irrelevant": 0.0}
_MISS = {"verdict": "miss", "first_relevant_rank": None, "reciprocal_rank": 0.0, "precision": 0.0, "irrelevant": 0.2}


def _search(found: bool, degraded_reason: str | None = None) -> dict[str, object]:
    """Build the record of one search that ran: whether it found the answer section, and the app's reason when a
    weaker path answered it."""
    return {
        "request_id": "evalharness_request",
        "error": None,
        "degraded_reason": degraded_reason,
        "hits": (),
        "score": _HIT if found else _MISS,
    }


def _pair(case_id: str, question: dict[str, object], rewording: dict[str, object]) -> dict[str, object]:
    """Build one pair of a paraphrase run from the records of its two searches of the primary pass."""
    outcome = {"found": "neither", "overlap": 0.0}
    return {
        "case_id": case_id,
        "gold_section": "1.1",
        "passes": {"primary": {"question": question, "rewording": rewording, "outcome": outcome}},
    }


def _pairs_record(*pairs: dict[str, object]) -> ParaphraseRunRecord:
    """Build a record of a paraphrase run of ``pairs`` under the same settings as the baseline retrieval run."""
    values: dict[str, object] = {
        "run_id": "1" * 32,
        "started_at": datetime(2026, 11, 14, tzinfo=UTC),
        "harness_commit": "0" * 40,
        "harness_dirty": False,
        "pair_file": "datasets/evalharness_course/paraphrase_v0.toml",
        "set_sha256": "c" * 64,
        "course_id": "evalharness_course",
        "corpus_snapshot": _SNAPSHOT_BEFORE,
        **_app(_APP_BEFORE),
        "embedding_space": "evalharness_space",
        "k": 5,
        "min_score": 0.83,
        "strategy": "vector",
        "reranked": False,
        "summaries": {},
        "pairs": pairs,
    }
    return ParaphraseRunRecord.model_validate(values)


@pytest.mark.covers("US-013", "AC-1")
@pytest.mark.parametrize(
    ("after", "factor"),
    [
        pytest.param({"corpus_snapshot": _SNAPSHOT_AFTER}, "corpus", id="the-declared-factor-alone-differs"),
        pytest.param({}, None, id="nothing-declared-nothing-differs"),
        pytest.param(_app(_APP_AFTER), "app", id="the-app-commit-alone-differs"),
        pytest.param(_app(_APP_BEFORE + "-dirty"), "app", id="uncommitted-app-changes-on-the-same-commit"),
        pytest.param({"strategy": "hybrid"}, "strategy", id="the-search-strategy-alone-differs"),
        pytest.param({"reranked": True}, "rerank", id="the-reranker-alone-was-switched-on"),
    ],
)
def test_runs_that_differ_only_in_the_declared_factor_are_compared(
    after: dict[str, object], factor: Factor | None
) -> None:
    """Checks a run against a second run that differs only in its corpus with the factor "corpus", against a repeat
    of it with no factor, against runs of another app commit, or of the same commit with uncommitted changes,
    with the factor "app", and against runs of the same commit served by another search strategy, or with the
    reranker on, with the factors "strategy" and "rerank"; expects no refusal in all six cases.

    A pass means the comparisons the experiments need (a source added to the corpus, a repeated run, a change of the
    app's code, committed or not yet, a search variant switched by configuration) are allowed. A failure means the
    check refuses valid comparisons, so these experiments would be blocked; for "app" it also means a code change
    is not seen as the change it is.
    """
    check_comparable(_record(), _record(**after), factor)


@pytest.mark.covers("US-013", "AC-1")
@pytest.mark.parametrize(
    ("after", "factor"),
    [
        pytest.param({"corpus_snapshot": _SNAPSHOT_AFTER, "k": 10}, "corpus", id="a-second-value-differs-too"),
        pytest.param({"corpus_snapshot": _SNAPSHOT_AFTER}, "k", id="another-value-than-the-declared-one-differs"),
        pytest.param({}, "corpus", id="the-declared-factor-did-not-change"),
        pytest.param({"min_score": 0.85}, None, id="a-value-differs-with-no-factor-declared"),
        pytest.param({"gold_sha256": "d" * 64}, None, id="other-questions-or-rules"),
        pytest.param({"sets": ("A",)}, None, id="other-exam-sets"),
        pytest.param(_app(_APP_AFTER), None, id="the-app-commit-differs-with-no-factor-declared"),
        pytest.param({"corpus_snapshot": _SNAPSHOT_AFTER, **_app(_APP_AFTER)}, "corpus", id="app-changed-with-corpus"),
        pytest.param({"strategy": "hybrid"}, None, id="the-strategy-differs-with-no-factor-declared"),
        pytest.param({"strategy": "hybrid", "reranked": True}, "strategy", id="rerank-switched-on-with-the-strategy"),
    ],
)
def test_runs_that_differ_in_anything_but_the_declared_factor_are_refused(
    after: dict[str, object], factor: Factor | None
) -> None:
    """Checks a run against ten second runs: corpus and k both changed, the corpus changed while k was declared,
    nothing changed while the corpus was declared, the lowest score changed with no factor declared, another gold
    set, other exam sets, another app commit with no factor declared, another app commit with the corpus declared,
    another search strategy on the same commit with no factor declared, and the reranker switched on together with
    the strategy while only the strategy was declared. Expects a refusal for each.

    A pass means a difference can always be put down to the one change named before the comparison. A failure
    means a comparison went through although something else changed as well (for example k moved with the corpus,
    and a gain from the larger k is reported as the corpus's gain; the app's code changed between two runs that
    are read as a repeat; or a search variant set by configuration, which leaves the app's commit as it was, is
    read as a repeat or credited with the reranker's gain), or although the questions were not the same.
    """
    with pytest.raises(NotComparable):
        check_comparable(_record(), _record(**after), factor)


@pytest.mark.covers("US-013", "AC-1")
@pytest.mark.parametrize(
    ("before", "after", "factor"),
    [
        pytest.param({"app_settings": None}, {"app_settings": None}, None, id="recorded-before-app-settings-were-kept"),
        pytest.param(_app("unknown"), _app("unknown"), None, id="built-without-a-commit"),
        pytest.param(_app("unknown"), _app(_APP_AFTER), "app", id="one-built-without-a-commit"),
        pytest.param(_app(_APP_BEFORE + "-dirty"), _app(_APP_BEFORE + "-dirty"), None, id="same-dirty-commit"),
        pytest.param(_app(_APP_BEFORE + "-dirty"), _app(_APP_BEFORE + "-dirty"), "app", id="same-dirty-commit-as-app"),
    ],
)
def test_runs_whose_app_code_cannot_be_told_are_refused(
    before: dict[str, object], after: dict[str, object], factor: Factor | None
) -> None:
    """Checks pairs of runs whose app code cannot be told: records without app settings (written before runs kept
    them), apps built without a commit ("unknown"), and two runs of the same commit with uncommitted changes.
    Every other recorded value is equal. Expects a refusal that names the app, with and without the factor "app".

    A pass means a comparison never rests on a guess about which code answered. A failure means two runs whose code
    may differ are read as a repeat, or a change of the code is assumed although nothing shows it, so a code change
    (a new search strategy, a mutant image) could be credited with a gain it did not make. All values but the app
    identity are equal, so the refusal can come only from that rule.
    """
    with pytest.raises(NotComparable, match="app") as refusal:
        check_comparable(_record(**before), _record(**after), factor)

    assert "differ" not in str(refusal.value), f"refused by the one-factor rule, not the app identity: {refusal.value}"


def _failed_case() -> dict[str, object]:
    """Build one question whose search of the primary pass failed and whose diagnostic search found nothing."""
    failed = {"verdict": "failed", "first_relevant_rank": None, "reciprocal_rank": 0.0, "precision": 0.0}
    empty = failed | {"verdict": "empty"}
    result = {"request_id": "evalharness_request", "error": "embedding_unavailable", "hits": ()}
    return {
        "case_id": "A-1",
        "exam_set": "A",
        "chapter": "1",
        "knowledge_level": None,
        "gold_section": "1.1",
        "passes": {
            "primary": result | {"score": failed | {"irrelevant": 0.0}},
            "diagnostic": result | {"score": empty | {"irrelevant": 0.0}},
        },
    }


@pytest.mark.covers("US-016", "AC-6")
def test_a_run_with_a_failed_search_is_not_compared() -> None:
    """Checks a repeated run against a run in which one search of the primary pass failed; expects a refusal that
    names the failed search.

    A pass means an outage during one of the runs cannot show up as lost questions and a "worse" verdict. A failure
    means the comparison counts a search that measured nothing as not found.
    """
    with pytest.raises(NotComparable, match="failed search"):
        check_comparable(_record(), _record(cases=(_failed_case(),)), None)


def _retrieval_run_without_reranker() -> RunRecord:
    """Build a retrieval run whose one question was found, but on the path without the reranker."""
    case = {"case_id": "A-1", "exam_set": "A", "chapter": "1", "knowledge_level": None, "gold_section": "1.1"}
    searched = _search(found=True, degraded_reason="reranker_unavailable")
    return _record(cases=(case | {"passes": {"primary": searched, "diagnostic": _search(found=True)}},))


def _paraphrase_run_without_reranker() -> ParaphraseRunRecord:
    """Build a paraphrase run whose one pair found its answer twice, the rewording on the path without the
    reranker."""
    rewording = _search(found=True, degraded_reason="reranker_unavailable")
    return _pairs_record(_pair("QA-1", _search(found=True), rewording))


@pytest.mark.covers("US-013", "AC-1")
@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param(_record(), _retrieval_run_without_reranker(), id="retrieval-run"),
        pytest.param(_pairs_record(), _paraphrase_run_without_reranker(), id="rewording-of-a-paraphrase-run"),
    ],
)
def test_a_run_with_a_search_answered_on_a_weaker_path_is_not_compared(before: AnyRecord, after: AnyRecord) -> None:
    """Checks a run against a repeat of it in which one search was answered without the reranker because the rerank
    server was down: the search of a retrieval question, and the rewording's search of a paraphrase pair. The search
    answered 200 and found the answer section. Expects a refusal that names the weaker path.

    A pass means a comparison of search variants never reads a fallback's results as the variant's own. A failure
    means such a run is compared: a rerank variant whose server dropped out for part of the run is judged on a mix
    of two orders, and nothing else shows it, because every search succeeded and scored normally. For paraphrase
    runs it also means the check reads only one side of the pairs.
    """
    with pytest.raises(NotComparable, match="weaker path"):
        check_comparable(before, after, None)


@pytest.mark.covers("US-013", "AC-4")
def test_paraphrase_runs_are_compared_on_their_rewordings() -> None:
    """Builds a paraphrase run of two pairs: in the first the question finds its answer and the rewording does not,
    in the second it is the other way round. Expects the compared result to be "not found" for the first pair and
    "found" for the second.

    A pass means two paraphrase runs are compared on what their rewordings found. A failure means the comparison
    reads the questions' side: they keep the course's own term and are found by every variant alike, so a variant
    that helps learners who ask in other words would show no change, and the decision between search variants
    would rest on a number that cannot move.
    """
    record = _pairs_record(
        _pair("QA-1", _search(found=True), _search(found=False)),
        _pair("QA-2", _search(found=False), _search(found=True)),
    )

    found = found_by_id(record)

    assert found == {"QA-1": False, "QA-2": True}, f"the compared result per pair was {found}"
