"""Unit tests of the check that decides whether two recorded retrieval runs may be compared."""

from datetime import UTC, datetime

import pytest

from evalharness.evaluation.compare import Factor, NotComparable, check_comparable
from evalharness.evaluation.record import RunRecord

_SNAPSHOT_BEFORE = "a" * 64
_SNAPSHOT_AFTER = "b" * 64


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
        "embedding_space": "evalharness_space",
        "k": 5,
        "min_score": 0.83,
        "summaries": {},
        "cases": (),
    }
    return RunRecord.model_validate(values | changes)


@pytest.mark.covers("US-013", "AC-1")
@pytest.mark.parametrize(
    ("after", "factor"),
    [
        pytest.param({"corpus_snapshot": _SNAPSHOT_AFTER}, "corpus", id="the-declared-factor-alone-differs"),
        pytest.param({}, None, id="nothing-declared-nothing-differs"),
    ],
)
def test_runs_that_differ_only_in_the_declared_factor_are_compared(
    after: dict[str, object], factor: Factor | None
) -> None:
    """Checks a run against a second run that differs only in its corpus with the factor "corpus", and against a
    repeat of it with no factor; expects no refusal in both cases.

    A pass means the two comparisons S2b needed (a source added to the corpus, a repeated run) are allowed. A
    failure means the check refuses valid comparisons, so every corpus experiment would be blocked.
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
    ],
)
def test_runs_that_differ_in_anything_but_the_declared_factor_are_refused(
    after: dict[str, object], factor: Factor | None
) -> None:
    """Checks a run against six second runs: corpus and k both changed, the corpus changed while k was declared,
    nothing changed while the corpus was declared, the lowest score changed with no factor declared, another gold
    set, other exam sets. Expects a refusal for each.

    A pass means a difference can always be put down to the one change named before the comparison. A failure
    means a comparison went through although something else changed as well (for example k moved with the corpus,
    and a gain from the larger k is reported as the corpus's gain), or although the questions were not the same.
    """
    with pytest.raises(NotComparable):
        check_comparable(_record(), _record(**after), factor)
