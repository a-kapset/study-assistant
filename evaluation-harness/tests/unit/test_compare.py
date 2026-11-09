"""Unit tests of the check that decides whether two recorded retrieval runs may be compared."""

from datetime import UTC, datetime

import pytest

from evalharness.evaluation.compare import Factor, NotComparable, check_comparable
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
        pytest.param(_app(_APP_AFTER), "app", id="the-app-commit-alone-differs"),
        pytest.param(_app(_APP_BEFORE + "-dirty"), "app", id="uncommitted-app-changes-on-the-same-commit"),
    ],
)
def test_runs_that_differ_only_in_the_declared_factor_are_compared(
    after: dict[str, object], factor: Factor | None
) -> None:
    """Checks a run against a second run that differs only in its corpus with the factor "corpus", against a repeat
    of it with no factor, and against runs of another app commit, or of the same commit with uncommitted changes,
    with the factor "app"; expects no refusal in all four cases.

    A pass means the comparisons the experiments need (a source added to the corpus, a repeated run, a change of the
    app's code, committed or not yet) are allowed. A failure means the check refuses valid comparisons, so these
    experiments would be blocked; for "app" it also means a code change is not seen as the change it is.
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
    ],
)
def test_runs_that_differ_in_anything_but_the_declared_factor_are_refused(
    after: dict[str, object], factor: Factor | None
) -> None:
    """Checks a run against eight second runs: corpus and k both changed, the corpus changed while k was declared,
    nothing changed while the corpus was declared, the lowest score changed with no factor declared, another gold
    set, other exam sets, another app commit with no factor declared, another app commit with the corpus declared.
    Expects a refusal for each.

    A pass means a difference can always be put down to the one change named before the comparison. A failure
    means a comparison went through although something else changed as well (for example k moved with the corpus,
    and a gain from the larger k is reported as the corpus's gain; or the app's code changed between two runs that
    are read as a repeat), or although the questions were not the same.
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
