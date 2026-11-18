"""Unit tests of the proof that a mutant is live: two readings of the app's reported settings, never the stack."""

import pytest
from pydantic import JsonValue

from evalharness.mutation.proof import apply_problem, changes

FIRST: dict[str, JsonValue] = {"git_commit": "evalharness_commit", "search": {"default_k": 5, "min_score": 0.83}}


def _reading(**search: JsonValue) -> dict[str, JsonValue]:
    """Return the first reading with some search settings set to other values, as the app would report them."""
    first_search = FIRST["search"]
    assert isinstance(first_search, dict)
    return {**FIRST, "search": {**first_search, **search}}


@pytest.mark.covers("US-019", "AC-2", "AC-4")
@pytest.mark.parametrize(
    ("declared", "reported", "valid"),
    [
        pytest.param({"APP_SEARCH__MIN_SCORE": "0.88"}, _reading(min_score=0.88), True, id="exactly the declared"),
        pytest.param({"APP_SEARCH__MIN_SCORE": "0.88"}, _reading(), False, id="nothing changed"),
        pytest.param({"APP_SEARCH__MIN_SCORE": "0.88"}, _reading(min_score=0.86), False, id="another value"),
        pytest.param(
            {"APP_SEARCH__MIN_SCORE": "0.88"}, _reading(min_score=0.88, default_k=1), False, id="one more changed"
        ),
        pytest.param({}, _reading(), True, id="baseline with nothing changed"),
        pytest.param({}, _reading(default_k=1), False, id="baseline with a change"),
    ],
)
def test_a_mutant_run_is_valid_only_when_exactly_the_declared_settings_changed(
    declared: dict[str, str], reported: dict[str, JsonValue], valid: bool
) -> None:
    """The proof compares what the app reports after the restart with the reading before it. It accepts the run
    only when every declared setting changed to its declared value and nothing else changed; the baseline declares
    nothing, so for it any change is a problem.

    A pass says that a mutant that was never applied, applied with another value, or applied together with a second
    change is told apart from a live one. A failure points to the comparison in proof.py. Without it a mutant the
    app never ran would read as a defect no check sees, and a second change would make a kill unattributable.
    """
    problem = apply_problem(declared, changes(FIRST, reported))

    assert (problem is None) == valid, f"declared {declared}, reported {reported['search']}: problem {problem!r}"
