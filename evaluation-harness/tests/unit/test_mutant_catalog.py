"""Unit tests of the mutant catalog and of how the outcome of the api tests is read; files written by the tests,
never the stack."""

from pathlib import Path

import pytest

from evalharness.mutation.catalog import CatalogError, load_catalog
from evalharness.mutation.oracles import read_test_outcomes

_CATALOG = """
course_id = "evalharness_course"

[checks]
retrieval_gold = "gold.toml"

[mutants.baseline]
kind = "baseline"
expected_symptom = "evalharness"

[mutants.low_score]
kind = "configuration"
component = "search"
settings = {{ APP_SEARCH__MIN_SCORE = "0.88" }}
expected_symptom = "evalharness"
expected_killers = ["{killer}"]
origin = "invented"
"""

_RESULTS = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest">
  <testcase classname="t" name="test_held"/>
  <testcase classname="t" name="test_broken"><failure message="relation broken"/></testcase>
  <testcase classname="t" name="test_not_started"><error message="fixture failed"/></testcase>
  <testcase classname="t" name="test_broken_and_teardown"><failure message="x"/><error message="y"/></testcase>
</testsuite></testsuites>
"""


@pytest.mark.covers("US-019", "AC-1")
@pytest.mark.parametrize(
    "killer",
    [
        pytest.param("paraphrase", id="a check the catalog does not make"),
        pytest.param("retrieval_gold_v0", id="a name no check has"),
    ],
)
def test_a_catalog_whose_mutant_expects_a_check_that_is_not_made_is_refused(tmp_path: Path, killer: str) -> None:
    """A mutant names a check as its expected killer that the catalog does not make, or that does not exist at all.
    Reading the catalog must fail and name the mutant.

    A pass says that every expected cell of the matrix belongs to a check that really runs. A failure points to the
    checks of catalog.py. Without it a mistyped name would turn an expected cell into one "not aimed at" without any
    sign, and a check that misses its mutant would never be counted as a blind spot.
    """
    path = tmp_path / "catalog.toml"
    path.write_text(_CATALOG.format(killer=killer), encoding="utf-8")

    with pytest.raises(CatalogError, match=killer) as refused:
        load_catalog(path)
    assert "low_score" in str(refused.value), f"the refusal does not name the mutant: {refused.value}"


def test_the_outcome_of_each_api_test_is_read_and_an_error_is_not_a_failure(tmp_path: Path) -> None:
    """A JUnit file as pytest writes it holds a passed test, a failed one, one whose setup failed and one that failed
    and then failed again in its teardown. Each must be read as passed, failed, error and failed.

    A pass says that a relation test that could not start (for example its fixture found no app) is kept apart from
    a relation that was tried and broken. A failure points to read_test_outcomes in oracles.py. Without it a check
    that never ran would count as one that noticed the mutant.
    """
    path = tmp_path / "results.xml"
    path.write_text(_RESULTS, encoding="utf-8")

    outcomes = read_test_outcomes(path)

    assert outcomes == {
        "test_held": "passed",
        "test_broken": "failed",
        "test_not_started": "error",
        "test_broken_and_teardown": "failed",
    }, f"outcomes read: {outcomes}"
