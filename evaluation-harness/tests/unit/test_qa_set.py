"""Unit tests of the Q&A set file: what is refused before a run can start."""

import pytest
from pydantic import ValidationError

from evalharness.datasets.qa_set import QaSet


@pytest.mark.covers("US-011", "AC-1")
def test_two_cases_with_one_id_are_refused() -> None:
    """Loads a set in which two questions share the id "QA-1"; expects the set to be refused, naming the id.

    A pass means every question of a set has its own id. A failure means a question copied together with its id
    would be mixed up with the other one in a run record, and comparing two runs case by case would compare
    different questions.
    """
    case = {"question": "What is a test?", "gold_section": "1.1"}
    data = {
        "course_id": "evalharness_course",
        "targets": {"answered_min": 0.85, "gold_cited_min": 0.75, "unsupported_answers_max": 0},
        "cases": [{"id": "QA-1", **case}, {"id": "QA-1", **case}],
    }

    with pytest.raises(ValidationError) as refused:
        QaSet.model_validate(data)

    assert "QA-1" in str(refused.value), f"the refusal does not name the repeated id: {refused.value}"
