"""Unit tests of how the kill matrix reads one check's record from a mutant run's folder: records written by the tests
to a temporary folder, read with the committed dataset files of the first course; never the stack."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import BaseModel

from evalharness.evaluation.answer_record import AnswerRunRecord
from evalharness.evaluation.paraphrase_record import ParaphraseRunRecord
from evalharness.evaluation.record import RunRecord
from evalharness.mutation.catalog import CheckName, Mutant
from evalharness.mutation.matrix import read_measured
from evalharness.mutation.record import CheckResult

NO_MUTANT = Mutant(kind="baseline", expected_symptom="evalharness")
# The values every run record shares; each check's record adds its own.
_RUN: dict[str, object] = {
    "run_id": "0" * 32,
    "started_at": datetime(2026, 11, 20, tzinfo=UTC),
    "harness_commit": "0" * 40,
    "harness_dirty": False,
    "course_id": "evalharness_course",
    "corpus_snapshot": "a" * 64,
    "app_settings": {"git_commit": "e" * 40},
}
_SEARCHED = {"embedding_space": "evalharness_space", "k": 5, "min_score": 0.83, "summaries": {}}
_FAILED = {
    "verdict": "failed",
    "first_relevant_rank": None,
    "reciprocal_rank": 0.0,
    "precision": 0.0,
    "irrelevant": 0.0,
}
_HIT = {"verdict": "hit", "first_relevant_rank": 1, "reciprocal_rank": 1.0, "precision": 0.2, "irrelevant": 0.0}


def _search(error: str | None) -> dict[str, object]:
    """Build the record of one search: failed with ``error``, or found its section when ``error`` is None."""
    return {
        "request_id": "evalharness_request",
        "error": error,
        "hits": (),
        "score": _FAILED if error is not None else _HIT,
    }


def _retrieval() -> RunRecord:
    """A retrieval run in which one search got no response in time and one failed by the app's own error."""
    cases = [
        {
            "case_id": case_id,
            "exam_set": "A",
            "chapter": "1",
            "knowledge_level": None,
            "gold_section": "1.1",
            "passes": {"primary": _search(error)},
        }
        for case_id, error in (("QA-SLIP", "timeout"), ("QA-APP", "http_500"))
    ]
    values = {
        **_RUN,
        **_SEARCHED,
        "declaration_file": "datasets/istqb_ctfl/retrieval_gold_v0.toml",
        "gold_sha256": "c" * 64,
        "sets": ("A",),
        "held_out_use": None,
        "cases": cases,
    }
    return RunRecord.model_validate(values)


def _paraphrase() -> ParaphraseRunRecord:
    """A paraphrase run in which the rewording of one pair got no response in time and the other pair ran."""
    pairs = [
        {
            "case_id": case_id,
            "gold_section": "1.1",
            "passes": {
                "primary": {
                    "question": _search(None),
                    "rewording": _search(error),
                    "outcome": {"found": "kept" if error is None else "failed", "overlap": None},
                }
            },
        }
        for case_id, error in (("QA-SLIP", "timeout"), ("QA-OK", None))
    ]
    values = {
        **_RUN,
        **_SEARCHED,
        "pair_file": "datasets/istqb_ctfl/paraphrase_v0.toml",
        "set_sha256": "c" * 64,
        "pairs": pairs,
    }
    return ParaphraseRunRecord.model_validate(values)


def _answers() -> AnswerRunRecord:
    """An answer run in which one ask got no response in time and one failed by the app's own error."""
    asks = [
        {
            "case_id": case_id,
            "repetition": 1,
            "gold_section": "1.1",
            "search_request_id": "evalharness_search",
            "ask_request_id": "evalharness_ask",
            "ask_error": error,
            "failure_label": label,
            "abstain_reason": None,
            "cited_sections": (),
            "invalid_citations": None,
            "search_sections": None,
            "model_calls": (),
            "cost_usd": None,
            "score": {"verdict": "failed", "gold_retrieved": None},
        }
        for case_id, error, label in (("QA-SLIP", "timeout", "no_response"), ("QA-APP", "http_500", "app_error"))
    ]
    values = {
        **_RUN,
        "qa_file": "datasets/istqb_ctfl/qa_v1.toml",
        "qa_sha256": "c" * 64,
        "models_returned": (),
        "repetitions": 1,
        "summaries": (),
        "cases": asks,
    }
    return AnswerRunRecord.model_validate(values)


@pytest.mark.covers("US-019", "AC-6")
@pytest.mark.parametrize(
    ("check", "record"),
    [
        pytest.param("retrieval_gold", _retrieval(), id="retrieval run"),
        pytest.param("paraphrase", _paraphrase(), id="paraphrase run"),
        pytest.param("answer_run", _answers(), id="answer run"),
        pytest.param("failure_labels", _answers(), id="failure labels"),
    ],
)
def test_every_check_leaves_out_the_questions_the_stack_failed_on(
    tmp_path: Path, check: CheckName, record: BaseModel
) -> None:
    """Writes a record of the check to a run folder, with one question that got no response in time and one that
    the app itself failed (or that ran), and reads it as the matrix does, for the run without a mutant. Expects only
    the question without a response among the questions left out of the comparison.

    What a pass tells: every check's cell gets the questions the stack failed on from its own record, so the rule of
    leaving them out applies to every column. A failure points to read_measured in matrix.py: the rule would exist
    in the verdict but no column would use it, and a timeout would again void a column or kill a mutant.
    """
    (tmp_path / "record.json").write_text(record.model_dump_json(), encoding="utf-8")

    measured = read_measured(check, CheckResult(status="ran", result_file="record.json"), tmp_path, NO_MUTANT)

    assert measured.left_out == {"QA-SLIP"}, f"{check} left out {set(measured.left_out)}, expected only QA-SLIP"
