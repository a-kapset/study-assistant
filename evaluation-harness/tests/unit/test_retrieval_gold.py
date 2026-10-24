"""Unit tests of building a gold set: small invented banks in a temporary folder, never the real question bank."""

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evalharness.datasets.retrieval_gold import GoldDeclaration, GoldSetError, build_gold_set

BANK_FILE = "bank.jsonl"
# Same rule as the first course's declaration: "FL-1.2.3" -> section "1.2" in chapter "1".
OBJECTIVE_SECTION = r"FL-(?P<section>(?P<chapter>\d+)\.\d+)\.\d+"


def _item(item_id: str, objective_id: str) -> str:
    """Return one bank line in the common format with the fields the gold set reads, plus one it ignores."""
    return json.dumps(
        {
            "id": item_id,
            "stem": f"Question {item_id}?",
            "objective_id": objective_id,
            "knowledge_level": "K2",
            "keys": ["a"],
        }
    )


def _bank(folder: Path, *lines: str) -> str:
    """Write a bank file into ``folder`` and return its SHA-256, as a declaration would pin it."""
    content = ("\n".join(lines) + "\n").encode("utf-8")
    (folder / BANK_FILE).write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _declaration(bank_sha256: str, **changes: object) -> GoldDeclaration:
    """Build a declaration for the test bank: set D held out, recall at 5 on set A; ``changes`` replace its keys."""
    fields: dict[str, object] = {
        "course_id": "evalharness_course",
        "bank_file": BANK_FILE,
        "bank_sha256": bank_sha256,
        "objective_section": OBJECTIVE_SECTION,
        "held_out_sets": ["D"],
        "primary": {"metric": "recall", "k": 5, "sets": ["A"], "target": 0.8},
    }
    return GoldDeclaration.model_validate(fields | changes)


@pytest.mark.covers("US-007", "AC-1")
def test_a_bank_with_another_hash_is_refused(tmp_path: Path) -> None:
    """Pins the hash of one bank, then rewrites the bank with a changed question and builds the gold set from it.

    A pass means a run cannot start on questions other than the ones the target and earlier runs were measured on.
    A failure means a regenerated or edited bank would be measured silently, and its numbers compared with runs on
    the old questions as if nothing had changed.
    """
    pinned = _bank(tmp_path, _item("A-1", "FL-1.2.3"))
    _bank(tmp_path, _item("A-1", "FL-1.3.1"))

    with pytest.raises(GoldSetError) as refused:
        build_gold_set(_declaration(pinned), tmp_path)

    assert refused.value.code == "bank_changed", f"a changed bank must be refused as bank_changed, got {refused.value}"


@pytest.mark.covers("US-007", "AC-1")
def test_the_answer_section_and_chapter_come_from_the_learning_objective(tmp_path: Path) -> None:
    """Builds a gold set from one question with objective FL-4.2.1 and expects section 4.2, chapter 4 and set A.

    A pass means every question is scored against the section its learning objective belongs to, and its results
    are reported under the right chapter and question set. A failure means the gold labels are wrong, so the whole
    run measures against wrong answers; nothing in the app checks these labels.
    """
    gold = build_gold_set(_declaration(_bank(tmp_path, _item("A-7", "FL-4.2.1"))), tmp_path)

    case = gold.cases[0]
    assert (case.gold_section, case.chapter, case.exam_set) == ("4.2", "4", "A"), (
        f"objective FL-4.2.1 of question A-7 must give section 4.2, chapter 4, set A, got {case}"
    )


@pytest.mark.covers("US-007", "AC-1")
def test_questions_of_held_out_sets_are_marked(tmp_path: Path) -> None:
    """Builds a gold set from one question of set A and one of set D, with D declared as held out.

    A pass means the runner can keep held-out questions out of a run unless they are asked for. A failure means
    held-out questions would be mixed into the tuning runs, and the final measurement would no longer be clean.
    """
    sha = _bank(tmp_path, _item("A-1", "FL-1.2.3"), _item("D-1", "FL-1.2.3"))

    gold = build_gold_set(_declaration(sha), tmp_path)

    held_out = {case.case_id: case.held_out for case in gold.cases}
    assert held_out == {"A-1": False, "D-1": True}, f"only the question of set D must be held out, got {held_out}"


@pytest.mark.covers("US-007", "AC-1")
@pytest.mark.parametrize(
    "objective_id",
    [
        pytest.param("FL-1.2", id="no-objective-number"),
        pytest.param("FL-1.2.3x", id="extra-characters-at-the-end"),
    ],
)
def test_an_objective_that_names_no_section_is_refused_with_its_line(tmp_path: Path, objective_id: str) -> None:
    """Puts a question whose objective id does not follow the declared rule on line 2 of the bank.

    A pass means a question never gets a guessed or partial answer section, and the error points to the line to
    fix. A failure means such a question is scored against a wrong section or the error does not say where it is.
    """
    sha = _bank(tmp_path, _item("A-1", "FL-1.2.3"), _item("A-2", objective_id))

    with pytest.raises(GoldSetError) as refused:
        build_gold_set(_declaration(sha), tmp_path)

    assert (refused.value.code, "line 2:" in str(refused.value)) == ("objective_not_matched", True), (
        f"objective {objective_id!r} on line 2 must give objective_not_matched at line 2, got {refused.value}"
    )


@pytest.mark.covers("US-007", "AC-2")
def test_a_main_result_on_held_out_questions_is_refused() -> None:
    """Builds a declaration whose main result is measured on set D while set D is also declared as held out.

    A pass means the target can never be checked, or tuned, on the held-out questions. A failure means an edit of
    the declaration could quietly turn the final measurement set into a tuning set.
    """
    with pytest.raises(ValidationError, match="held out"):
        _declaration("0" * 64, primary={"metric": "recall", "k": 5, "sets": ["A", "D"], "target": 0.8})


@pytest.mark.covers("US-007", "AC-1")
def test_the_gold_set_hash_changes_with_what_is_measured(tmp_path: Path) -> None:
    """Builds the same bank twice, once with k 5 and once with k 3, and expects two different gold-set hashes.

    A pass means two runs with the same gold-set hash were measured with the same questions, labels and rules, so
    they may be compared. A failure means a run with another k or target would look comparable to the old ones.
    """
    sha = _bank(tmp_path, _item("A-1", "FL-1.2.3"))
    at_5 = build_gold_set(_declaration(sha), tmp_path)
    at_3 = build_gold_set(
        _declaration(sha, primary={"metric": "recall", "k": 3, "sets": ["A"], "target": 0.8}), tmp_path
    )

    assert at_5.sha256 != at_3.sha256, f"k 5 and k 3 must give different gold-set hashes, both are {at_5.sha256}"
