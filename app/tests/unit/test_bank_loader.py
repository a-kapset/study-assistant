"""Reading a course's question bank from the bank files its manifest lists."""

import json
from pathlib import Path

import pytest

from study_assistant.courses.models import CourseManifest
from study_assistant.exams.loader import QuestionBankError, read_course_bank


def _manifest(*bank_files: str) -> CourseManifest:
    """A course with one text source and the given bank files, in this order."""
    sources: list[dict[str, str]] = [{"file": "text.pdf", "role": "material", "authority": "primary"}]
    sources += [{"file": file, "role": "question_bank", "authority": "primary"} for file in bank_files]
    return CourseManifest.model_validate(
        {
            "id": "river_board",
            "title": "The river board",
            "description": "A course used only by this test.",
            "language": "en",
            "structure": {"section_heading": r"^Part (?P<id>\d+) (?P<title>.+)$"},
            "sources": sources,
            "exam": {"pass_mark_percent": 60} if bank_files else None,
        }
    )


def _write_bank(folder: Path, name: str, *item_ids: str) -> None:
    """Write a bank file with one valid question per id."""
    lines = [
        json.dumps(
            {
                "id": item_id,
                "stem": f"Question {item_id} about the river board.",
                "options": [{"id": "a", "text": "Answer a"}, {"id": "b", "text": "Answer b"}],
                "keys": ["a"],
                "objective_id": "OBJ-01",
                "source": {"document": "rules_questions.pdf", "version": "1", "question": item_id},
            }
        )
        for item_id in item_ids
    ]
    (folder / name).write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


@pytest.mark.covers("US-020", "AC-10")
def test_questions_of_every_bank_file_are_read_in_manifest_order(tmp_path: Path) -> None:
    """Two bank files, listed second-file-first in the manifest. Catches a loader that reads only the first bank
    or orders the files by name instead of by the manifest."""
    _write_bank(tmp_path, "bank_one.jsonl", "Q1", "Q2")
    _write_bank(tmp_path, "bank_two.jsonl", "Q3")

    items = read_course_bank(_manifest("bank_two.jsonl", "bank_one.jsonl"), tmp_path)

    assert [item.id for item in items] == ["Q3", "Q1", "Q2"], f"wrong questions or order: {[item.id for item in items]}"


@pytest.mark.covers("US-020", "AC-10")
def test_a_question_id_used_in_two_bank_files_is_refused(tmp_path: Path) -> None:
    """Each file is valid alone, but both hold Q2. Catches a second question silently replacing the first under
    the same id: answers stored for one would be shown with the other."""
    _write_bank(tmp_path, "bank_one.jsonl", "Q1", "Q2")
    _write_bank(tmp_path, "bank_two.jsonl", "Q2")

    with pytest.raises(QuestionBankError) as caught:
        read_course_bank(_manifest("bank_one.jsonl", "bank_two.jsonl"), tmp_path)

    message = str(caught.value)
    assert "'Q2'" in message and "bank_one.jsonl" in message and "bank_two.jsonl" in message, (
        f"the error must name the id and both files: {message}"
    )


@pytest.mark.covers("US-020", "AC-10")
def test_a_missing_bank_file_is_an_error_naming_the_file(tmp_path: Path) -> None:
    """The manifest lists a bank that is not in the course folder. Catches a missing file read as an empty bank:
    ingestion would then retire every stored question."""
    with pytest.raises(QuestionBankError) as caught:
        read_course_bank(_manifest("bank_one.jsonl"), tmp_path)

    assert "bank_one.jsonl" in str(caught.value), f"the error must name the file: {caught.value}"


def test_a_course_without_a_bank_gives_no_questions(tmp_path: Path) -> None:
    """A course with text sources only: no file is opened and no question returned, so ingesting it never fails
    for a bank it does not have."""
    assert read_course_bank(_manifest(), tmp_path) == (), "a course without a bank file must give no questions"
