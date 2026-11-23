"""Reading the question bank of a course: every bank file its manifest lists, as one checked list of questions."""

from pathlib import Path

from study_assistant.courses.models import CourseManifest
from study_assistant.exams.bank import BankError, BankItem, read_bank


class QuestionBankError(Exception):
    """A course's question bank cannot be used; the message names the file and why."""


def read_course_bank(manifest: CourseManifest, course_dir: Path) -> tuple[BankItem, ...]:
    """Read the bank files of a course from ``course_dir`` in manifest order; a course without one gives no questions.

    The first file that cannot be read, or holds an unusable line, stops the reading. Question ids must be unique
    across all files of the course, because sessions and answers refer to a question by its id.
    """
    items: list[BankItem] = []
    seen: dict[str, str] = {}
    for source in manifest.sources:
        if source.role != "question_bank":
            continue
        try:
            read = read_bank(course_dir / source.file)
        except OSError as exc:
            raise QuestionBankError(f"{source.file}: cannot read the file: {exc.strerror}") from exc
        except BankError as exc:
            raise QuestionBankError(str(exc)) from exc
        for item in read:
            if item.id in seen:
                raise QuestionBankError(f"{source.file}: question id {item.id!r} is already used in {seen[item.id]}")
            seen[item.id] = source.file
            items.append(item)
    return tuple(items)
