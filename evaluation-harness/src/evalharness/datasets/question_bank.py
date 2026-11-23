"""The harness's own reading of a course's question bank: for each question its options, its key and its objective.

The checks of practice exams take the answer key from here, from the bank file itself, and never from what the
app replies: an app that stored a wrong key would otherwise agree with itself. Only the fields those checks need
are read; the full format of a bank belongs to the app's bank reader.
"""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class QuestionBankError(ValueError):
    """The bank file cannot be used as the reference of a check; the message names the file and the line."""


class _Option(BaseModel):
    """One answer option; only its id is needed."""

    model_config = ConfigDict(extra="ignore", frozen=True)
    id: str = Field(min_length=1)


class _Line(BaseModel):
    """The fields of one bank line the exam checks need; other fields are ignored."""

    model_config = ConfigDict(extra="ignore", frozen=True)
    id: str = Field(min_length=1)
    options: tuple[_Option, ...] = Field(min_length=2)
    keys: tuple[str, ...] = Field(min_length=1)
    objective_id: str = Field(min_length=1)


class BankQuestion(BaseModel):
    """One question as the exam checks see it."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    item_id: str
    # The ids of its options, in the order the bank prints them.
    option_ids: tuple[str, ...]
    keys: frozenset[str]
    objective_id: str


def read_question_bank(path: Path) -> dict[str, BankQuestion]:
    """Read a bank file into its questions by id, in file order; blank lines are skipped.

    A line that is not a question, a repeated id, or a key that is not among the options stops the reading: a
    reference with a hole would make a check pass or fail for the wrong reason.
    """
    questions: dict[str, BankQuestion] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise QuestionBankError(f"{path.name}: cannot read the file: {error.strerror}") from error
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            parsed = _Line.model_validate(json.loads(line))
        except (ValueError, ValidationError) as error:
            raise QuestionBankError(f"{path.name}, line {number}: not a bank question") from error
        option_ids = tuple(option.id for option in parsed.options)
        if parsed.id in questions:
            raise QuestionBankError(f"{path.name}, line {number}: the id {parsed.id!r} is used twice")
        if not set(parsed.keys) <= set(option_ids):
            raise QuestionBankError(f"{path.name}, line {number}: a key of {parsed.id!r} is not among its options")
        questions[parsed.id] = BankQuestion(
            item_id=parsed.id, option_ids=option_ids, keys=frozenset(parsed.keys), objective_id=parsed.objective_id
        )
    return questions
