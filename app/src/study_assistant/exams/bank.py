"""A course's question bank in the common format: one JSON object per line, one question per object.

Every course writes its bank in this format, whatever its exams look like, so the code that reads it knows no
course. A bank is checked as a whole when it is read: an invalid line stops the reading and names its line.
"""

import json
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# Why a bank file was refused; the caller can act on the code without parsing the message.
type BankErrorCode = Literal["invalid_json", "invalid_item", "duplicate_item_id"]


class BankError(ValueError):
    """A question bank line cannot be used; says which line, why (a code) and what exactly is wrong."""

    def __init__(self, path: Path, line: int, code: BankErrorCode, detail: str) -> None:
        """Keep the line and the code apart from the message, so tests and callers need not parse text."""
        super().__init__(f"{path.name}, line {line}: {code}: {detail}")
        self.line = line
        self.code = code


class BankModel(BaseModel):
    """Base for bank records: an unknown or misspelt key is an error, not silently ignored."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class BankOption(BankModel):
    """One answer option of a question, e.g. id "b" with its text."""

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class BankSource(BankModel):
    """Where a question comes from, so a person can find it in the original document."""

    document: str = Field(min_length=1)
    version: str = Field(min_length=1)
    # The question's number as the document prints it, e.g. "12" or "A7".
    question: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)


class BankItem(BankModel):
    """One multiple-choice question with its answer key and the learning objective it tests."""

    id: str = Field(min_length=1)
    stem: str = Field(min_length=1)
    options: tuple[BankOption, ...] = Field(min_length=2)
    # One or more option ids; a question with several keys asks to select all of them.
    keys: tuple[str, ...] = Field(min_length=1)
    objective_id: str = Field(min_length=1)
    knowledge_level: str | None = None
    source: BankSource

    @model_validator(mode="after")
    def _check_keys(self) -> Self:
        """Refuse a question whose key could never be answered: option ids repeat, or a key is not an option."""
        ids = [option.id for option in self.options]
        if len(set(ids)) != len(ids):
            raise ValueError(f"option ids repeat: {ids}")
        if len(set(self.keys)) != len(self.keys):
            raise ValueError(f"keys repeat: {list(self.keys)}")
        missing = [key for key in self.keys if key not in ids]
        if missing:
            raise ValueError(f"keys {missing} are not among the option ids {ids}")
        return self


def read_bank(path: Path) -> tuple[BankItem, ...]:
    """Read and check a whole bank file; the first unusable line raises ``BankError`` naming it.

    Blank lines are skipped. Item ids must be unique in the file, because answers and results refer to them.
    """
    items: list[BankItem] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as lines:
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            item = _parse_line(path, number, line)
            if item.id in seen:
                raise BankError(path, number, "duplicate_item_id", f"item id {item.id!r} is used twice")
            seen.add(item.id)
            items.append(item)
    return tuple(items)


def _parse_line(path: Path, number: int, line: str) -> BankItem:
    """Turn one line into an item; broken JSON and a wrong record get different codes."""
    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        raise BankError(path, number, "invalid_json", exc.msg) from exc
    try:
        return BankItem.model_validate(data)
    except ValidationError as exc:
        # Only where and what; the question text itself is left out of the message.
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'item'}: {error['msg']}"
            for error in exc.errors(include_input=False, include_url=False)
        )
        raise BankError(path, number, "invalid_item", problems) from exc
