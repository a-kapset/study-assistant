"""The Q&A set of a course: questions a learner could ask, with the course section that answers each one, and
questions the course does not answer.

The questions are written for the set, not taken from course materials, so the file is committed. It also states,
before any run, which results count as good enough.
"""

import tomllib
from collections import Counter
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    """Base for the set's records: an unknown or misspelt key is an error, not silently ignored."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class QaTargets(_Strict):
    """The results a run must reach, fixed before the first run."""

    # Share of answerable questions the app answers instead of declining.
    answered_min: float = Field(gt=0, le=1)
    # Share of answered answerable questions whose citations include the gold section.
    gold_cited_min: float = Field(gt=0, le=1)
    # Unanswerable questions the app may answer instead of declining; an answer without support is a blocker.
    unsupported_answers_max: int = Field(ge=0)


class QaCase(_Strict):
    """One question; without a gold section the course does not answer it and the app must decline."""

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    # The section that answers the question; a citation of it or of one of its subsections counts.
    gold_section: str | None = None


class QaSet(_Strict):
    """The course the questions are about, the targets of a run, and the questions."""

    course_id: str = Field(min_length=1)
    targets: QaTargets
    cases: tuple[QaCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_ids(self) -> Self:
        """Refuse two cases with the same id: their results would be mixed up in a run record."""
        repeated = sorted(case_id for case_id, n in Counter(case.id for case in self.cases).items() if n > 1)
        if repeated:
            raise ValueError(f"case ids used more than once: {repeated}")
        return self


def load_qa_set(path: Path) -> QaSet:
    """Read and check a Q&A set file (TOML)."""
    with path.open("rb") as file:
        return QaSet.model_validate(tomllib.load(file))
