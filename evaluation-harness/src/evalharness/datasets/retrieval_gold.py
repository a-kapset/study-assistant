"""The retrieval gold set of a course: every question of its question bank with the course section that answers it.

The questions are course material and stay on the local machine, so the gold set is never stored: it is built from
the local bank each time a run starts. A committed declaration file pins that bank by its SHA-256 and states, before
any run, what a run measures and which result is good enough.
"""

import hashlib
import json
import re
import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# Why a gold set could not be built; a caller can act on the code without parsing the message.
type GoldSetErrorCode = Literal["bank_changed", "invalid_bank_line", "objective_not_matched"]


class GoldSetError(ValueError):
    """The gold set cannot be built from this bank; says why with a code and what exactly is wrong."""

    def __init__(self, code: GoldSetErrorCode, detail: str) -> None:
        """Keep the code apart from the message, so tests and callers need not parse text."""
        super().__init__(f"{code}: {detail}")
        self.code = code


class _Strict(BaseModel):
    """Base for the gold set's records: an unknown or misspelt key is an error, not silently ignored."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PrimaryMetric(_Strict):
    """The main result of a run, fixed before the first run: a metric at k on some exam sets, and its target."""

    metric: Literal["recall"]
    k: int = Field(ge=1)
    sets: tuple[str, ...] = Field(min_length=1)
    target: float = Field(gt=0, le=1)


class GoldDeclaration(_Strict):
    """The committed part of a gold set: its bank, how a question finds its section, and what a run measures."""

    course_id: str = Field(min_length=1)
    bank_file: str = Field(min_length=1)
    bank_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    # Reads the answer section and its chapter from a question's learning objective id.
    objective_section: re.Pattern[str]
    held_out_sets: tuple[str, ...]
    # Times the held-out sets were looked at before run records existed; the runner adds the recorded uses to it.
    held_out_uses_before_records: int = Field(default=0, ge=0)

    primary: PrimaryMetric

    @model_validator(mode="after")
    def _check(self) -> Self:
        """Refuse a pattern without the two groups a case needs, and a main result measured on held-out questions."""
        missing = sorted({"section", "chapter"} - set(self.objective_section.groupindex))
        if missing:
            raise ValueError(f"objective_section has no group(s) {missing}")
        overlap = sorted(set(self.primary.sets) & set(self.held_out_sets))
        if overlap:
            raise ValueError(f"sets {overlap} are held out and cannot carry the main result")
        return self


class GoldCase(_Strict):
    """One question of the bank, the section that answers it, and the tags its results are reported by."""

    # The bank's item id, e.g. "A-12"; the part before the dash is the exam set.
    case_id: str
    # The question stem alone: the answer options often name other sections and would pull the search there.
    query: str
    gold_section: str
    exam_set: str
    chapter: str
    knowledge_level: str | None
    origin: Literal["sample_exam"]
    held_out: bool


class GoldSet(_Strict):
    """The cases built from one bank under one declaration, and a hash that names exactly this gold set."""

    declaration: GoldDeclaration
    cases: tuple[GoldCase, ...]
    # Two runs with the same value were scored against the same questions, sections and rules.
    sha256: str


class _BankLine(BaseModel):
    """The four fields of a bank question the gold set needs. Other fields of the line are ignored here and not
    checked: the full format of a bank belongs to the app's bank reader."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str = Field(pattern=r"^[^-\s]+-\S+$")
    stem: str = Field(min_length=1)
    objective_id: str
    knowledge_level: str | None = None


def load_declaration(path: Path) -> GoldDeclaration:
    """Read and check a gold set's declaration file (TOML)."""
    with path.open("rb") as file:
        return GoldDeclaration.model_validate(tomllib.load(file))


def build_gold_set(declaration: GoldDeclaration, sources_dir: Path) -> GoldSet:
    """Build the gold set from the bank in ``sources_dir``; a bank with another hash is refused."""
    content = (sources_dir / declaration.bank_file).read_bytes()
    actual = hashlib.sha256(content).hexdigest()
    if actual != declaration.bank_sha256:
        raise GoldSetError(
            "bank_changed",
            f"{declaration.bank_file} has SHA-256 {actual}, the declaration pins {declaration.bank_sha256}",
        )
    cases = tuple(
        _case(number, line, declaration)
        for number, line in enumerate(content.decode("utf-8").splitlines(), start=1)
        if line.strip()
    )
    named = {"declaration": declaration.model_dump(mode="json"), "cases": [case.model_dump() for case in cases]}
    digest = hashlib.sha256(json.dumps(named, sort_keys=True).encode("utf-8")).hexdigest()
    return GoldSet(declaration=declaration, cases=cases, sha256=digest)


def _case(number: int, line: str, declaration: GoldDeclaration) -> GoldCase:
    """Turn one bank line into a gold case; the line number goes into the error when the line cannot be used."""
    try:
        item = _BankLine.model_validate_json(line)
    except ValidationError as error:
        raise GoldSetError("invalid_bank_line", f"line {number}: {error}") from error
    match = declaration.objective_section.fullmatch(item.objective_id)
    if match is None:
        raise GoldSetError(
            "objective_not_matched",
            f"line {number}: {item.objective_id!r} does not match {declaration.objective_section.pattern!r}",
        )
    exam_set = item.id.split("-", 1)[0]
    return GoldCase(
        case_id=item.id,
        query=item.stem,
        gold_section=match["section"],
        exam_set=exam_set,
        chapter=match["chapter"],
        knowledge_level=item.knowledge_level,
        origin="sample_exam",
        held_out=exam_set in declaration.held_out_sets,
    )
