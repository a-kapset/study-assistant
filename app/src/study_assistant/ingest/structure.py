"""Finding the structure of a course source in its cleaned text: its sections and its learning objectives.

The course manifest says what a section heading, an objective id and a knowledge level look like (regular
expressions with named groups); this module only applies them and holds no course-specific text. Every line
of the source ends up in exactly one section, so the text can later be split section by section without
losing any of it.

A heading pattern alone also matches lines that only look like headings: numbered list items ("1. Testing
shows the presence of defects") or an entry repeated in a list of contents. When the section ids are numbers
("2", "4.2.1"), real headings follow each other in increasing order, so the longest run of headings whose
numbers strictly increase is kept and the rest are treated as ordinary text.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from study_assistant.courses.models import StructureRules
from study_assistant.ingest.normalize import NormalizedDocument

# A numbered section id: "2", "4.2", "4.2.1".
_NUMBERED_ID = re.compile(r"\d+(?:\.\d+)*")


class CourseStructureError(ValueError):
    """The course's structure rules do not fit its source text; the message names the file and the problem."""


@dataclass(frozen=True)
class SourceLine:
    """One line of cleaned text and the number of the page it is on."""

    page: int
    text: str


@dataclass(frozen=True)
class Section:
    """A section with its lines, heading line included; the text before the first heading has no id."""

    id: str | None
    title: str
    # The nearest enclosing section ("4.2" for "4.2.1"), when the ids are numbered and that section exists.
    parent_id: str | None
    lines: tuple[SourceLine, ...]


@dataclass(frozen=True)
class Objective:
    """A learning objective, defined by the line that starts with its id."""

    id: str
    section_id: str | None
    knowledge_level: str | None
    line: SourceLine


@dataclass(frozen=True)
class DocumentStructure:
    """The sections of one source in reading order, and the objectives it defines."""

    path: Path
    # SHA-256 of the source file the structure was found in.
    sha256: str
    sections: tuple[Section, ...]
    objectives: tuple[Objective, ...]


def extract_structure(document: NormalizedDocument, rules: StructureRules) -> DocumentStructure:
    """Split a cleaned source into sections by the course's heading rule, then find its objectives.

    Pages without text are skipped; their numbers are simply absent from the lines.
    """
    lines = [
        SourceLine(page=page.number, text=text)
        for page in document.pages
        if page.status == "ok"
        for text in page.text.split("\n")
    ]
    sections = split_into_sections(lines, rules.section_heading)
    objectives = find_objectives(sections, rules, document.path)
    return DocumentStructure(path=document.path, sha256=document.sha256, sections=sections, objectives=objectives)


def split_into_sections(lines: Sequence[SourceLine], heading: re.Pattern[str]) -> tuple[Section, ...]:
    """Cut the lines at every accepted heading; lines before the first heading form a section without id."""
    candidates = [(index, match) for index, line in enumerate(lines) if (match := heading.fullmatch(line.text))]
    accepted = [candidates[kept] for kept in accepted_headings([match["id"] for _, match in candidates])]

    starts = [index for index, _ in accepted]
    if not starts or starts[0] > 0:
        starts.insert(0, 0)
    headings = dict(accepted)

    sections: list[Section] = []
    known_ids: set[str] = set()
    for start, end in zip(starts, [*starts[1:], len(lines)], strict=True):
        if start == end:
            continue
        match = headings.get(start)
        section_id = match["id"] if match else None
        sections.append(
            Section(
                id=section_id,
                title=match["title"] if match else "",
                parent_id=_parent_id(section_id, known_ids),
                lines=tuple(lines[start:end]),
            )
        )
        if section_id is not None:
            known_ids.add(section_id)
    return tuple(sections)


def accepted_headings(ids: Sequence[str]) -> list[int]:
    """Choose which heading candidates are real headings; returns their positions in ``ids``, in order.

    Numbered ids: the longest run of candidates whose numbers strictly increase ("4.2" < "4.2.1" < "4.3" < "5"),
    the earliest such run when several are equally long. Any other ids: the first occurrence of each id.
    """
    if not all(_NUMBERED_ID.fullmatch(section_id) for section_id in ids):
        first: dict[str, int] = {}
        for position, section_id in enumerate(ids):
            first.setdefault(section_id, position)
        return sorted(first.values())

    keys = [tuple(int(part) for part in section_id.split(".")) for section_id in ids]
    # run_length[j]: length of the longest increasing run that ends at candidate j; previous[j]: the candidate before.
    run_length = [1] * len(keys)
    previous = [-1] * len(keys)
    for j in range(len(keys)):
        for i in range(j):
            if keys[i] < keys[j] and run_length[i] + 1 > run_length[j]:
                run_length[j] = run_length[i] + 1
                previous[j] = i
    if not keys:
        return []
    position = max(range(len(keys)), key=lambda j: (run_length[j], -j))
    run: list[int] = []
    while position != -1:
        run.append(position)
        position = previous[position]
    return run[::-1]


def find_objectives(sections: Sequence[Section], rules: StructureRules, path: Path) -> tuple[Objective, ...]:
    """Find the objectives defined in the sections: a line that starts with an objective id defines it.

    Later mentions of the same id (a list of changes, a cross-reference) are not definitions. The objective
    belongs to the section named by the pattern's "section" group when it has one, otherwise to the section
    whose text defines it.
    """
    if rules.objective_id is None:
        return ()
    section_ids = {section.id for section in sections}
    objectives: dict[str, Objective] = {}
    for section in sections:
        for line in section.lines:
            match = rules.objective_id.match(line.text)
            if match is None or match[0] in objectives:
                continue
            section_id = match.groupdict().get("section") or section.id
            if section_id not in section_ids:
                raise CourseStructureError(
                    f"{path}: objective {match[0]} on page {line.page} belongs to section {section_id}, "
                    "which no heading in the file defines"
                )
            objectives[match[0]] = Objective(
                id=match[0],
                section_id=section_id,
                knowledge_level=_knowledge_level(line.text, rules.knowledge_level),
                line=line,
            )
    return tuple(objectives.values())


def _knowledge_level(text: str, pattern: re.Pattern[str] | None) -> str | None:
    """Read the knowledge level from an objective's line, if the course grades its objectives."""
    if pattern is None:
        return None
    match = pattern.search(text)
    return match["level"] if match else None


def _parent_id(section_id: str | None, known_ids: set[str]) -> str | None:
    """Find the nearest already seen section whose numbered id is a prefix of this one ("4" for "4.2.1")."""
    if section_id is None or not _NUMBERED_ID.fullmatch(section_id):
        return None
    parts = section_id.split(".")
    for length in range(len(parts) - 1, 0, -1):
        candidate = ".".join(parts[:length])
        if candidate in known_ids:
            return candidate
    return None
