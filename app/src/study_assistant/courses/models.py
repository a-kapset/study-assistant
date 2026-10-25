"""The course manifest: metadata and rules for one course, read from courses/<course_id>/course.yaml.

Everything course-specific reaches the app through these models, so the app code holds no course names.
"""

import re
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

# A course id becomes a directory name and part of stored records, so it is limited to safe characters.
COURSE_ID_PATTERN = r"^[a-z0-9_]+$"

# Two-letter language code of the course material, e.g. "en".
COURSE_LANGUAGE_PATTERN = r"^[a-z]{2}$"


# What a source file is for: course text, term definitions, or exam questions in the common format.
type SourceRole = Literal["material", "glossary", "question_bank"]

# Primary sources define what is correct for the course; supplementary ones add background but never override them.
type SourceAuthority = Literal["primary", "supplementary"]


class ManifestModel(BaseModel):
    """Base for manifest parts: an unknown or misspelt key is an error, not silently ignored."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class StructureRules(ManifestModel):
    """How to find the course structure in its source text; each value is a regular expression.

    Invalid expressions are rejected when the manifest is loaded, not when the sources are first read.
    """

    # Matched against a whole line. Named groups: "id" (e.g. "2" or "4.2.1") and "title".
    section_heading: re.Pattern[str]
    # A course built from ordinary books may have no learning objectives; then only its sections are found.
    # A line that starts with a match defines an objective. An optional named group "section" gives the id of the
    # section the objective belongs to, for a course that lists its objectives apart from the sections' text.
    objective_id: re.Pattern[str] | None = None
    # Not every course grades its objectives by knowledge level. Searched in the objective's line; named group "level".
    knowledge_level: re.Pattern[str] | None = None

    @model_validator(mode="after")
    def _check_groups(self) -> Self:
        """Reject rules that structure extraction could not apply, before any source is read."""
        _require_groups("section_heading", self.section_heading, {"id", "title"})
        if self.knowledge_level is not None:
            if self.objective_id is None:
                raise ValueError("knowledge_level is given, but there is no objective_id to attach the levels to")
            _require_groups("knowledge_level", self.knowledge_level, {"level"})
        return self


def _require_groups(field: str, pattern: re.Pattern[str], groups: set[str]) -> None:
    """Fail when a pattern lacks a named group whose value structure extraction reads from each match."""
    missing = sorted(groups - pattern.groupindex.keys())
    if missing:
        raise ValueError(f"{field} needs the named group(s) {', '.join(missing)}, written as (?P<name>...)")


class CourseSource(ManifestModel):
    """One source file of a course; the file itself stays outside the repository."""

    # Path relative to the course's local source directory.
    file: str = Field(min_length=1)
    role: SourceRole
    authority: SourceAuthority
    # Whole lines the page layout repeats (running header, footer, page number); text normalization drops a
    # line only when one of these matches all of it. Each file has its own layout, so the rules are per source.
    noise_lines: tuple[re.Pattern[str], ...] = ()
    # The sections whose text is searched, matched against the whole section id. The text of other sections (front
    # matter, references, appendices, an index, a chapter's list of objectives) stays in the document's structure
    # but gets no chunks, so it cannot push teaching text out of the results; text before the first heading has no
    # id and is never matched. Without a pattern every section is searched.
    content_sections: re.Pattern[str] | None = None


class CourseManifest(ManifestModel):
    """One course: what it is called, how its sources are structured, and which files they are."""

    id: str = Field(pattern=COURSE_ID_PATTERN)
    title: str = Field(min_length=1)
    # A short description of what the course covers.
    description: str = Field(min_length=1)
    language: str = Field(pattern=COURSE_LANGUAGE_PATTERN)
    structure: StructureRules
    sources: tuple[CourseSource, ...] = Field(min_length=1)
