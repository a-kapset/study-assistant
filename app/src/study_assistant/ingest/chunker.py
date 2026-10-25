"""Splitting a course source into chunks for retrieval, one section at a time.

A chunk never crosses a section boundary, so it is about one topic and carries that section's id and the
learning objectives that apply to it. Inside a section the text is cut to a maximum size with some overlap
by LangChain's recursive splitter, which prefers to cut between paragraphs, then lines, then words.

Every chunk records which rules produced it (the revision of the text-processing rules, the size and overlap,
the splitter version). Re-chunking with other rules changes the corpus, so chunks made by different rules must
be told apart.
"""

import hashlib
import re
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version

from langchain_text_splitters import RecursiveCharacterTextSplitter

from study_assistant.courses.models import SourceAuthority
from study_assistant.ingest.structure import DocumentStructure, Objective, Section

CHUNKER_NAME = "sections-recursive"
# Revision of the rules that turn a source into chunks: text cleaning, structure and splitting. Raise it with any
# change in that chain that changes the chunks; a test on the demo course fails until it is raised.
CHUNKER_REVISION = 1


@dataclass(frozen=True)
class ChunkSize:
    """How long a chunk may be and how much consecutive chunks of a section share, in characters."""

    size: int
    overlap: int

    def __post_init__(self) -> None:
        """Reject sizes the splitter cannot honour: an overlap must be shorter than a chunk."""
        if not 0 <= self.overlap < self.size:
            raise ValueError(f"chunk overlap must be at least 0 and below the size {self.size}, got {self.overlap}")


@dataclass(frozen=True)
class Chunk:
    """A piece of a section's text with everything retrieval, citations and evaluation need to know about it."""

    id: str
    text: str
    course_id: str
    section_id: str | None
    # The objectives of the chunk's section and of the sections that enclose it.
    lo_ids: tuple[str, ...]
    page_start: int
    page_end: int
    language: str
    source_hash: str
    source_authority: SourceAuthority
    chunker_version: str


def chunker_version(size: ChunkSize) -> str:
    """Name the rules that produce chunks: the text-processing revision, the size settings and the splitter version."""
    return (
        f"{CHUNKER_NAME}/{CHUNKER_REVISION};size={size.size};overlap={size.overlap};"
        f"langchain-text-splitters={version('langchain-text-splitters')}"
    )


def chunk_source(
    structure: DocumentStructure,
    *,
    course_id: str,
    language: str,
    authority: SourceAuthority,
    size: ChunkSize,
    content_sections: re.Pattern[str] | None = None,
) -> tuple[Chunk, ...]:
    """Split the sections of one source into chunks, in reading order.

    With ``content_sections`` only the sections whose id matches it as a whole are chunked; without it, all are.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=size.size, chunk_overlap=size.overlap, add_start_index=True)
    rules = chunker_version(size)
    sections = {section.id: section for section in structure.sections}
    chunks: list[Chunk] = []
    for section in structure.sections:
        if content_sections is not None and (section.id is None or content_sections.fullmatch(section.id) is None):
            continue
        text = "\n".join(line.text for line in section.lines)
        line_starts = _line_starts(section)
        lo_ids = objectives_in_scope(section, sections, structure.objectives)
        for ordinal, piece in enumerate(splitter.create_documents([text])):
            start = int(piece.metadata["start_index"])
            end = start + len(piece.page_content) - 1
            chunks.append(
                Chunk(
                    id=chunk_id(course_id, structure.sha256, section.id, ordinal, piece.page_content),
                    text=piece.page_content,
                    course_id=course_id,
                    section_id=section.id,
                    lo_ids=lo_ids,
                    page_start=_page_at(section, line_starts, start),
                    page_end=_page_at(section, line_starts, end),
                    language=language,
                    source_hash=structure.sha256,
                    source_authority=authority,
                    chunker_version=rules,
                )
            )
    return tuple(chunks)


def chunk_id(course_id: str, source_hash: str, section_id: str | None, ordinal: int, text: str) -> str:
    """Derive a chunk's id from what it is: the same course, file, section, position and text give the same id.

    The fields are joined with a separator that does not occur in them, so two different field lists never
    give the hash the same input.
    """
    fields = (course_id, source_hash, section_id or "", str(ordinal), text)
    return hashlib.sha256("\x1f".join(fields).encode("utf-8")).hexdigest()


def objectives_in_scope(
    section: Section, sections: Mapping[str | None, Section], objectives: Sequence[Objective]
) -> tuple[str, ...]:
    """List the objectives of a section and of every section that encloses it, in the order they are defined."""
    scope: set[str | None] = {section.id}
    parent = section.parent_id
    while parent is not None:
        scope.add(parent)
        parent = sections[parent].parent_id
    return tuple(objective.id for objective in objectives if objective.section_id in scope)


def _line_starts(section: Section) -> list[int]:
    """Offsets at which each line of the section starts in its joined text (lines are joined by one newline)."""
    starts: list[int] = []
    offset = 0
    for line in section.lines:
        starts.append(offset)
        offset += len(line.text) + 1
    return starts


def _page_at(section: Section, line_starts: Sequence[int], offset: int) -> int:
    """Return the page of the line that holds the character at ``offset`` in the section's joined text."""
    return section.lines[bisect_right(line_starts, offset) - 1].page
