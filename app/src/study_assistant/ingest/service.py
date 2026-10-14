"""Ingesting a course: its source files become documents, sections, objectives and chunks in the database.

Every source is read and chunked before anything is written, so a source that cannot be read stops the run
with the database as it was. The writing then happens in one transaction per course:

- a document whose file and rules are unchanged is kept as it is;
- a changed one is deleted with all its sections, objectives and chunks, then written again, so no chunk of
  its old edition stays searchable;
- a document the manifest no longer lists is deleted the same way, so search stops finding text that is no
  longer part of the course.

Running it twice on the same files therefore changes nothing. Only the course being ingested is touched:
another course's rows stay as they are, even where both courses have a file of the same name.
"""

import asyncio
import hashlib
import json
import re
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from study_assistant.courses.models import CourseManifest, CourseSource
from study_assistant.ingest.chunker import Chunk, ChunkSize, chunk_source, chunker_version
from study_assistant.ingest.normalize import normalize_document
from study_assistant.ingest.parser import ParserVersion, SourceParseError, parse_pdf
from study_assistant.ingest.structure import CourseStructureError, DocumentStructure, extract_structure

# Roles whose files are course text; a question bank has its own format and is loaded elsewhere.
_TEXT_ROLES = frozenset({"material", "glossary"})

# What happened to a document listed in the manifest.
type DocumentOutcome = Literal["added", "replaced", "unchanged"]


class IngestError(Exception):
    """A course source cannot be ingested; the message names the file and why. Nothing has been written."""


@dataclass(frozen=True)
class Edition:
    """What a document's rows were made from: equal editions give equal rows, so the document can be kept."""

    sha256: str
    chunker_version: str
    parser: str
    parser_version: str
    engine_version: str
    rules_sha256: str


@dataclass(frozen=True)
class PreparedDocument:
    """One source file read, structured and chunked, ready to be written."""

    source: CourseSource
    edition: Edition
    page_count: int
    structure: DocumentStructure
    chunks: tuple[Chunk, ...]


class CorpusCounts(BaseModel):
    """How much of a course is stored, counted in the database."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    documents: int
    # Sections with a heading; text before a file's first heading is stored but not counted as a section.
    sections: int
    learning_objectives: int
    chunks: int


class DocumentReport(BaseModel):
    """One source file of the manifest and what ingestion did with it."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source_file: str
    outcome: DocumentOutcome


class IngestReport(BaseModel):
    """The result of ingesting one course: what happened to each file and what the database holds afterwards."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    course_id: str
    documents: list[DocumentReport]
    # Documents stored before but no longer listed in the manifest; their rows are deleted.
    removed: list[str]
    corpus: CorpusCounts


class CourseWriter(Protocol):
    """Writes of one course inside one transaction; other courses' rows are never touched."""

    async def editions(self) -> dict[str, Edition]:
        """Return the stored documents of the course, keyed by source file."""
        ...

    async def save_course(self, title: str, language: str) -> None:
        """Create the course's row, or update its title and language."""
        ...

    async def delete_documents(self, source_files: Sequence[str]) -> None:
        """Delete these documents of the course with all their rows."""
        ...

    async def insert_document(self, document: PreparedDocument) -> None:
        """Store a document with its sections, objectives and chunks."""
        ...

    async def counts(self) -> CorpusCounts:
        """Count the course's stored rows, as the transaction sees them."""
        ...


class CorpusStore(Protocol):
    """Where course corpora are stored."""

    def course_writer(self, course_id: str) -> AbstractAsyncContextManager[CourseWriter]:
        """Open one transaction for writing a course; it commits when the block ends without an error.

        Two ingestions of the same course wait for each other instead of interleaving.
        """
        ...

    async def counts(self, course_id: str) -> CorpusCounts:
        """Count a course's stored rows; all zero for a course never ingested."""
        ...


class CourseIngestor:
    """Ingests the courses the app serves from their source files into a corpus store."""

    def __init__(self, store: CorpusStore, source_dir: Path, size: ChunkSize) -> None:
        """Keep where sources are and how to chunk them; nothing is read until a course is ingested."""
        self._store = store
        self._source_dir = source_dir
        self._size = size

    async def ingest(self, manifest: CourseManifest) -> IngestReport:
        """Read every source of the course, then bring its stored corpus in line with them in one transaction."""
        # Parsing is CPU work; a worker thread keeps the server answering other requests meanwhile.
        prepared = await asyncio.to_thread(prepare_course, manifest, self._source_dir / manifest.id, self._size)
        async with self._store.course_writer(manifest.id) as writer:
            stored = await writer.editions()
            outcomes = {
                document.source.file: outcome(stored.get(document.source.file), document.edition)
                for document in prepared
            }
            removed = sorted(stored.keys() - outcomes.keys())
            await writer.save_course(manifest.title, manifest.language)
            replaced = [file for file, result in outcomes.items() if result == "replaced"]
            await writer.delete_documents([*removed, *replaced])
            for document in prepared:
                if outcomes[document.source.file] != "unchanged":
                    await writer.insert_document(document)
            corpus = await writer.counts()
        return IngestReport(
            course_id=manifest.id,
            documents=[DocumentReport(source_file=file, outcome=result) for file, result in outcomes.items()],
            removed=removed,
            corpus=corpus,
        )


def outcome(stored: Edition | None, current: Edition) -> DocumentOutcome:
    """Decide what to do with a document: add it if it is new, keep it if nothing it was made from changed."""
    if stored is None:
        return "added"
    return "unchanged" if stored == current else "replaced"


def prepare_course(manifest: CourseManifest, course_dir: Path, size: ChunkSize) -> tuple[PreparedDocument, ...]:
    """Read, structure and chunk every text source of a course; the first source that fails stops it."""
    return tuple(
        prepare_document(manifest, source, course_dir, size)
        for source in manifest.sources
        if source.role in _TEXT_ROLES
    )


def prepare_document(
    manifest: CourseManifest, source: CourseSource, course_dir: Path, size: ChunkSize
) -> PreparedDocument:
    """Turn one source file into a document ready to store; an unreadable or unsupported file is an ``IngestError``."""
    path = course_dir / source.file
    if path.suffix.lower() != ".pdf":
        raise IngestError(f"{source.file}: only PDF sources can be ingested so far")
    try:
        parsed = parse_pdf(path)
        structure = extract_structure(normalize_document(parsed, source.noise_lines), manifest.structure)
    except OSError as exc:
        raise IngestError(f"{source.file}: cannot read the file: {exc.strerror}") from exc
    except (SourceParseError, CourseStructureError) as exc:
        raise IngestError(f"{source.file}: {exc}") from exc
    chunks = chunk_source(
        structure, course_id=manifest.id, language=manifest.language, authority=source.authority, size=size
    )
    return PreparedDocument(
        source=source,
        edition=_edition(parsed.sha256, parsed.parser, chunker_version(size), rules_hash(manifest, source)),
        page_count=len(parsed.pages),
        structure=structure,
        chunks=chunks,
    )


def rules_hash(manifest: CourseManifest, source: CourseSource) -> str:
    """Hash the manifest values that shape a source's rows, so editing any of them rebuilds the document."""
    rules = manifest.structure
    shaping = {
        "language": manifest.language,
        "role": source.role,
        "authority": source.authority,
        "noise_lines": [_pattern(line) for line in source.noise_lines],
        "section_heading": _pattern(rules.section_heading),
        "objective_id": _pattern(rules.objective_id) if rules.objective_id else None,
        "knowledge_level": _pattern(rules.knowledge_level) if rules.knowledge_level else None,
    }
    return hashlib.sha256(json.dumps(shaping, sort_keys=True).encode("utf-8")).hexdigest()


def _pattern(pattern: re.Pattern[str]) -> list[object]:
    """A regular expression as plain data: its text and its flags, both of which change what it matches."""
    return [pattern.pattern, int(pattern.flags)]


def _edition(sha256: str, parser: ParserVersion, chunker: str, rules_sha256: str) -> Edition:
    """Collect what a document's rows are made from into one comparable value."""
    return Edition(
        sha256=sha256,
        chunker_version=chunker,
        parser=parser.name,
        parser_version=parser.version,
        engine_version=parser.engine_version,
        rules_sha256=rules_sha256,
    )
