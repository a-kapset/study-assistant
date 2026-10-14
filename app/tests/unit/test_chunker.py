import hashlib
import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from study_assistant.courses.registry import load_course
from study_assistant.ingest.chunker import CHUNKER_REVISION, Chunk, ChunkSize, chunk_id, chunk_source, chunker_version
from study_assistant.ingest.normalize import normalize_document
from study_assistant.ingest.parser import parse_pdf
from study_assistant.ingest.structure import DocumentStructure, Objective, Section, SourceLine, extract_structure

# app/tests/unit/<this file> -> repository root.
REPO_COURSES = Path(__file__).resolve().parents[3] / "courses"

DEFAULT_SIZE = ChunkSize(size=1000, overlap=100)

# The chunks of the demo course at this size, reduced to one hash. Update both values together, and only on purpose:
# raise CHUNKER_REVISION in the chunker when the processing rules changed; keep it when only the demo source did.
GOLDEN_SIZE = ChunkSize(size=500, overlap=50)
GOLDEN_REVISION = 1
GOLDEN_FINGERPRINT = "94f43a4032ad6872eb6e8fad01260a938e7903c3312c9f1ee36c9ac1e3fffb4e"


def _demo_chunks(size: ChunkSize) -> tuple[Chunk, ...]:
    """Run the whole chain on the committed demo course and return its chunks."""
    course = load_course(REPO_COURSES, "demo")
    source = course.sources[0]
    document = normalize_document(parse_pdf(REPO_COURSES / "demo" / source.file), source.noise_lines)
    structure = extract_structure(document, course.structure)
    return chunk_source(structure, course_id=course.id, language=course.language, authority=source.authority, size=size)


def _structure(*sections: Section, objectives: tuple[Objective, ...] = ()) -> DocumentStructure:
    """Build a structure by hand, so a chunker test does not depend on how structure is found."""
    return DocumentStructure(path=Path("source.pdf"), sha256="a" * 64, sections=sections, objectives=objectives)


def _section(section_id: str | None, parent_id: str | None, *lines: tuple[int, str]) -> Section:
    """Build a section from (page, text) lines."""
    return Section(
        id=section_id,
        title="",
        parent_id=parent_id,
        lines=tuple(SourceLine(page=page, text=text) for page, text in lines),
    )


def _chunk(structure: DocumentStructure, size: ChunkSize = DEFAULT_SIZE) -> tuple[Chunk, ...]:
    """Chunk a hand-built structure with fixed course fields."""
    return chunk_source(structure, course_id="alpha", language="en", authority="primary", size=size)


@pytest.mark.covers("US-004", "AC-11")
def test_demo_course_chunks_carry_their_section_objectives_and_pages() -> None:
    """Guards the whole chain on the real demo PDF: sections, objectives, pages, the size cap and provenance."""
    chunks = _demo_chunks(DEFAULT_SIZE)
    source_hash = hashlib.sha256((REPO_COURSES / "demo" / "vardolin_rules.pdf").read_bytes()).hexdigest()

    assert chunks[0].section_id is None
    assert chunks[0].lo_ids == ()
    for chunk in chunks[1:]:
        assert chunk.lo_ids
        assert all(objective.startswith(f"OBJ-{int(chunk.section_id or 0):02d}-") for objective in chunk.lo_ids)
    assert all(len(chunk.text) <= 1000 for chunk in chunks)
    assert {page for chunk in chunks for page in (chunk.page_start, chunk.page_end)} == {1, 3, 4}
    assert len({chunk.id for chunk in chunks}) == len(chunks)
    assert {
        (chunk.course_id, chunk.language, chunk.source_authority, chunk.source_hash, chunk.chunker_version)
        for chunk in chunks
    } == {("demo", "en", "primary", source_hash, chunker_version(DEFAULT_SIZE))}


@pytest.mark.covers("US-004", "AC-12")
def test_demo_chunking_matches_its_recorded_fingerprint() -> None:
    """Guards the processing revision: changed chunks under an unchanged revision would look like the old corpus.

    Any change in cleaning, structure or splitting that changes the demo course's chunks fails here until the
    revision in the chunker is raised and the fingerprint above is updated.
    """
    chunks = _demo_chunks(GOLDEN_SIZE)
    content = [[c.section_id, c.page_start, c.page_end, list(c.lo_ids), c.text] for c in chunks]
    fingerprint = hashlib.sha256(json.dumps(content).encode("utf-8")).hexdigest()

    assert (CHUNKER_REVISION, fingerprint) == (GOLDEN_REVISION, GOLDEN_FINGERPRINT), (
        "The demo course is chunked differently now. If the processing rules changed, raise CHUNKER_REVISION; "
        "then set GOLDEN_REVISION and GOLDEN_FINGERPRINT in this test to the new values."
    )


@pytest.mark.covers("US-004", "AC-11")
def test_objectives_of_enclosing_sections_reach_nested_chunks() -> None:
    """Guards objective scope: a subsection inherits its parents' objectives, a sibling section does not."""
    structure = _structure(
        _section(None, None, (1, "Foreword")),
        _section("1", None, (1, "1 Basics")),
        _section("1.1", "1", (1, "1.1 Terms")),
        _section("1.1.1", "1.1", (2, "1.1.1 Words")),
        _section("1.2", "1", (2, "1.2 Errors")),
        objectives=(
            Objective(id="LO-0", section_id=None, knowledge_level=None, line=SourceLine(page=1, text="LO-0")),
            Objective(id="LO-1", section_id="1", knowledge_level=None, line=SourceLine(page=1, text="LO-1")),
            Objective(id="LO-11", section_id="1.1", knowledge_level=None, line=SourceLine(page=1, text="LO-11")),
        ),
    )

    chunks = _chunk(structure)

    assert [(chunk.section_id, chunk.lo_ids) for chunk in chunks] == [
        (None, ("LO-0",)),
        ("1", ("LO-1",)),
        ("1.1", ("LO-1", "LO-11")),
        ("1.1.1", ("LO-1", "LO-11")),
        ("1.2", ("LO-1",)),
    ]


@pytest.mark.covers("US-004", "AC-11")
def test_a_chunk_over_a_page_break_names_both_pages() -> None:
    """Guards citations: the first and the last page of a chunk come from the lines its text starts and ends on."""
    structure = _structure(_section("1", None, (1, "1 Basics"), (1, "end of page one"), (2, "start of page two")))

    (chunk,) = _chunk(structure)

    assert (chunk.page_start, chunk.page_end) == (1, 2)


@pytest.mark.covers("US-004", "AC-12")
def test_chunk_id_changes_with_every_field_and_fields_cannot_run_together() -> None:
    """Guards id uniqueness: each field must count, and "1" + "11" must not equal "11" + "1" once joined."""
    base = chunk_id("alpha", "a" * 64, "1", 11, "text")
    variants = [
        chunk_id("beta", "a" * 64, "1", 11, "text"),
        chunk_id("alpha", "b" * 64, "1", 11, "text"),
        chunk_id("alpha", "a" * 64, "2", 11, "text"),
        chunk_id("alpha", "a" * 64, "1", 12, "text"),
        chunk_id("alpha", "a" * 64, "1", 11, "other"),
        chunk_id("alpha", "a" * 64, "11", 1, "text"),
        chunk_id("alpha", "a" * 64, None, 11, "text"),
    ]

    assert base not in variants
    assert len(set(variants)) == len(variants)
    assert base == chunk_id("alpha", "a" * 64, "1", 11, "text")


@pytest.mark.parametrize(("size", "overlap"), [(100, 100), (100, 150), (100, -1)])
def test_overlap_must_be_shorter_than_a_chunk(size: int, overlap: int) -> None:
    """Guards the size settings: an overlap as long as a chunk is refused with a message naming both values."""
    with pytest.raises(ValueError, match=rf"below the size {size}, got {overlap}"):
        ChunkSize(size=size, overlap=overlap)


def test_chunker_version_names_the_revision_the_sizes_and_the_splitter() -> None:
    """Guards the version string: two corpora made with different settings must not carry the same version."""
    text = chunker_version(ChunkSize(size=800, overlap=80))

    assert f"/{CHUNKER_REVISION};" in text
    assert "size=800;overlap=80" in text
    assert f"langchain-text-splitters={version('langchain-text-splitters')}" in text
    assert text != chunker_version(ChunkSize(size=800, overlap=40))


@dataclass(frozen=True)
class _Generated:
    """A generated structure whose words are all different, so every word marks one place in the text."""

    structure: DocumentStructure
    # For every word: its section id and the page of its line.
    words: dict[str, tuple[str | None, int]]


@st.composite
def _generated(draw: st.DrawFn) -> _Generated:
    """Sections of numbered words spread over lines, empty lines and pages; no passage repeats anywhere."""
    counter = 0
    page = 1
    words: dict[str, tuple[str | None, int]] = {}
    sections: list[Section] = []
    for number in range(draw(st.integers(min_value=1, max_value=4))):
        section_id = None if number == 0 and draw(st.booleans()) else str(number + 1)
        lines: list[SourceLine] = []
        for index in range(draw(st.integers(min_value=1, max_value=12))):
            if draw(st.integers(min_value=0, max_value=4)) == 0:
                page += 1
            if index > 0 and draw(st.integers(min_value=0, max_value=5)) == 0:
                lines.append(SourceLine(page=page, text=""))
                continue
            line_words = [f"w{counter + offset}" for offset in range(draw(st.integers(min_value=1, max_value=9)))]
            counter += len(line_words)
            words.update((word, (section_id, page)) for word in line_words)
            lines.append(SourceLine(page=page, text=" ".join(line_words)))
        sections.append(Section(id=section_id, title="", parent_id=None, lines=tuple(lines)))
    return _Generated(_structure(*sections), words)


_SIZES = st.integers(min_value=20, max_value=120).flatmap(
    lambda size: st.builds(ChunkSize, size=st.just(size), overlap=st.integers(min_value=0, max_value=size // 2))
)


@pytest.mark.covers("US-004", "AC-11")
@settings(max_examples=200, derandomize=True, deadline=None)
@given(_generated(), _SIZES)
def test_chunks_cover_each_section_in_order_within_the_cap_and_cite_true_pages(
    generated: _Generated, size: ChunkSize
) -> None:
    """Guards the splitting and page rules on text without repeated passages, where offsets are unambiguous.

    No chunk is longer than the cap; every word lands in a chunk of its own section; chunks follow the text in
    order; each chunk's first and last pages are the pages of its first and last words; ids are unique and the
    same on a second run.
    """
    chunks = _chunk(generated.structure, size)

    seen: set[str] = set()
    previous_first = -1
    previous_section: str | None = "start"
    for chunk in chunks:
        tokens = chunk.text.split()
        assert len(chunk.text) <= size.size
        assert all(generated.words[token][0] == chunk.section_id for token in tokens)
        first = int(tokens[0][1:])
        if chunk.section_id == previous_section:
            assert first > previous_first
        previous_first, previous_section = first, chunk.section_id
        assert (chunk.page_start, chunk.page_end) == (generated.words[tokens[0]][1], generated.words[tokens[-1]][1])
        seen.update(tokens)
    assert seen == set(generated.words)
    assert len({chunk.id for chunk in chunks}) == len(chunks)
    assert [chunk.id for chunk in chunks] == [chunk.id for chunk in _chunk(generated.structure, size)]
