from dataclasses import dataclass
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from study_assistant.courses.models import StructureRules
from study_assistant.courses.registry import load_course
from study_assistant.ingest.normalize import CleanPage, NormalizedDocument, normalize_document
from study_assistant.ingest.parser import ParserVersion, parse_pdf
from study_assistant.ingest.structure import (
    CourseStructureError,
    DocumentStructure,
    accepted_headings,
    extract_structure,
)

# app/tests/unit/<this file> -> repository root.
REPO_COURSES = Path(__file__).resolve().parents[3] / "courses"

PARSER = ParserVersion(name="test", version="0", engine_version="0")

# A course with flat numbered units whose objectives are listed inside them.
UNITS = StructureRules.model_validate(
    {"section_heading": r"^Unit (?P<id>\d+)\. (?P<title>.+)$", "objective_id": r"OBJ-\d{2}-\d{2}"}
)

# A course with dotted section numbers, objectives listed apart from their sections, and knowledge levels.
DOTTED = StructureRules.model_validate(
    {
        "section_heading": r"^(?P<id>\d+(?:\.\d+)*) (?P<title>[A-Z].*)$",
        "objective_id": r"LO-(?P<section>\d+\.\d+)\.\d+",
        "knowledge_level": r"\[(?P<level>L\d)\]",
    }
)

# A course built from ordinary books: dotted chapter numbers and no objectives.
BOOK = StructureRules.model_validate({"section_heading": r"^(?P<id>\d+(?:\.\d+)*) (?P<title>[A-Z].*)$"})


def _document(*pages: str) -> NormalizedDocument:
    """Build a cleaned document from page texts; an empty text is a page without a text layer."""
    return NormalizedDocument(
        path=Path("source.pdf"),
        sha256="0" * 64,
        parser=PARSER,
        pages=tuple(
            CleanPage(number=number, text=text, status="ok" if text else "no_text_layer")
            for number, text in enumerate(pages, start=1)
        ),
    )


def _outline(structure: DocumentStructure) -> list[tuple[str | None, str, str | None]]:
    """Reduce a structure to (id, title, parent id) per section, for comparing with an expected outline."""
    return [(section.id, section.title, section.parent_id) for section in structure.sections]


@pytest.mark.covers("US-004", "AC-13", "AC-15", "AC-16")
def test_demo_course_structure_is_found_in_its_pdf() -> None:
    """Guards the whole path on real engine output: four units with their titles and every objective in its unit."""
    course = load_course(REPO_COURSES, "demo")
    source = course.sources[0]
    document = normalize_document(parse_pdf(REPO_COURSES / "demo" / source.file), source.noise_lines)

    structure = extract_structure(document, course.structure)

    assert structure.sha256 == document.sha256
    assert _outline(structure) == [
        (None, "", None),
        ("1", "Setting up the board", None),
        ("2", "Taking a turn", None),
        ("3", "Trading resources", None),
        ("4", "Scoring and the end of the game", None),
    ]
    assert len(structure.objectives) == 11
    assert all(objective.id[4:6] == f"{int(objective.section_id or 0):02d}" for objective in structure.objectives)
    assert all(objective.knowledge_level is None for objective in structure.objectives)


@pytest.mark.covers("US-004", "AC-16")
def test_numbered_list_items_and_out_of_order_headings_stay_text() -> None:
    """Guards the false-heading rule: a numbered list or a heading repeated out of order must not open a section."""
    document = _document(
        "1 Basics\nintro\n1.1 Terms\n1 Testing shows defects\n2 Exhaustive testing is impossible\n1.2 Errors",
        "2 Lifecycle\n2.2 Levels\nsee 2.1\n2.1 Models\n2.2 Levels\n3 Static",
    )

    structure = extract_structure(document, BOOK)

    assert [section.id for section in structure.sections] == ["1", "1.1", "1.2", "2", "2.1", "2.2", "3"]
    terms = structure.sections[1]
    assert [line.text for line in terms.lines] == [
        "1.1 Terms",
        "1 Testing shows defects",
        "2 Exhaustive testing is impossible",
    ]


@pytest.mark.covers("US-004", "AC-16")
def test_every_line_is_in_exactly_one_section_with_its_page() -> None:
    """Guards against text lost or duplicated between sections, and against pages without text breaking numbering."""
    document = _document("Foreword\n1 One\nfirst", "", "second\n2 Two\nthird")

    structure = extract_structure(document, BOOK)

    lines = [(line.page, line.text) for section in structure.sections for line in section.lines]
    assert lines == [(1, "Foreword"), (1, "1 One"), (1, "first"), (3, "second"), (3, "2 Two"), (3, "third")]
    assert structure.sections[0].id is None


@pytest.mark.covers("US-004", "AC-16")
def test_sections_know_their_nearest_enclosing_section() -> None:
    """Guards the parent links that objectives of a parent section are later inherited through."""
    document = _document("1 Basics\n1.1 Terms\n1.1.1 Words\n1.2 Errors\n2 Lifecycle\n2.1.1 Deep")

    structure = extract_structure(document, BOOK)

    assert _outline(structure) == [
        ("1", "Basics", None),
        ("1.1", "Terms", "1"),
        ("1.1.1", "Words", "1.1"),
        ("1.2", "Errors", "1"),
        ("2", "Lifecycle", None),
        ("2.1.1", "Deep", "2"),
    ]


@pytest.mark.covers("US-004", "AC-13")
def test_objective_goes_to_the_section_its_id_names_with_its_level() -> None:
    """Guards objectives listed at a chapter's start: they belong to the section in their id, not where printed."""
    document = _document(
        "1 Basics\nLO-1.1.1 [L1] Recall a term\nLO-1.2.1 [L2] Explain an error\n1.1 Terms\n1.2 Errors\n"
        "Changes: LO-1.2.2 was removed\nLO-1.1.1 [L3] A second definition is ignored"
    )

    structure = extract_structure(document, DOTTED)

    assert [(o.id, o.section_id, o.knowledge_level, o.line.page) for o in structure.objectives] == [
        ("LO-1.1.1", "1.1", "L1", 1),
        ("LO-1.2.1", "1.2", "L2", 1),
    ]


@pytest.mark.covers("US-004", "AC-13")
def test_objective_without_a_section_group_belongs_where_it_is_defined() -> None:
    """Guards courses that list objectives inside their sections: the defining section is the objective's own."""
    document = _document("Unit 1. Setup\nOBJ-01-01 Describe the board\nUnit 2. Turns\nOBJ-02-01 List the phases")

    structure = extract_structure(document, UNITS)

    assert [(o.id, o.section_id) for o in structure.objectives] == [("OBJ-01-01", "1"), ("OBJ-02-01", "2")]


@pytest.mark.covers("US-004", "AC-13")
def test_objective_naming_a_missing_section_is_an_error() -> None:
    """Guards against an objective silently attached to nothing when the rules and the text disagree."""
    document = _document("1 Basics\nLO-1.3.1 [L1] Recall a term\n1.1 Terms")

    with pytest.raises(CourseStructureError, match=r"source\.pdf: objective LO-1\.3\.1 on page 1 .* section 1\.3"):
        extract_structure(document, DOTTED)


@pytest.mark.covers("US-004", "AC-16")
def test_ids_that_are_not_numbers_keep_their_first_heading() -> None:
    """Guards courses with lettered sections: no ordering applies, and a repeated id is text, not a new section."""
    rules = StructureRules.model_validate({"section_heading": r"^Appendix (?P<id>[A-Z])\. (?P<title>.+)$"})
    document = _document("Appendix B. Terms\nAppendix A. Rules\nAppendix B. Terms again")

    structure = extract_structure(document, rules)

    assert [section.id for section in structure.sections] == ["B", "A"]


@pytest.mark.covers("US-004", "AC-16")
@pytest.mark.parametrize(
    ("ids", "expected"),
    [
        ([], []),
        (["0", "0.1", "1", "2", "0.2", "0.3", "1"], [0, 1, 4, 5, 6]),
        (["1", "1.1", "4.5", "1.2", "2"], [0, 1, 3, 4]),
        (["3", "1", "2"], [1, 2]),
        (["2", "1"], [0]),
    ],
    ids=[
        "empty",
        "numbered-list-after-a-subsection",
        "heading-out-of-order",
        "longer-run-later",
        "earliest-of-equal-runs",
    ],
)
def test_accepted_headings_form_the_longest_increasing_run(ids: list[str], expected: list[int]) -> None:
    """Guards the choice itself: the longest strictly increasing run, the earliest one on a tie."""
    assert accepted_headings(ids) == expected


@dataclass(frozen=True)
class _Generated:
    """A generated source: its page texts and the sections and objectives it was built from."""

    pages: tuple[str, ...]
    sections: tuple[tuple[str, str], ...]
    objectives: tuple[tuple[str, str], ...]


_WORDS = st.lists(st.sampled_from(["river", "barge", "salt", "trade", "season", "price"]), min_size=1, max_size=6)
_TITLE = _WORDS.map(lambda words: " ".join(words).capitalize())
_BODY_LINE = _WORDS.map(" ".join)


@st.composite
def _paginate(draw: st.DrawFn, lines: list[str]) -> tuple[str, ...]:
    """Spread lines over pages of random length, with some pages left without text."""
    pages: list[str] = []
    rest = lines
    while rest:
        if draw(st.booleans()) and draw(st.booleans()):
            pages.append("")
        size = draw(st.integers(min_value=1, max_value=8))
        pages.append("\n".join(rest[:size]))
        rest = rest[size:]
    return tuple(pages)


@st.composite
def _unit_course(draw: st.DrawFn) -> _Generated:
    """A course of flat units with objectives inside them; a unit may refer back to an earlier unit by heading."""
    lines = draw(st.lists(_BODY_LINE, max_size=2))
    sections: list[tuple[str, str]] = []
    objectives: list[tuple[str, str]] = []
    for unit in range(1, draw(st.integers(min_value=1, max_value=5)) + 1):
        title = draw(_TITLE)
        sections.append((str(unit), title))
        lines.append(f"Unit {unit}. {title}")
        for number in range(1, draw(st.integers(min_value=0, max_value=3)) + 1):
            objective = f"OBJ-{unit:02d}-{number:02d}"
            objectives.append((objective, str(unit)))
            lines.append(f"{objective} {draw(_BODY_LINE)}")
        lines.extend(draw(st.lists(_BODY_LINE, max_size=4)))
        if unit > 1 and draw(st.booleans()):
            # A repeated earlier heading, as a cross-reference would quote it.
            lines.append(f"Unit {draw(st.integers(min_value=1, max_value=unit - 1))}. {draw(_TITLE)}")
    return _Generated(draw(_paginate(lines)), tuple(sections), tuple(objectives))


@st.composite
def _book(draw: st.DrawFn) -> _Generated:
    """A book with dotted chapter numbers, no objectives, and numbered lists that restart below the chapter."""
    lines: list[str] = []
    sections: list[tuple[str, str]] = []
    for chapter in range(1, draw(st.integers(min_value=1, max_value=4)) + 1):
        ids = [str(chapter)]
        for section in range(1, draw(st.integers(min_value=0, max_value=3)) + 1):
            ids.append(f"{chapter}.{section}")
            ids.extend(f"{chapter}.{section}.{sub}" for sub in range(1, draw(st.integers(0, 2)) + 1))
        for section_id in ids:
            title = draw(_TITLE)
            sections.append((section_id, title))
            lines.append(f"{section_id} {title}")
            lines.extend(draw(st.lists(_BODY_LINE, max_size=3)))
            if chapter > 1 and draw(st.booleans()):
                # A numbered list that starts again at 1 and stays below the chapter number.
                count = draw(st.integers(min_value=1, max_value=chapter - 1))
                lines.extend(f"{item} {draw(_TITLE)}" for item in range(1, count + 1))
    return _Generated(draw(_paginate(lines)), tuple(sections), ())


@pytest.mark.covers("US-004", "AC-13", "AC-15", "AC-16")
@settings(max_examples=200, derandomize=True, deadline=None)
@given(st.one_of(_unit_course().map(lambda g: (g, UNITS)), _book().map(lambda g: (g, BOOK))))
def test_generated_courses_give_back_the_structure_they_were_built_from(
    case: tuple[_Generated, StructureRules],
) -> None:
    """Guards the structure rules on two course shapes at once, with decoy headings and pages without text.

    Every heading the generator wrote is found with its title, nothing else becomes a section, every line ends
    up in exactly one section in reading order, and every objective is found in the section it was written for.
    """
    generated, rules = case
    document = _document(*generated.pages)

    structure = extract_structure(document, rules)

    numbered = [(section.id, section.title) for section in structure.sections if section.id is not None]
    assert numbered == list(generated.sections)
    found_lines = [line.text for section in structure.sections for line in section.lines]
    assert found_lines == [line for page in generated.pages if page for line in page.split("\n")]
    assert [(o.id, o.section_id) for o in structure.objectives] == list(generated.objectives)


@pytest.mark.covers("US-004", "AC-16")
def test_heading_pattern_is_matched_against_the_whole_line() -> None:
    """Guards against a heading found inside a sentence when a course's pattern is not anchored."""
    rules = StructureRules.model_validate({"section_heading": r"Part (?P<id>\d+) (?P<title>\w+)"})
    document = _document("Part 1 Rules\nas said in Part 2 Notes below")

    structure = extract_structure(document, rules)

    assert [section.id for section in structure.sections] == ["1"]
