import re
from pathlib import Path

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from study_assistant.courses.registry import load_course
from study_assistant.ingest.normalize import (
    build_vocabulary,
    clean_layout,
    join_broken_words,
    normalize_document,
    normalize_text,
)
from study_assistant.ingest.parser import parse_pdf

# app/tests/unit/<this file> -> repository root.
REPO_COURSES = Path(__file__).resolve().parents[3] / "courses"
DEMO_DIR = REPO_COURSES / "demo"

MARK = "\ufffe"
FOOTER = [re.compile(r"Page \d+ of \d+")]


def _reference_text(markdown: str) -> str:
    """Turn the demo Markdown into the words a reader sees: markup removed, whitespace collapsed."""
    text = markdown.replace("&shy;", "").replace("<!-- figure page -->", "")
    text = re.sub(r"^#+ ", "", text, flags=re.MULTILINE)
    return " ".join(text.split())


@pytest.mark.covers("US-004", "AC-9", "AC-10")
def test_demo_pdf_normalizes_to_its_known_text() -> None:
    """Guards the whole cleaning path on real engine output against the source it was built from.

    Any lost or reordered paragraph, a header or footer left in or a sentence removed by a too-greedy pattern,
    and a word broken at a line end and not repaired all make the text differ from the Markdown.
    """
    manifest = load_course(REPO_COURSES, "demo")
    parsed = parse_pdf(DEMO_DIR / manifest.sources[0].file)

    document = normalize_document(parsed, manifest.sources[0].noise_lines)

    assert [(page.number, page.status) for page in document.pages] == [
        (1, "ok"),
        (2, "no_text_layer"),
        (3, "ok"),
        (4, "ok"),
    ]
    assert (document.parser, document.sha256) == (parsed.parser, parsed.sha256)
    text = " ".join(" ".join(page.text for page in document.pages).split())
    assert text == _reference_text((DEMO_DIR / "vardolin_rules.md").read_text(encoding="utf-8"))


@pytest.mark.covers("US-004", "AC-10")
@pytest.mark.parametrize(
    ("broken", "vocabulary", "expected"),
    [
        ("turn" + MARK + "order", {"turn-order", "turnorder"}, "turn-order"),
        ("rec" + MARK + "eives", {"receives"}, "receives"),
        ("score" + MARK + "sheet", {"score", "sheet"}, "score-sheet"),
        ("fur" + MARK + "thest", set(), "furthest"),
        ("end-of" + MARK + "game", {"end-of-game"}, "end-of-game"),
        ("Neg" + MARK + "otiation", {"negotiation"}, "Negotiation"),
    ],
    ids=["hyphenated-seen", "joined-seen", "both-parts-are-words", "no-evidence", "multi-part", "keeps-case"],
)
def test_broken_word_is_repaired_from_document_evidence(broken: str, vocabulary: set[str], expected: str) -> None:
    """Guards each rule and their order: a wrong choice either glues a compound or splits an ordinary word."""
    assert join_broken_words(f"a {broken} b", frozenset(vocabulary)) == f"a {expected} b"


@pytest.mark.covers("US-004", "AC-10")
def test_broken_word_is_not_evidence_for_itself() -> None:
    """Guards the vocabulary: counting the halves of a broken word as words would always keep a false hyphen."""
    assert build_vocabulary(["score" + MARK + "sheet", "the rules"]) == frozenset({"the", "rules"})


@pytest.mark.covers("US-004", "AC-10")
def test_noise_patterns_remove_whole_lines_only() -> None:
    """Guards against prefix matching: a sentence that only starts like a footer must stay in the text."""
    text = "Page 2 of 4 barges stay on the board.\nPage 2 of 4"

    assert clean_layout(text, FOOTER) == "Page 2 of 4 barges stay on the board."


@pytest.mark.covers("US-004", "AC-10")
def test_layout_noise_is_removed_and_paragraph_breaks_kept() -> None:
    """Guards the layout rules: line-end variants (a lone CR too), inner spacing, blank-line runs and blank edges."""
    text = "\r\n  First\tline  here \rthird\r\nsecond\u00a0line\r\r\n\n\nNext paragraph  \n\n"

    assert clean_layout(text, []) == "First line here\nthird\nsecond line\n\nNext paragraph"


@pytest.mark.covers("US-004", "AC-10")
def test_letters_are_brought_to_one_unicode_form() -> None:
    """Guards against two identical-looking words staying different byte sequences."""
    assert normalize_text("cafe\u0301", [], frozenset()) == "caf\u00e9"


# Pieces that real extracted text is made of, including the awkward ones, mixed with arbitrary text.
_DIRTY_PIECES = st.sampled_from(
    [MARK, "\r\n", "\r", "\n", " ", "\t", "\u00a0", "-", "e\u0301", "\u0301", "Page 3 of 9", "word", "Word", "x"]
)
_DIRTY_TEXT = st.lists(st.one_of(st.text(max_size=5), _DIRTY_PIECES), max_size=30).map("".join)


@settings(max_examples=300, derandomize=True, deadline=None)
@given(_DIRTY_TEXT)
# A stray mark between a letter and a combining accent: deleting it would let the accent compose on a second pass.
@pytest.mark.covers("US-004", "AC-10")
@example("e" + MARK + "\u0301")
def test_normalizing_twice_changes_nothing(text: str) -> None:
    """Guards the output's cleanliness by the module's own rules: if cleaning it again changes it, one rule left
    behind something another rule removes, so the rules are incomplete or in the wrong order."""
    vocabulary = build_vocabulary([text, "word turn-order"])

    once = normalize_text(text, FOOTER, vocabulary)

    assert normalize_text(once, FOOTER, vocabulary) == once
    assert MARK not in once
    assert "\r" not in once
