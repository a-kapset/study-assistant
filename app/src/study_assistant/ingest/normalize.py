"""Cleaning extracted page text so that it reads like the original document and nothing else.

What a PDF engine returns is the text as it was drawn on the page, not as it was written: a running header
and footer on every page, a word split across two lines, line endings and spacing that depend on the layout,
and the same letter stored in different Unicode forms. Left in place, this noise ends up in every chunk: it
blurs the embeddings, repeats the same header line in search results and breaks words that a learner or a
question would use.

Normalization removes that noise and nothing more. Letter case, stop words and word forms are kept: the text
is embedded by a model trained on ordinary text, shown to learners as a quotation and read by a language
model, and all three need the original words. Each rule is a separate pure function, so it can be tested on
plain strings, and the output is clean by these same rules: cleaning it again changes nothing. If it did,
one rule would be leaving behind something another rule removes.

The rules that depend on a particular course (which lines are its running header and footer) come from the
course manifest as regular expressions; this module holds no course-specific text.
"""

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from study_assistant.ingest.parser import ParsedDocument, ParserVersion

# What a page holds after cleaning: text, or nothing at all (a scan, a picture, a page with only a header).
type PageStatus = Literal["ok", "no_text_layer"]

# PDFium marks the place where a word was split at a line end with U+FFFE and joins the two halves.
# It does the same for a real hyphen that happened to fall at a line end, so the mark alone cannot say
# whether the hyphen belongs to the word.
LINE_BREAK_MARK = "\ufffe"

# A word broken at a line end: one or more word parts (possibly hyphenated already), the mark, the rest.
_BROKEN_WORD = re.compile(r"([\w-]*\w)\ufffe(\w[\w-]*)")

# A word for the document's vocabulary: letters and digits, optionally joined by hyphens ("turn-order").
_WORD = re.compile(r"\w+(?:-\w+)*")

# Any token that still carries the mark; such tokens are left out of the vocabulary.
_MARKED_TOKEN = re.compile(r"\S*\ufffe\S*")

# Runs of whitespace inside a line (spaces, tabs, no-break spaces), but not the line ends themselves.
_INLINE_SPACE = re.compile(r"[^\S\n]+")


@dataclass(frozen=True)
class CleanPage:
    """One page after normalization; pages keep their numbers from the source file."""

    number: int
    text: str
    status: PageStatus


@dataclass(frozen=True)
class NormalizedDocument:
    """All pages of one source file after normalization, with the parser version that read them."""

    path: Path
    sha256: str
    parser: ParserVersion
    pages: tuple[CleanPage, ...]


def normalize_document(document: ParsedDocument, noise_lines: Sequence[re.Pattern[str]]) -> NormalizedDocument:
    """Normalize every page of a parsed document.

    The vocabulary used to repair words broken at line ends is built from the whole document first, because
    the evidence for a word ("is this hyphen part of the word?") is usually on another page.
    """
    texts = [unicode_form(page.text) for page in document.pages]
    vocabulary = build_vocabulary(texts)
    pages = tuple(
        _clean_page(page.number, text, noise_lines, vocabulary)
        for page, text in zip(document.pages, texts, strict=True)
    )
    return NormalizedDocument(path=document.path, sha256=document.sha256, parser=document.parser, pages=pages)


def normalize_text(text: str, noise_lines: Sequence[re.Pattern[str]], vocabulary: frozenset[str]) -> str:
    """Apply every rule to one page's text, in the order the rules depend on each other.

    Unicode form first, so that the other rules compare letters, not byte sequences; broken words next,
    while the mark is still there; layout last, so that removing a line can never leave a mark or a
    half-composed letter behind for a second pass to change.
    """
    return clean_layout(join_broken_words(unicode_form(text), vocabulary), noise_lines)


def unicode_form(text: str) -> str:
    """Bring every letter to its composed Unicode form (NFC).

    The same visible letter can be stored as one code point or as a base letter plus a combining mark;
    without this, two identical-looking words would not match in search, deduplication or comparison.
    """
    return unicodedata.normalize("NFC", text)


def build_vocabulary(texts: Iterable[str]) -> frozenset[str]:
    """Collect the words of a document, lower-cased, as evidence for repairing broken words.

    Tokens that still carry the line-break mark are left out: they are the words in question, not evidence.
    Lower-casing makes "Negotiation" at the start of a sentence count as evidence for "negotiation".
    """
    words: set[str] = set()
    for text in texts:
        words.update(word.lower() for word in _WORD.findall(_MARKED_TOKEN.sub(" ", text)))
    return frozenset(words)


def join_broken_words(text: str, vocabulary: frozenset[str]) -> str:
    """Repair words that were split at a line end, deciding from the document itself whether to keep a hyphen.

    For "left<mark>right" the rules are tried in order:
    1. the hyphenated form "left-right" occurs elsewhere in the document -> keep the hyphen;
    2. the joined form "leftright" occurs elsewhere -> join;
    3. both "left" and "right" are words of the document -> keep the hyphen (a compound seen only here);
    4. otherwise -> join (a long word split by the layout).
    A mark that is not between two word characters is replaced by a space, which the layout rules then remove.
    """

    def repair(match: re.Match[str]) -> str:
        """Choose between the hyphenated and the joined form of one broken word."""
        left, right = match.group(1), match.group(2)
        hyphenated, joined = f"{left}-{right}", left + right
        if hyphenated.lower() in vocabulary:
            return hyphenated
        if joined.lower() in vocabulary:
            return joined
        if left.lower() in vocabulary and right.lower() in vocabulary:
            return hyphenated
        return joined

    return _BROKEN_WORD.sub(repair, text).replace(LINE_BREAK_MARK, " ")


def clean_layout(text: str, noise_lines: Sequence[re.Pattern[str]]) -> str:
    """Remove what the page layout added: line-end variants, extra spacing, header and footer lines, blank lines.

    Every line is trimmed and its inner spacing collapsed to single spaces, so that the same sentence always
    has the same characters. A line that matches one of the course's header or footer patterns as a whole is
    dropped; matching the whole line, not a prefix, keeps a sentence that merely starts like a header.
    Runs of blank lines become one, so a paragraph break the PDF did keep survives, and the text has no
    blank lines at its start or end.
    """
    lines: list[str] = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = _INLINE_SPACE.sub(" ", raw_line).strip()
        if any(pattern.fullmatch(line) for pattern in noise_lines):
            continue
        if not line and (not lines or not lines[-1]):
            continue
        lines.append(line)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def page_status(text: str) -> PageStatus:
    """Classify a cleaned page: a page with no text left has no usable text layer and must not be ingested."""
    return "ok" if text else "no_text_layer"


def _clean_page(
    number: int, text: str, noise_lines: Sequence[re.Pattern[str]], vocabulary: frozenset[str]
) -> CleanPage:
    """Normalize one page and record whether any text is left on it."""
    cleaned = normalize_text(text, noise_lines, vocabulary)
    return CleanPage(number=number, text=cleaned, status=page_status(cleaned))
