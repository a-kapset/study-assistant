"""Reading the text layer of a PDF source, one page at a time.

The parser only extracts; deciding what is noise (running headers, line-end hyphens) and whether a page has
any text at all is left to normalization, which knows the course's rules. Every parsed document records which
parser and PDF engine produced it, so a change in the extracted text can be traced to a change of version.
"""

import hashlib
import threading
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import pypdfium2

PARSER_NAME = "pypdfium2"

# PDFium is not safe to call from several threads at once; two uploads parsed in parallel could crash the process.
_PDFIUM_LOCK = threading.Lock()


class SourceParseError(ValueError):
    """A source file is not a readable PDF; the message names the file and what the PDF engine reported."""


@dataclass(frozen=True)
class ParserVersion:
    """Which parser produced a text, down to the PDF engine build that decides line breaks and hyphens."""

    name: str
    version: str
    engine_version: str


@dataclass(frozen=True)
class RawPage:
    """The text layer of one page exactly as the engine returned it; pages are numbered from 1."""

    number: int
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    """All pages of one source file, in order, with the parser version that read them."""

    path: Path
    # SHA-256 of the exact bytes that were parsed: identifies this edition of the file, whatever its name.
    sha256: str
    parser: ParserVersion
    pages: tuple[RawPage, ...]


def parser_version() -> ParserVersion:
    """Report the installed parser and PDF engine versions."""
    return ParserVersion(
        name=PARSER_NAME,
        version=version("pypdfium2"),
        engine_version=str(pypdfium2.version.PDFIUM_INFO),
    )


def parse_pdf(path: Path) -> ParsedDocument:
    """Extract the text layer of every page of a PDF file.

    A page without a text layer (a scan, a picture) comes back with empty text rather than being skipped,
    so page numbers stay aligned with the file. The file is read once, so the hash and the text come from the
    same bytes even if the file is replaced while it is being parsed.
    """
    data = path.read_bytes()
    with _PDFIUM_LOCK:
        try:
            pdf = pypdfium2.PdfDocument(data)
        except pypdfium2.PdfiumError as exc:
            raise SourceParseError(f"{path} cannot be read as a PDF: {exc}") from exc
        try:
            pages = tuple(_read_page(pdf, index) for index in range(len(pdf)))
        except pypdfium2.PdfiumError as exc:
            raise SourceParseError(f"{path}: a page cannot be read: {exc}") from exc
        finally:
            pdf.close()
    return ParsedDocument(path=path, sha256=hashlib.sha256(data).hexdigest(), parser=parser_version(), pages=pages)


def _read_page(pdf: pypdfium2.PdfDocument, index: int) -> RawPage:
    """Read one page's text and release the engine's page objects, innermost first."""
    page = pdf[index]
    try:
        text_page = page.get_textpage()
        try:
            text: str = text_page.get_text_range()
        finally:
            text_page.close()
    finally:
        page.close()
    return RawPage(number=index + 1, text=text)
