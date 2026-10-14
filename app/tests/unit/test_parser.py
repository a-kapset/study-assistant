import hashlib
import io
from importlib.metadata import version
from pathlib import Path

import pypdfium2
import pytest

from study_assistant.ingest.parser import SourceParseError, parse_pdf

# The committed demo course PDF: app/tests/unit/<this file> -> repository root.
DEMO_PDF = Path(__file__).resolve().parents[3] / "courses" / "demo" / "vardolin_rules.pdf"


def _blank_pdf(path: Path, pages: int) -> Path:
    """Write a PDF of ``pages`` empty pages (no text layer at all) to ``path`` and return it."""
    pdf = pypdfium2.PdfDocument.new()
    for _ in range(pages):
        pdf.new_page(200, 200)
    buffer = io.BytesIO()
    pdf.save(buffer)
    pdf.close()
    path.write_bytes(buffer.getvalue())
    return path


@pytest.mark.covers("US-004", "AC-9")
def test_every_page_comes_back_in_order_numbered_from_one() -> None:
    """Guards page numbering: a skipped, reordered or zero-based page breaks every "page N" reference."""
    document = parse_pdf(DEMO_PDF)

    assert [page.number for page in document.pages] == [1, 2, 3, 4]
    assert document.pages[0].text.startswith("The Rules of Vardolin")


@pytest.mark.covers("US-004", "AC-9")
def test_pages_without_a_text_layer_are_kept_with_empty_text(tmp_path: Path) -> None:
    """Guards against dropping empty pages: a scan must stay visible so it can be reported, not vanish."""
    document = parse_pdf(_blank_pdf(tmp_path / "scan.pdf", pages=2))

    assert [(page.number, page.text) for page in document.pages] == [(1, ""), (2, "")]


@pytest.mark.covers("US-004", "AC-8")
def test_unreadable_file_raises_an_error_naming_it(tmp_path: Path) -> None:
    """Guards against a broken source being loaded as an empty document instead of failing with its name."""
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"this is not a PDF")

    with pytest.raises(SourceParseError) as caught:
        parse_pdf(path)

    assert str(path) in str(caught.value)


def test_parsed_document_records_the_parser_and_engine_versions() -> None:
    """Guards traceability: without the versions a change in extracted text cannot be tied to an upgrade."""
    parser = parse_pdf(DEMO_PDF).parser

    assert parser.name == "pypdfium2"
    assert parser.version == version("pypdfium2")
    assert parser.engine_version


def test_parsed_document_records_the_hash_of_the_bytes_it_read() -> None:
    """Guards source identity: chunk ids and corpus versions rest on this hash, so it must be of the file's bytes."""
    document = parse_pdf(DEMO_PDF)

    assert document.sha256 == hashlib.sha256(DEMO_PDF.read_bytes()).hexdigest()
