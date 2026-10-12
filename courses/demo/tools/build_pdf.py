"""Build the demo course PDF from its Markdown source.

The Markdown file is the known text of the course; the PDF built from it is what the app ingests, so the
PDF path (pages, running header and footer, words broken at line ends) is exercised with a text whose every
word is known. The output is byte-for-byte the same on every run, so the committed PDF only changes when
the Markdown or this script changes.

Run from the repository root (Poetry runs the command inside app/):
    poetry -C app install --with tools
    poetry -C app run python ../courses/demo/tools/build_pdf.py
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fpdf import FPDF

COURSE_DIR = Path(__file__).resolve().parent.parent
SOURCE = COURSE_DIR / "vardolin_rules.md"
TARGET = COURSE_DIR / "vardolin_rules.pdf"

# Repeated on every page; the course manifest holds the patterns that remove these lines again.
RUNNING_HEADER = "The Rules of Vardolin, edition 1"
FOOTER_TEMPLATE = "Page {page} of {{nb}}"

# A fixed creation date: without it every build would differ by its timestamp.
CREATION_DATE = datetime(2026, 1, 1, tzinfo=UTC)

# A visible marker in the source for a soft hyphen: the place where a long word may be broken at a line end.
SOFT_HYPHEN_MARKER = "&shy;"
SOFT_HYPHEN = "\u00ad"
FIGURE_MARKER = "<!-- figure page -->"

type BlockKind = Literal["title", "heading", "paragraph", "figure"]


@dataclass(frozen=True)
class Block:
    """One unit of the source: the document title, a unit heading, a paragraph or a figure page."""

    kind: BlockKind
    text: str = ""


def parse_blocks(markdown: str) -> list[Block]:
    """Split the source into blocks separated by blank lines, using only the Markdown subset the course needs."""
    blocks: list[Block] = []
    for chunk in markdown.split("\n\n"):
        text = " ".join(chunk.split()).replace(SOFT_HYPHEN_MARKER, SOFT_HYPHEN)
        if not text:
            continue
        if text == FIGURE_MARKER:
            blocks.append(Block("figure"))
        elif text.startswith("## "):
            blocks.append(Block("heading", text.removeprefix("## ")))
        elif text.startswith("# "):
            blocks.append(Block("title", text.removeprefix("# ")))
        else:
            blocks.append(Block("paragraph", text))
    return blocks


class RulesPdf(FPDF):
    """An A4 document with the running header and the page footer on every page."""

    def header(self) -> None:
        """Write the running header at the top of the page."""
        self.set_font("helvetica", size=9)
        self.cell(0, 8, RUNNING_HEADER, new_x="LMARGIN", new_y="NEXT")
        self.ln(4)

    def footer(self) -> None:
        """Write "Page N of M" at the bottom of the page."""
        self.set_y(-15)
        self.set_font("helvetica", size=9)
        self.cell(0, 8, FOOTER_TEMPLATE.format(page=self.page_no()), align="C")


def draw_board(pdf: FPDF) -> None:
    """Draw the river board with shapes only, so the page carries no text of its own."""
    centre_x, centre_y = 105.0, 140.0
    pdf.circle(centre_x, centre_y, 20, style="D")
    for column in range(4):
        for row in range(4):
            if column in (1, 2) and row in (1, 2):
                continue  # the Lantern Isle sits in the middle of the ring
            pdf.rect(60 + column * 24, 95 + row * 24, 20, 20, style="D")


def render(blocks: list[Block]) -> bytes:
    """Lay the blocks out on pages and return the PDF bytes."""
    pdf = RulesPdf()
    pdf.set_creation_date(CREATION_DATE)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    for block in blocks:
        if block.kind == "figure":
            pdf.add_page()
            draw_board(pdf)
            pdf.add_page()
        elif block.kind == "title":
            pdf.set_font("helvetica", style="B", size=18)
            pdf.multi_cell(0, 10, block.text, align="L", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)
        elif block.kind == "heading":
            pdf.set_font("helvetica", style="B", size=13)
            pdf.multi_cell(0, 8, block.text, align="L", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
        else:
            pdf.set_font("helvetica", size=11)
            pdf.multi_cell(0, 6, block.text, align="L", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)
    return bytes(pdf.output())


def main() -> None:
    """Build the PDF next to its Markdown source and report the result."""
    data = render(parse_blocks(SOURCE.read_text(encoding="utf-8")))
    TARGET.write_bytes(data)
    print(f"Wrote {TARGET.name}: {len(data)} bytes")


if __name__ == "__main__":
    main()
