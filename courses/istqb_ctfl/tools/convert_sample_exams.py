"""Convert the ISTQB CTFL v4.0 sample exams into the app's common question-bank format.

Each sample exam set comes as two PDFs: the questions (stem, options, "Select ONE option") and the answers (a key
table with the correct options, learning objective and K-level of every question). The two documents are read
separately and checked against each other before anything is written, because a misread table row or a lost
option would otherwise become a wrong answer key without any visible error.

The PDFs and the bank written from them stay on the local machine; only this script is committed.

Run from the repository root (Poetry runs the command inside app/):
    poetry -C app run python ../courses/istqb_ctfl/tools/convert_sample_exams.py <exams folder> <bank file>
e.g. <exams folder> = C:/materials/istqb/Sample_Exams, <bank file> = C:/materials/istqb/sample_exams_bank.jsonl
"""

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from study_assistant.exams.bank import BankItem, BankOption, BankSource, read_bank
from study_assistant.ingest.parser import parse_pdf

# "ISTQB_CTFL_v4.0_Sample-Exam-A-Questions_v1.7.pdf" -> set "A", questions document, version "1.7".
DOCUMENT_NAME = re.compile(r"Sample-Exam-(?P<set>[A-Z])-(?P<kind>Questions|Answers)_v(?P<version>[\d.]+)\.pdf$")

# The running header of every page. The questions use "Sample Exams set A", the answers "Sample Exam set A".
RUNNING_HEADER = (
    re.compile(r"Certified Tester, Foundation Level"),
    re.compile(r"Sample Exams? set [A-Z]"),
    re.compile(r"Sample Exam – (Questions|Answers)"),
    re.compile(r"Version \S+ Page \d+ of \d+ Release .*"),
    re.compile(r"© International Software Testing Qualifications Board"),
)

# "Question #12 (1 Point)" or "Question #A7 (1 Point)" for an additional question; table of contents lines end
# with a row of dots and a page number, so they do not match a whole line.
QUESTION_HEADER = re.compile(r"Question #(?P<number>A?\d+) \(\d+ Points?\)")
OPTION = re.compile(r"(?P<id>[a-e])\)\s*(?P<text>.*)")
# One set writes "Select ONE options." in places, so the plural "s" is optional.
SELECT = re.compile(r"Select (?P<count>ONE|TWO|THREE) options?\.?")
SELECT_COUNTS = {"ONE": 1, "TWO": 2, "THREE": 3}

# A key table row: "6 a, e FL-1.4.5 K2 1". The PDF prints the table in two columns, so one text line holds two rows.
KEY_ROW = re.compile(
    r"(?P<number>A?\d+) (?P<keys>[a-e](?:, ?[a-e])*) (?P<objective>FL-\d+\.\d+\.\d+) (?P<level>K\d) \d+"
)
# The key tables end where the answers with their justifications begin.
KEY_TABLES_END = re.compile(r"^(Appendix: )?Answers( to Additional Sample Questions)?$")


@dataclass
class Question:
    """A question as read from the questions document, before its key is known."""

    number: str
    page: int
    stem: list[str] = field(default_factory=list)
    options: list[list[str]] = field(default_factory=list)
    select: int | None = None


@dataclass(frozen=True)
class KeyRow:
    """One row of an answer key table."""

    keys: tuple[str, ...]
    objective_id: str
    knowledge_level: str


@dataclass(frozen=True)
class ExamSet:
    """The two documents of one sample exam set."""

    name: str
    questions: Path
    answers: Path
    version: str


def page_lines(path: Path) -> list[tuple[int, str]]:
    """Return (page number, line) pairs of a PDF without the running header and without trailing spaces."""
    return [
        (page.number, line.strip())
        for page in parse_pdf(path).pages
        for line in page.text.splitlines()
        if not any(pattern.fullmatch(line.strip()) for pattern in RUNNING_HEADER)
    ]


def read_questions(path: Path) -> dict[str, Question]:
    """Read every question of a questions document: its stem lines, its options and how many to select.

    A line that does not start a new option continues the last option; before the first option it belongs to
    the stem, so scenarios and lists such as "i. ... ii. ..." stay part of the stem.
    """
    questions: dict[str, Question] = {}
    current: Question | None = None
    for page, line in page_lines(path):
        header = QUESTION_HEADER.fullmatch(line)
        if header:
            current = Question(number=header["number"], page=page)
            questions[current.number] = current
            continue
        if current is None or current.select is not None or not line:
            continue
        select = SELECT.fullmatch(line)
        option = OPTION.fullmatch(line)
        if select:
            current.select = SELECT_COUNTS[select["count"]]
        elif option:
            current.options.append([option["id"], option["text"]])
        elif current.options:
            current.options[-1][1] += f" {line}"
        else:
            current.stem.append(line)
    return questions


def read_key(path: Path) -> dict[str, KeyRow]:
    """Read the answer key tables (main and additional questions) of an answers document."""
    rows: dict[str, KeyRow] = {}
    in_tables = False
    for _, line in page_lines(path):
        if line in ("Answer Key", "Appendix: Answer Key for Additional Sample Questions"):
            in_tables = True
        elif KEY_TABLES_END.match(line):
            in_tables = False
        elif in_tables:
            for row in KEY_ROW.finditer(line):
                keys = tuple(key.strip() for key in row["keys"].split(","))
                rows[row["number"]] = KeyRow(keys, row["objective"], row["level"])
    return rows


def differences(exam: ExamSet, questions: dict[str, Question], key: dict[str, KeyRow]) -> list[str]:
    """List every way the two documents of a set disagree; an empty list means they can be joined."""
    found = [f"{exam.name}: question {n} has no key row" for n in questions if n not in key]
    found += [f"{exam.name}: key row {n} has no question" for n in key if n not in questions]
    for number, question in questions.items():
        row = key.get(number)
        option_ids = [option_id for option_id, _ in question.options]
        if option_ids != [chr(ord("a") + i) for i in range(len(option_ids))]:
            found.append(f"{exam.name} #{number}: option ids {option_ids} are not a, b, c, ... in order")
        if row is None:
            continue
        if question.select != len(row.keys):
            found.append(f"{exam.name} #{number}: asks to select {question.select}, the key has {list(row.keys)}")
        if not set(row.keys) <= set(option_ids):
            found.append(f"{exam.name} #{number}: key {list(row.keys)} is not among the options {option_ids}")
    return found


def bank_items(exam: ExamSet, questions: dict[str, Question], key: dict[str, KeyRow]) -> list[BankItem]:
    """Join the questions with their key rows into bank items; the bank model checks each one again."""
    return [
        BankItem(
            id=f"{exam.name}-{number}",
            stem="\n".join(question.stem),
            options=tuple(BankOption(id=option_id, text=text) for option_id, text in question.options),
            keys=key[number].keys,
            objective_id=key[number].objective_id,
            knowledge_level=key[number].knowledge_level,
            source=BankSource(document=exam.questions.name, version=exam.version, question=number, page=question.page),
        )
        for number, question in questions.items()
    ]


def find_sets(folder: Path) -> list[ExamSet]:
    """Pair the questions and answers documents of each set found in the folder; a set missing one is an error."""
    documents: dict[str, dict[str, re.Match[str]]] = {}
    for path in sorted(folder.glob("*.pdf")):
        match = DOCUMENT_NAME.search(path.name)
        if match:
            documents.setdefault(match["set"], {})[match["kind"]] = match
    sets = []
    for name, pair in sorted(documents.items()):
        if set(pair) != {"Questions", "Answers"}:
            sys.exit(f"set {name}: only the {', '.join(pair)} document is in {folder}")
        sets.append(
            ExamSet(
                name=name,
                questions=folder / pair["Questions"].string,
                answers=folder / pair["Answers"].string,
                version=pair["Questions"]["version"],
            )
        )
    if not sets:
        sys.exit(f"no sample exam documents in {folder}")
    return sets


def main() -> None:
    """Convert every set in the folder; write the bank only when no set has a difference."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("exams", type=Path, help="folder with the sample exam PDFs")
    parser.add_argument("bank", type=Path, help="bank file to write (JSON Lines)")
    args = parser.parse_args()

    items: list[BankItem] = []
    problems: list[str] = []
    for exam in find_sets(args.exams):
        questions = read_questions(exam.questions)
        key = read_key(exam.answers)
        found = differences(exam, questions, key)
        problems += found
        if not found:
            items += bank_items(exam, questions, key)
        print(f"set {exam.name} v{exam.version}: {len(questions)} questions, {len(key)} key rows")
    if problems:
        sys.exit("nothing written; the documents disagree:\n" + "\n".join(problems))

    args.bank.write_text(
        "".join(item.model_dump_json(exclude_none=True) + "\n" for item in items), encoding="utf-8", newline="\n"
    )
    read_back = read_bank(args.bank)
    digest = hashlib.sha256(args.bank.read_bytes()).hexdigest()
    print(f"{len(read_back)} items written to {args.bank}\nsha256 {digest}")


if __name__ == "__main__":
    main()
