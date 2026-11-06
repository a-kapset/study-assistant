"""Check a Q&A set against the stored course before it is used: does the course really answer the answerable
questions in their gold sections, and really not answer the others?

A question is written with the words it turns on (its key terms). For an answerable question every key term must
occur in a passage of its gold section or a subsection; otherwise the gold section is likely wrong and a correct
answer would score as citing another section. For an unanswerable question no key term may occur in any passage of
the course; otherwise the course may answer it after all, and a correct answer would count as unsupported.

    poetry run python -m evalharness.datasets.check_qa_set datasets/<course>/qa_v1.toml

Run it from evaluation-harness/ on the stack with the course ingested; it reads the passages through the read-only
role and asks no question. Exit code 0 when every case passes, 1 when a case has a problem (each one is printed).
The check looks at words only: whether a passage really answers a question stays the writer's judgement.
"""

import re
import sys
from collections.abc import Sequence
from pathlib import Path

from evalharness.corpus_db import CorpusReader
from evalharness.datasets.qa_set import QaSet, load_qa_set
from evalharness.oracles.retrieval import is_relevant
from evalharness.settings import HarnessSettings, load_stack_settings


def _normal(text: str) -> str:
    """Lower-case ``text`` and turn every run of white space into one space, as line breaks of a PDF fall anywhere."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _occurs(term: str, text: str) -> bool:
    """Say whether ``term`` occurs in ``text`` at the start of a word: "error" is found in "errors", but "pact" is
    not found in "impact"."""
    return re.search(rf"(?<!\w){re.escape(term)}", text) is not None


def find_problems(qa: QaSet, passages: Sequence[tuple[str | None, str]]) -> list[str]:
    """Return one line per problem of the set against ``passages`` (section, text); empty when every case passes."""
    texts = [(section, _normal(text)) for section, text in passages]
    problems = []
    for case in qa.cases:
        if not case.key_terms:
            problems.append(f"{case.id}: no key terms")
            continue
        for term in map(_normal, case.key_terms):
            if case.gold_section is None:
                where = sorted({section or "-" for section, text in texts if _occurs(term, text)})
                if where:
                    problems.append(f"{case.id}: '{term}' occurs in the course (sections {', '.join(where)})")
            elif not any(_occurs(term, text) for section, text in texts if is_relevant(section, case.gold_section)):
                problems.append(f"{case.id}: '{term}' occurs in no passage of section {case.gold_section}")
    return problems


def main(argv: Sequence[str]) -> int:
    """Check the set file named in ``argv`` and return the exit code: 0 when it passes, 1 when a case has a problem."""
    if len(argv) != 1:
        print("usage: python -m evalharness.datasets.check_qa_set <qa_file>", file=sys.stderr)
        return 2
    qa = load_qa_set(Path(argv[0]))
    settings = HarnessSettings()
    with CorpusReader(settings, load_stack_settings(settings.stack_env_file)) as corpus:
        passages = corpus.section_passages(qa.course_id)
    problems = find_problems(qa, passages)
    for problem in problems:
        print(problem)
    answerable = sum(case.gold_section is not None for case in qa.cases)
    print(f"{len(qa.cases)} cases ({answerable} answerable) against {len(passages)} passages: {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
