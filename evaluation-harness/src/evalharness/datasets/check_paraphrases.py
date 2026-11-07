"""Check that no question or rewording of a paraphrase set copies a held-out question of the course's question bank.

The held-out questions are kept for a final measurement and nothing is tuned on them. The paraphrase relation will
be used to choose search settings, so a pair that is close to a held-out question would let those questions leak
into the tuning. A pair counts as too close when most of its four-word sequences also occur in one held-out question:
that is what a copied or lightly edited question looks like. Two questions on the same topic in their own words
share far fewer, so they pass. The check sees word copies only, not the same question in other words.

    poetry run python -m evalharness.datasets.check_paraphrases datasets/<course>/paraphrase_v0.toml \
        datasets/<course>/retrieval_gold_v0.toml

Run it from evaluation-harness/; it reads the local question bank named by the gold declaration from the stack's
local course folder and needs no running service. Exit code 0 when no pair is too close, 1 when one is (each one is
printed), 2 when the bank cannot be read as declared.
"""

import re
import sys
from collections.abc import Sequence
from pathlib import Path

from evalharness.datasets.paraphrase_set import load_paraphrase_set
from evalharness.datasets.retrieval_gold import GoldSetError, build_gold_set, load_declaration
from evalharness.evaluation.retrieval import RunRefused, bank_folder
from evalharness.settings import HarnessSettings, load_stack_settings

# Length of the word sequences compared. Shorter ones are shared by ordinary phrasing ("what is the difference
# between" alone holds two four-word sequences); longer ones would miss a copy with a few words changed.
SEQUENCE_WORDS = 4
# Share of a text's word sequences found in one held-out question from which the text counts as a copy of it.
MAX_SHARED = 0.5


def word_sequences(text: str) -> set[tuple[str, ...]]:
    """Return the text's sequences of SEQUENCE_WORDS words, in lower case and without punctuation."""
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i : i + SEQUENCE_WORDS]) for i in range(len(words) - SEQUENCE_WORDS + 1)}


def shared_share(text: str, other: str) -> float:
    """Return the share of the text's word sequences that also occur in ``other``; 0 when it has none."""
    own = word_sequences(text)
    return len(own & word_sequences(other)) / len(own) if own else 0.0


def find_copies(texts: Sequence[tuple[str, str]], held_out: Sequence[tuple[str, str]]) -> list[str]:
    """Return one line per text (label, text) that counts as a copy of a held-out question (id, text)."""
    problems = []
    for label, text in texts:
        for question_id, question in held_out:
            share = shared_share(text, question)
            if share >= MAX_SHARED:
                problems.append(f"{label}: shares {share:.2f} of its word sequences with held-out {question_id}")
    return problems


def main(argv: Sequence[str]) -> int:
    """Check the pair file against the held-out questions of the gold declaration named in ``argv``."""
    if len(argv) != 2:
        print("usage: python -m evalharness.datasets.check_paraphrases <pair_file> <gold_declaration>", file=sys.stderr)
        return 2
    pairs = load_paraphrase_set(Path(argv[0]))
    declaration = load_declaration(Path(argv[1]))
    settings = HarnessSettings()
    stack = load_stack_settings(settings.stack_env_file)
    try:
        folder = bank_folder(declaration, stack.local_course_id, stack.local_course_sources, settings.stack_env_file)
        gold = build_gold_set(declaration, folder)
    except (RunRefused, GoldSetError) as error:
        print(f"cannot read the bank: {error}", file=sys.stderr)
        return 2
    held_out = [(case.case_id, case.query) for case in gold.cases if case.held_out]
    texts = [(f"{p.case_id} question", p.question) for p in pairs.pairs]
    texts += [(f"{p.case_id} rewording", p.paraphrase) for p in pairs.pairs]
    problems = find_copies(texts, held_out)
    for problem in problems:
        print(problem)
    highest = max((shared_share(text, question), label, qid) for label, text in texts for qid, question in held_out)
    print(
        f"{len(texts)} texts against {len(held_out)} held-out questions: highest shared share {highest[0]:.2f} "
        f"({highest[1]} / {highest[2]}), {len(problems)} at or above {MAX_SHARED}"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
