"""Check that no question or rewording of a paraphrase set asks the same thing as a held-out question of the course's
question bank.

The held-out questions are kept for a final measurement and nothing is tuned on them. The paraphrase relation will
be used to choose search settings, so a pair that asks what a held-out question asks would let that question leak
into the tuning. A copy is not the only risk: the pairs were written by someone who had seen the bank, so a held-out
question may come back in other words. The check therefore compares meaning, not words: every text and every
held-out question is embedded by the stack's own embedding model, and a text is flagged when it is closer to a
held-out question than any two questions of the set from different chapters are to each other. Such two questions
are known to ask different things, so their highest similarity is the line above which "the same question" begins;
it is measured on every run, not chosen by hand. A flagged pair is read by a person: when it really asks the same
thing, it is left out of the pair file; when it does not, it is listed under ``cleared`` with the reason.

    poetry run python -m evalharness.datasets.check_paraphrases datasets/<course>/paraphrase_v0.toml \
        datasets/<course>/retrieval_gold_v0.toml

Run it from evaluation-harness/ with the stack's embedding server up; it reads the local question bank named by the
gold declaration. Exit code 0 when no text is flagged, 1 when one is (each one is printed with the held-out question),
2 when the bank cannot be read as declared.
"""

import sys
from collections.abc import Sequence, Set
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import httpx2

from evalharness.datasets.paraphrase_set import load_paraphrase_set
from evalharness.datasets.retrieval_gold import GoldSetError, build_gold_set, load_declaration
from evalharness.evaluation.retrieval import RunRefused, bank_folder
from evalharness.settings import HarnessSettings, StackSettings, load_stack_settings

type Vector = Sequence[float]

# How many texts go to the embedding server in one request.
_BATCH = 32


@dataclass(frozen=True)
class Close:
    """A text of the pair set and the held-out question it is closest to, above the line."""

    label: str
    held_out_id: str
    similarity: float


def similarity(first: Vector, second: Vector) -> float:
    """Return the cosine similarity of two vectors the server returned normalised to length 1 (their dot product)."""
    return sum(a * b for a, b in zip(first, second, strict=True))


def different_question_line(questions: Sequence[tuple[str, Vector]]) -> float:
    """Return the highest similarity between two questions (chapter, vector) from different chapters.

    Questions from different chapters of the course ask about different things, so no pair of them is "the same
    question"; the highest similarity among them is how close two different questions can get with this model.
    """
    values = [similarity(a, b) for (ca, a), (cb, b) in combinations(questions, 2) if ca != cb]
    if not values:
        raise ValueError("no two questions from different chapters: the line cannot be measured")
    return max(values)


def find_close(
    texts: Sequence[tuple[str, Vector]],
    held_out: Sequence[tuple[str, Vector]],
    line: float,
    cleared: Set[tuple[str, str]] = frozenset(),
) -> list[Close]:
    """Return every text (label, vector) closer than ``line`` to a held-out question (id, vector), closest first.

    A pair (label, held-out id) in ``cleared`` was read and found to ask something else, so it is not returned.
    """
    found = [
        Close(label=label, held_out_id=question_id, similarity=value)
        for label, vector in texts
        for question_id, held in held_out
        if (value := similarity(vector, held)) > line and (label, question_id) not in cleared
    ]
    return sorted(found, key=lambda close: close.similarity, reverse=True)


def embed(texts: Sequence[str], stack: StackSettings) -> list[list[float]]:
    """Embed ``texts`` as questions on the stack's embedding server, with the prefix the app puts before a question."""
    url = f"http://127.0.0.1:{stack.embedder_published_port}/embed"
    vectors: list[list[float]] = []
    with httpx2.Client(timeout=60) as client:
        for start in range(0, len(texts), _BATCH):
            batch = [stack.app_embedding__query_prefix + text for text in texts[start : start + _BATCH]]
            response = client.post(url, json={"inputs": batch})
            response.raise_for_status()
            vectors += response.json()
    return vectors


def main(argv: Sequence[str]) -> int:
    """Check the pair file against the held-out questions of the gold declaration named in ``argv``."""
    if len(argv) != 2:
        print("usage: python -m evalharness.datasets.check_paraphrases <pair_file> <gold_declaration>", file=sys.stderr)
        return 2
    pair_set = load_paraphrase_set(Path(argv[0]))
    pairs = pair_set.pairs
    declaration = load_declaration(Path(argv[1]))
    settings = HarnessSettings()
    stack = load_stack_settings(settings.stack_env_file)
    try:
        folder = bank_folder(declaration, stack.local_course_id, stack.local_course_sources, settings.stack_env_file)
        gold = build_gold_set(declaration, folder)
    except (RunRefused, GoldSetError) as error:
        print(f"cannot read the bank: {error}", file=sys.stderr)
        return 2
    held_out = [case for case in gold.cases if case.held_out]
    labels = [f"{p.case_id} question" for p in pairs] + [f"{p.case_id} rewording" for p in pairs]
    vectors = embed([p.question for p in pairs] + [p.paraphrase for p in pairs] + [c.query for c in held_out], stack)
    texts = list(zip(labels, vectors[: len(labels)], strict=True))
    held = [(case.case_id, vector) for case, vector in zip(held_out, vectors[len(labels) :], strict=True)]
    chapters = [p.gold_section.split(".", 1)[0] for p in pairs]
    line = different_question_line(list(zip(chapters, vectors[: len(pairs)], strict=True)))
    close = find_close(texts, held, line, {(c.text, c.held_out_id) for c in pair_set.cleared})
    queries = {case.case_id: case.query for case in held_out}
    text_of = dict(zip(labels, [p.question for p in pairs] + [p.paraphrase for p in pairs], strict=True))
    for item in close:
        print(f"{item.similarity:.3f}  {item.label}: {text_of[item.label]}")
        print(f"       {item.held_out_id}: {' '.join(queries[item.held_out_id].split())}")
    for cleared in pair_set.cleared:
        print(f"cleared by reading: {cleared.text} / {cleared.held_out_id}: {cleared.reason}")
    print(
        f"{len(texts)} texts against {len(held)} held-out questions; line {line:.3f} (the highest similarity of two "
        f"questions of the set from different chapters): {len(close)} above it, {len(pair_set.cleared)} cleared"
    )
    return 1 if close else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
