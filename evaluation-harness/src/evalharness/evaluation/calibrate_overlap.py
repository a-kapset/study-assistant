"""Calibrate the overlap a pair must reach to count as stable (the top-k stable relation's ``min_overlap``).

The value is read from two groups of searches with known answers, both made like the relation's diagnostic pass
(the app's number of hits, no lowest score):

- the same question with no change of meaning at all: each question against itself in lower case without
  punctuation. Their overlap shows how much the top k moves on a change that should move nothing;
- different questions on a nearby topic: each question against the next question of the set from the same chapter
  but another section. Their overlap shows how many passages two different questions share by topic alone.

The calibrated value is the smallest overlap that no pair of different questions reached: a pair at or above it
shares more than topic alone gives. The command also prints how many same-question pairs reach it, which says
whether the value still lets a meaning-preserving change count as stable.

    poetry run python -m evalharness.evaluation.calibrate_overlap datasets/<course>/paraphrase_v0.toml

Run it from evaluation-harness/ on the stack with the course ingested; no model is called and nothing is written.
"""

import asyncio
import logging
import re
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import structlog

from evalharness.datasets.paraphrase_set import ParaphrasePair, load_paraphrase_set
from evalharness.evaluation.retrieval import search
from evalharness.settings import HarnessSettings
from evalharness.transport import SutTransport


def unchanged_meaning(text: str) -> str:
    """Return ``text`` in lower case without punctuation: the same question, in other characters only."""
    return re.sub(r"[^\w\s-]", "", text).lower()


def nearby_different(pairs: Sequence[ParaphrasePair]) -> list[tuple[ParaphrasePair, ParaphrasePair]]:
    """Pair every question with the next question of the set (going round) from the same chapter and another section.

    A question with no such partner is left out. Sections are second-level, so "4.2" and "4.3" are neighbours.
    """
    found = []
    for i, pair in enumerate(pairs):
        chapter = pair.gold_section.split(".", 1)[0]
        others = [*pairs[i + 1 :], *pairs[:i]]
        partner = next(
            (o for o in others if o.gold_section.split(".", 1)[0] == chapter and o.gold_section != pair.gold_section),
            None,
        )
        if partner is not None:
            found.append((pair, partner))
    return found


def calibrated_min_overlap(different: Sequence[int], k: int) -> float:
    """Return the smallest overlap, as a share of k, that no pair of different questions reached.

    ``different`` holds the shared passage counts of the pairs of different questions. When even k shared passages
    occur, no value separates them and the result is 1.0, the strictest rule.
    """
    if not different:
        raise ValueError("no pairs of different questions: the overlap cannot be calibrated")
    return min(max(different) + 1, k) / k


async def _shared(sut: SutTransport, course_id: str, first: str, second: str) -> tuple[int, int]:
    """Search two texts with no lowest score and return how many passages they share, and the k the app used."""
    results = []
    for text in (first, second):
        searched = await search(sut, course_id, text, {"min_score": -1})
        if searched.result is None:
            raise RuntimeError(f"a search failed ({searched.error}); the calibration needs every search")
        results.append(searched.result)
    return len({h.chunk_id for h in results[0].hits} & {h.chunk_id for h in results[1].hits}), results[0].k


async def _calibrate(pair_file: Path, settings: HarnessSettings) -> None:
    """Run both groups of searches and print their distributions and the calibrated value."""
    pairs = load_paraphrase_set(pair_file)
    course = pairs.course_id
    async with SutTransport(settings) as sut:
        same = [await _shared(sut, course, p.question, unchanged_meaning(p.question)) for p in pairs.pairs]
        different = [await _shared(sut, course, p.question, o.question) for p, o in nearby_different(pairs.pairs)]
    k = same[0][1]
    same_counts = [shared for shared, _ in same]
    different_counts = [shared for shared, _ in different]
    value = calibrated_min_overlap(different_counts, k)
    reaching = sum(shared / k >= value for shared in same_counts)
    print(f"{course}, k={k}, searches with no lowest score; shared passages -> pairs")
    print(f"  same question, no change of meaning ({len(same_counts)}): {dict(sorted(Counter(same_counts).items()))}")
    print(
        f"  different questions, same chapter ({len(different_counts)}): "
        f"{dict(sorted(Counter(different_counts).items()))}"
    )
    print(
        f"calibrated min_overlap = {value:.1f} ({round(value * k)} of {k}): reached by no pair of different questions "
        f"and by {reaching} of {len(same_counts)} unchanged questions; the pair file declares "
        f"{pairs.rules.min_overlap}"
    )


def main(argv: Sequence[str]) -> int:
    """Run the calibration for the pair file named in ``argv``; exit code 0 when it ran."""
    if len(argv) != 1:
        print("usage: python -m evalharness.evaluation.calibrate_overlap <pair_file>", file=sys.stderr)
        return 2
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))
    asyncio.run(_calibrate(Path(argv[0]), HarnessSettings()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
