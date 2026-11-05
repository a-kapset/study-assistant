"""Command that compares two recorded retrieval runs of the same gold set, question by question.

    poetry run python -m evalharness.evaluation.compare <before.json> <after.json> --factor corpus
    poetry run python -m evalharness.evaluation.compare <before.json> <after.json>

Run it from evaluation-harness/, like the tests. --factor names the one recorded value that was changed on purpose
(corpus, embedding_space, k, min_score); leave it out for a repeated run or a change of the app's code, which a
record does not hold. The runs are compared only when they measured the same questions and the recorded values
differ in exactly that factor: when two things changed at once, a difference cannot be put down to either.
Exit code 0 when the runs were compared, whatever the result; 2 when the comparison was refused.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from evalharness.datasets.retrieval_gold import load_declaration
from evalharness.evaluation.record import CaseRecord, RunRecord
from evalharness.stats.paired import Change, change_verdict, flip_count, paired_change

type Factor = Literal["corpus", "embedding_space", "k", "min_score"]

# The run values a comparison checks, and the factor name each one goes by on the command line.
_FACTOR_FIELDS: dict[Factor, str] = {
    "corpus": "corpus_snapshot",
    "embedding_space": "embedding_space",
    "k": "k",
    "min_score": "min_score",
}


class NotComparable(ValueError):
    """The two runs cannot be compared; the message says which values stand in the way."""


def main(argv: Sequence[str]) -> int:
    """Run the command and return its exit code."""
    parser = argparse.ArgumentParser(prog="python -m evalharness.evaluation.compare")
    parser.add_argument("before", type=Path, help="the record of the run before the change")
    parser.add_argument("after", type=Path, help="the record of the run after the change")
    parser.add_argument("--factor", choices=list(_FACTOR_FIELDS), help="the one recorded value changed on purpose")
    args = parser.parse_args(argv)

    before = RunRecord.model_validate_json(args.before.read_bytes())
    after = RunRecord.model_validate_json(args.after.read_bytes())
    try:
        check_comparable(before, after, args.factor)
        min_meaningful = _min_meaningful_change(before)
    except NotComparable as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    _print_comparison(before, after, args.factor, min_meaningful)
    return 0


def check_comparable(before: RunRecord, after: RunRecord, factor: Factor | None) -> None:
    """Refuse two runs that did not measure the same questions, or whose recorded values differ in anything but
    the declared factor (or in anything at all when no factor is declared)."""
    for field in ("gold_sha256", "course_id", "sets"):
        if getattr(before, field) != getattr(after, field):
            raise NotComparable(f"{field} differs: {getattr(before, field)} / {getattr(after, field)}")
    differing = {name for name, field in _FACTOR_FIELDS.items() if getattr(before, field) != getattr(after, field)}
    expected = {factor} if factor else set()
    if differing != expected:
        declared = factor or "nothing"
        found = ", ".join(sorted(differing)) or "nothing"
        raise NotComparable(f"declared change: {declared}; recorded values that differ: {found}")


def _min_meaningful_change(record: RunRecord) -> float:
    """Read the smallest change worth acting on from the run's gold declaration; refuse when none is declared."""
    comparison = load_declaration(Path(record.declaration_file)).comparison
    if comparison is None:
        raise NotComparable(f"{record.declaration_file} declares no [comparison] min_meaningful_change")
    return comparison.min_meaningful_change


def _print_comparison(before: RunRecord, after: RunRecord, factor: Factor | None, min_meaningful: float) -> None:
    """Print what changed, the change of recall with its interval, flips and verdict, and the change of MRR."""
    print(f"compare {before.run_id[:8]} -> {after.run_id[:8]}  {before.course_id}  sets {' '.join(before.sets)}")
    if factor is None:
        print("declared change: none of the recorded values (a repeated run or a change of the app's code)")
    else:
        field = _FACTOR_FIELDS[factor]
        print(f"declared change: {factor} ({_short(getattr(before, field))} -> {_short(getattr(after, field))})")

    old_cases, new_cases = _cases(before), _cases(after)
    ids = sorted(old_cases)
    old_hits = [_hit(old_cases[case_id]) for case_id in ids]
    new_hits = [_hit(new_cases[case_id]) for case_id in ids]
    recall = paired_change([float(hit) for hit in old_hits], [float(hit) for hit in new_hits])
    flips = flip_count(old_hits, new_hits)
    p_value = f"{flips.p_value:.3f}" if flips.p_value is not None else "none (no question flipped)"
    print(
        f"\nrecall@{before.k} {_mean(old_hits):.3f} -> {_mean(new_hits):.3f}, "
        f"change {recall.change:+.3f}, {_range(recall)}"
    )
    print(f"  questions gained {flips.gained}, lost {flips.lost}; McNemar exact p = {p_value}")
    gained = [case_id for case_id, old, new in zip(ids, old_hits, new_hits, strict=True) if new and not old]
    lost = [case_id for case_id, old, new in zip(ids, old_hits, new_hits, strict=True) if old and not new]
    print(f"  gained: {' '.join(gained) or '-'}")
    print(f"  lost:   {' '.join(lost) or '-'}")
    print(f"  smallest change worth acting on {min_meaningful}: {change_verdict(recall, min_meaningful)}")

    old_ranks = [_rank(old_cases[case_id]) for case_id in ids]
    new_ranks = [_rank(new_cases[case_id]) for case_id in ids]
    mrr = paired_change(old_ranks, new_ranks)
    print(
        f"\nMRR {_mean(old_ranks):.3f} -> {_mean(new_ranks):.3f}, change {mrr.change:+.3f}, "
        f"{_range(mrr)}; no target, so no verdict"
    )


def _cases(record: RunRecord) -> dict[str, CaseRecord]:
    """Index a run's questions by their id; the gold-set hash already guarantees both runs have the same ids."""
    return {case.case_id: case for case in record.cases}


def _hit(case: CaseRecord) -> bool:
    """Whether the search a user sends found the answer section; a failed search counts as not found."""
    return case.passes["primary"].score.verdict == "hit"


def _rank(case: CaseRecord) -> float:
    """The reciprocal rank of the first passage of the answer section in the search a user sends."""
    return case.passes["primary"].score.reciprocal_rank


def _mean(values: Sequence[float]) -> float:
    """The mean of the values; True counts as 1 and False as 0, so for hits it is the recall."""
    return sum(values) / len(values)


def _range(change: Change) -> str:
    """Write the interval of a change with signs, so a range below zero reads clearly."""
    if change.interval is None:
        return "no interval: every question changed by the same amount"
    low, high = change.interval.low, change.interval.high
    return f"95% CI {low:+.3f} to {high:+.3f} (bootstrap over the per-question differences)"


def _short(value: object) -> str:
    """Write a recorded value; a corpus snapshot hash is cut to its first 12 characters, enough to tell two apart."""
    text = str(value)
    return text[:12] if len(text) == 64 else text


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
