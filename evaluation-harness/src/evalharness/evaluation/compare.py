"""Command that compares two recorded runs of the same questions, question by question.

    poetry run python -m evalharness.evaluation.compare <before.json> <after.json> --factor corpus
    poetry run python -m evalharness.evaluation.compare <before.json> <after.json> --factor strategy --planned 4
    poetry run python -m evalharness.evaluation.compare <before.json> <after.json>

Run it from evaluation-harness/, like the tests. Both records are retrieval runs of one gold set, or paraphrase runs
of one pair set; a paraphrase run is compared on its rewordings. --factor names the one recorded value that was
changed on purpose (corpus, embedding_space, k, min_score, strategy, rerank, or app for the app's code); leave it out
for a repeated run. The runs are compared only when they measured the same questions and the recorded values differ
in exactly that factor: when two things changed at once, a difference cannot be put down to either. The app's code
is told by the commit in the recorded app settings, so a run whose app commit cannot be told is refused: one
recorded before runs kept the app settings, one built without GIT_COMMIT ("unknown"), and two runs of the same
commit with uncommitted changes ("-dirty"), which may hold different code. A run with a failed search is refused
too: that question measured nothing, and counting it as not found would show a broken stack as worse retrieval. So
is a run with a search the app answered on a weaker path because a model server was down: it measured that path,
not the configured one. --planned says how many comparisons were planned together for one decision; the interval
the verdict reads is then widened so that all of them together go wrong no more often than one alone would. Exit
code 0 when the runs were compared, whatever the result; 2 when the comparison was refused.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from evalharness.datasets.paraphrase_set import load_paraphrase_set
from evalharness.datasets.retrieval_gold import load_declaration
from evalharness.evaluation.paraphrase_record import ParaphraseRunRecord
from evalharness.evaluation.record import PassResult, RunRecord
from evalharness.evaluation.retrieval import weaker_path_counts
from evalharness.stats.intervals import CONFIDENCE
from evalharness.stats.paired import Change, change_verdict, flip_count, paired_change, planned_confidence

type Factor = Literal["corpus", "embedding_space", "k", "min_score", "strategy", "rerank", "app"]
type AnyRecord = RunRecord | ParaphraseRunRecord

# The run values a comparison checks, and the factor name each one goes by on the command line. The app's code has
# no field of its own: it is the commit inside the recorded app settings.
_FACTOR_FIELDS: dict[Factor, str] = {
    "corpus": "corpus_snapshot",
    "embedding_space": "embedding_space",
    "k": "k",
    "min_score": "min_score",
    "strategy": "strategy",
    "rerank": "reranked",
}
_FACTORS: tuple[Factor, ...] = (*_FACTOR_FIELDS, "app")

# The build marks a commit with this ending when app/ had changes not yet committed.
_DIRTY = "-dirty"


class NotComparable(ValueError):
    """The two runs cannot be compared; the message says which values stand in the way."""


def main(argv: Sequence[str]) -> int:
    """Run the command and return its exit code."""
    parser = argparse.ArgumentParser(prog="python -m evalharness.evaluation.compare")
    parser.add_argument("before", type=Path, help="the record of the run before the change")
    parser.add_argument("after", type=Path, help="the record of the run after the change")
    parser.add_argument("--factor", choices=list(_FACTORS), help="the one recorded value changed on purpose")
    parser.add_argument(
        "--planned",
        type=int,
        default=1,
        help="how many comparisons were planned together for one decision (default 1); widens the verdict's interval",
    )
    args = parser.parse_args(argv)
    if args.planned < 1:
        parser.error("--planned counts the comparison being made, so it is at least 1")

    before, after = load_record(args.before), load_record(args.after)
    try:
        check_comparable(before, after, args.factor)
        min_meaningful = _min_meaningful_change(before)
    except NotComparable as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    _print_comparison(before, after, args.factor, min_meaningful, args.planned)
    return 0


def load_record(path: Path) -> AnyRecord:
    """Read a run record from its file; a paraphrase record is told from a retrieval one by its list of pairs."""
    content = path.read_bytes()
    if "pairs" in json.loads(content):
        return ParaphraseRunRecord.model_validate_json(content)
    return RunRecord.model_validate_json(content)


def check_comparable(before: AnyRecord, after: AnyRecord, factor: Factor | None) -> None:
    """Refuse two runs that did not measure the same questions, that did not measure every one of them on the
    configured search, or whose recorded values differ in anything but the declared factor (or in anything at all
    when no factor is declared)."""
    if type(before) is not type(after):
        raise NotComparable("one record is a retrieval run, the other a paraphrase run")
    same_questions = (
        ("gold_sha256", "course_id", "sets") if isinstance(before, RunRecord) else ("set_sha256", "course_id")
    )
    for field in same_questions:
        if getattr(before, field) != getattr(after, field):
            raise NotComparable(f"{field} differs: {getattr(before, field)} / {getattr(after, field)}")
    for record in (before, after):
        searches = primary_searches(record)
        failed = sum(search.score.verdict == "failed" for search in searches)
        if failed:
            raise NotComparable(
                f"run {record.run_id[:8]} has {failed} failed search(es), which measured nothing; repeat it"
            )
        weaker = weaker_path_counts(searches)
        if weaker:
            raise NotComparable(
                f"run {record.run_id[:8]} has searches answered on a weaker path ({weaker}), "
                "which did not measure the configured search; repeat it"
            )
    before_commit, after_commit = app_commit(before), app_commit(after)
    if before_commit == after_commit and before_commit.endswith(_DIRTY):
        raise NotComparable(
            f"both runs are app {before_commit}: uncommitted changes of one commit may hold other code; "
            "commit the app's changes, rebuild and repeat both runs"
        )
    differing: set[Factor] = {
        name for name, field in _FACTOR_FIELDS.items() if getattr(before, field) != getattr(after, field)
    }
    if before_commit != after_commit:
        differing.add("app")
    expected = {factor} if factor else set()
    if differing != expected:
        declared = factor or "nothing"
        found = ", ".join(sorted(differing)) or "nothing"
        raise NotComparable(f"declared change: {declared}; recorded values that differ: {found}")


def primary_searches(record: AnyRecord) -> list[PassResult]:
    """Return every search of the pass a user's search is measured by: one per question of a retrieval run, the
    question and the rewording of every pair of a paraphrase run."""
    if isinstance(record, RunRecord):
        return [case.passes["primary"] for case in record.cases]
    return [
        side for pair in record.pairs for side in (pair.passes["primary"].question, pair.passes["primary"].rewording)
    ]


def found_by_id(record: AnyRecord) -> dict[str, bool]:
    """Return, per question id, whether the search a user sends found the answer section: the compared result.

    A retrieval run is compared on its questions. A paraphrase run is compared on its rewordings: they ask without
    the course's own term, which is the kind of question a change of the search is meant to help, while the
    questions of the pairs keep that term.
    """
    if isinstance(record, RunRecord):
        return {case.case_id: case.passes["primary"].score.verdict == "hit" for case in record.cases}
    return {pair.case_id: pair.passes["primary"].rewording.score.verdict == "hit" for pair in record.pairs}


def app_commit(record: AnyRecord) -> str:
    """Return the commit of the app's code a run measured; refuse a run whose app commit cannot be told."""
    if record.app_settings is None:
        raise NotComparable(f"run {record.run_id[:8]} was recorded without the app's settings; repeat it")
    commit = record.app_settings.get("git_commit")
    if not isinstance(commit, str) or commit in ("", "unknown"):
        raise NotComparable(
            f"run {record.run_id[:8]} has no app commit ({commit!r}): "
            "build the app with GIT_COMMIT set, see app/README.md"
        )
    return commit


def _min_meaningful_change(record: AnyRecord) -> float:
    """Read the smallest change worth acting on from the file that declares the run's questions: the gold
    declaration of a retrieval run (refused when it declares none), the pair file of a paraphrase run."""
    if isinstance(record, ParaphraseRunRecord):
        return load_paraphrase_set(Path(record.pair_file)).rules.min_meaningful_change
    comparison = load_declaration(Path(record.declaration_file)).comparison
    if comparison is None:
        raise NotComparable(f"{record.declaration_file} declares no [comparison] min_meaningful_change")
    return comparison.min_meaningful_change


def _print_comparison(
    before: AnyRecord, after: AnyRecord, factor: Factor | None, min_meaningful: float, planned: int
) -> None:
    """Print what changed, the change of the compared result with its interval, flips and verdict, and one more
    number without a verdict: MRR for retrieval runs, the questions' own recall for paraphrase runs."""
    measured = f"sets {' '.join(before.sets)}" if isinstance(before, RunRecord) else f"{len(before.pairs)} pairs"
    print(f"compare {before.run_id[:8]} -> {after.run_id[:8]}  {before.course_id}  {measured}")
    if factor is None:
        print(f"declared change: none of the recorded values (a repeated run of app {_commit(app_commit(before))})")
    elif factor == "app":
        print(f"declared change: app ({_commit(app_commit(before))} -> {_commit(app_commit(after))})")
    else:
        field = _FACTOR_FIELDS[factor]
        print(f"declared change: {factor} ({_short(getattr(before, field))} -> {_short(getattr(after, field))})")
    confidence = planned_confidence(planned)
    if planned > 1:
        print(f"one of {planned} planned comparisons: the verdict reads a {_level(confidence)} interval")

    old_found, new_found = found_by_id(before), found_by_id(after)
    ids = sorted(old_found)
    old_hits = [old_found[case_id] for case_id in ids]
    new_hits = [new_found[case_id] for case_id in ids]
    recall = paired_change([float(hit) for hit in old_hits], [float(hit) for hit in new_hits], confidence)
    flips = flip_count(old_hits, new_hits)
    p_value = f"{flips.p_value:.3f}" if flips.p_value is not None else "none (no question flipped)"
    name = f"recall@{before.k}" if isinstance(before, RunRecord) else f"recall@{before.k} of the rewordings"
    print(
        f"\n{name} {_mean(old_hits):.3f} -> {_mean(new_hits):.3f}, "
        f"change {recall.change:+.3f}, {_range(recall, confidence)}"
    )
    print(f"  questions gained {flips.gained}, lost {flips.lost}; McNemar exact p = {p_value}")
    gained = [case_id for case_id, old, new in zip(ids, old_hits, new_hits, strict=True) if new and not old]
    lost = [case_id for case_id, old, new in zip(ids, old_hits, new_hits, strict=True) if old and not new]
    print(f"  gained: {' '.join(gained) or '-'}")
    print(f"  lost:   {' '.join(lost) or '-'}")
    print(f"  smallest change worth acting on {min_meaningful}: {change_verdict(recall, min_meaningful)}")

    label, old_scores, new_scores = _second_number(before, after, ids)
    second = paired_change(old_scores, new_scores)
    print(
        f"\n{label} {_mean(old_scores):.3f} -> {_mean(new_scores):.3f}, change {second.change:+.3f}, "
        f"{_range(second, CONFIDENCE)}; no target, so no verdict"
    )


def _second_number(before: AnyRecord, after: AnyRecord, ids: Sequence[str]) -> tuple[str, list[float], list[float]]:
    """Return the name and the per-question scores, before and after, of the number shown without a verdict.

    For retrieval runs it is the reciprocal rank of the first passage of the answer section (MRR). For paraphrase
    runs it is whether the question itself, with the course's term, found its answer: a change of the search that
    helps the rewordings should not cost the questions.
    """
    if isinstance(before, RunRecord) and isinstance(after, RunRecord):
        old_ranks = {case.case_id: case.passes["primary"].score.reciprocal_rank for case in before.cases}
        new_ranks = {case.case_id: case.passes["primary"].score.reciprocal_rank for case in after.cases}
        return "MRR", [old_ranks[case_id] for case_id in ids], [new_ranks[case_id] for case_id in ids]
    if isinstance(before, ParaphraseRunRecord) and isinstance(after, ParaphraseRunRecord):
        old_asked = {pair.case_id: pair.passes["primary"].question.score.verdict == "hit" for pair in before.pairs}
        new_asked = {pair.case_id: pair.passes["primary"].question.score.verdict == "hit" for pair in after.pairs}
        name = f"recall@{before.k} of the questions"
        return name, [float(old_asked[case_id]) for case_id in ids], [float(new_asked[case_id]) for case_id in ids]
    raise NotComparable("one record is a retrieval run, the other a paraphrase run")


def _mean(values: Sequence[float]) -> float:
    """The mean of the values; True counts as 1 and False as 0, so for hits it is the recall."""
    return sum(values) / len(values)


def _level(confidence: float) -> str:
    """Write the level of an interval as a percentage without needless zeros: "95%", "98.75%"."""
    return f"{round(confidence * 100, 4):g}%"


def _range(change: Change, confidence: float) -> str:
    """Write the interval of a change with its level and with signs, so a range below zero reads clearly."""
    if change.interval is None:
        return "no interval: every question changed by the same amount"
    low, high = change.interval.low, change.interval.high
    return f"{_level(confidence)} CI {low:+.3f} to {high:+.3f} (bootstrap over the per-question differences)"


def _short(value: object) -> str:
    """Write a recorded value: a corpus snapshot hash cut to its first 12 characters, enough to tell two apart, and
    the reranker's flag as on / off."""
    if isinstance(value, bool):
        return "on" if value else "off"
    text = str(value)
    return text[:12] if len(text) == 64 else text


def _commit(commit: str) -> str:
    """Write an app commit cut to its first 12 characters, keeping the mark of uncommitted changes."""
    base = commit.removesuffix(_DIRTY)
    return base[:12] + commit[len(base) :]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
