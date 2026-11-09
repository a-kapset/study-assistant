"""Command for a retrieval run of a gold set against the running stack.

    poetry run python -m evalharness.evaluation datasets/<course>/<gold set>.toml
    poetry run python -m evalharness.evaluation <declaration> --sets D --include-held-out
    poetry run python -m evalharness.evaluation <declaration> --require-target

Run it from evaluation-harness/, like the tests. Without --sets the run takes the sets of the declared main result.
The run is written to runs/retrieval/ as a new JSON file and its summary is printed. Exit code 0 when the run completed,
whatever its numbers; 2 when it was refused. With --require-target (used in CI) a completed run exits with 1 unless
even the lower end of its main result's 95 % interval reaches the target.
"""

import argparse
import asyncio
import logging
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import structlog

from evalharness.corpus_db import CorpusReader
from evalharness.datasets.retrieval_gold import GoldCase, GoldSet, GoldSetError, build_gold_set, load_declaration
from evalharness.evaluation.record import PassName, RunRecord, recorded_held_out_uses, write_record
from evalharness.evaluation.retrieval import RunRefused, bank_folder, run, select_cases
from evalharness.oracles.retrieval import Summary
from evalharness.settings import HarnessSettings, StackSettings, load_stack_settings
from evalharness.stats.intervals import Interval, TargetVerdict, mean_interval, share_interval, share_verdict
from evalharness.transport import SutTransport

# Retrieval records only: the held-out uses are counted over every record in this folder.
RUNS_DIR = Path("runs/retrieval")


def main(argv: Sequence[str]) -> int:
    """Run the command and return its exit code."""
    parser = argparse.ArgumentParser(prog="python -m evalharness.evaluation")
    parser.add_argument("declaration", type=Path, help="the gold set's declaration file")
    parser.add_argument("--sets", nargs="+", help="exam sets to run (default: the sets of the declared main result)")
    parser.add_argument("--include-held-out", action="store_true", help="allow held-out sets in --sets")
    parser.add_argument(
        "--require-target", action="store_true", help="exit with 1 unless the main result meets its target (for CI)"
    )
    args = parser.parse_args(argv)
    if args.require_target and args.sets:
        parser.error("--require-target compares the declared sets with their target; leave out --sets")

    # One log line per request would bury the summary; failed requests still show as warnings.
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))
    settings = HarnessSettings()
    stack = load_stack_settings(settings.stack_env_file)
    declaration = load_declaration(args.declaration)
    try:
        folder = bank_folder(declaration, stack.local_course_id, stack.local_course_sources, settings.stack_env_file)
        gold = build_gold_set(declaration, folder)
        cases = select_cases(gold, args.sets or declaration.primary.sets, args.include_held_out)
        record = asyncio.run(_run(gold, cases, settings, stack, args.declaration))
    except (RunRefused, GoldSetError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    path = write_record(record, RUNS_DIR)
    _print_summary(record, gold, path)
    # Without a verdict (the app uses another k than the declared one) the target cannot be confirmed either.
    verdict = target_verdict(record, gold)
    if args.require_target and verdict != "met":
        print(f"the main result's target verdict is {verdict or 'missing'}, not met", file=sys.stderr)
        return 1
    return 0


async def _run(
    gold: GoldSet, cases: Sequence[GoldCase], settings: HarnessSettings, stack: StackSettings, declaration_file: Path
) -> RunRecord:
    """Make the run against the stack: searches over HTTP, the corpus read through the read-only role."""
    declaration = gold.declaration
    held_out_use = None
    if any(case.held_out for case in cases):
        recorded = recorded_held_out_uses(RUNS_DIR, declaration.course_id)
        held_out_use = declaration.held_out_uses_before_records + recorded + 1
    commit, dirty = _harness_version()
    with CorpusReader(settings, stack) as corpus:
        async with SutTransport(settings) as sut:
            return await run(
                gold,
                cases,
                sut,
                corpus,
                declaration_file=declaration_file.as_posix(),
                held_out_use=held_out_use,
                harness_commit=commit,
                harness_dirty=dirty,
            )


def _harness_version() -> tuple[str, bool]:
    """Return the git commit of the checkout and whether the harness folder has changes not yet committed."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--", "."], capture_output=True, text=True, check=True)
    return commit, bool(status.stdout.strip())


def _print_summary(record: RunRecord, gold: GoldSet, path: Path) -> None:
    """Print the run's settings, a table per pass, and the main result against its target."""
    primary = gold.declaration.primary
    dirty = " + uncommitted changes" if record.harness_dirty else ""
    sets = " ".join(record.sets)
    print(f"run {record.run_id[:8]}  {record.course_id}  sets {sets}  harness {record.harness_commit[:8]}{dirty}")
    if record.held_out_use is not None:
        print(f"held-out use #{record.held_out_use}")
    print(f"record: {path}")
    labels: dict[PassName, str] = {
        "primary": f"min_score={record.min_score}",
        "diagnostic": "no lowest score",
    }
    for name, label in labels.items():
        print(f"\n{name} pass (k={record.k}, {label})")
        print(
            f"  {'slice':<12}{'questions':>10}{'recall':>8}{'95% CI':>14}{'MRR':>7}{'precision':>11}{'irrelevant':>12}"
            f"{'hit':>6}{'miss':>6}{'empty':>6}{'fail':>6}"
        )
        table = record.summaries[name]
        for key in ["all", *sorted(key for key in table if key != "all")]:
            s = table[key]
            interval = _range(share_interval(_hits(s), _ran(s))) if _ran(s) else "n/a"
            print(
                f"  {key:<12}{s.n:>10}{_value(s.recall):>8}{interval:>14}{_value(s.mrr):>7}{_value(s.precision):>11}"
                f"{_value(s.irrelevant):>12}{_hits(s):>6}{s.misses:>6}{s.empty:>6}{s.failed:>6}"
            )
        reasons = Counter(case.passes[name].error for case in record.cases if case.passes[name].error is not None)
        if reasons:
            listed = ", ".join(f"{reason} {n}" for reason, n in sorted(reasons.items()))
            print(f"  failed searches, left out of the shares: {listed}")
    main_result = record.summaries["primary"]["all"]
    if main_result.failed:
        print(f"\ninvalid: {main_result.failed} search(es) of the primary pass failed; no target verdict")
        return
    ranks = [case.passes["primary"].score.reciprocal_rank for case in record.cases]
    mrr_interval = mean_interval(ranks)
    mrr_range = _range(mrr_interval) if mrr_interval is not None else "none (every question scored the same)"
    print(f"\nMRR = {_value(main_result.mrr)}, 95% CI {mrr_range} (bootstrap over questions)")
    verdict = target_verdict(record, gold)
    if verdict is not None:
        recall_range = _range(share_interval(_hits(main_result), main_result.n))
        print(
            f"recall@{record.k} = {_value(main_result.recall)}, 95% CI {recall_range} (Wilson, n={main_result.n}), "
            f"target {primary.target}: {verdict}"
        )
    else:
        print(f"no target line: the target is for sets {' '.join(primary.sets)} at k={primary.k}")


def target_verdict(record: RunRecord, gold: GoldSet) -> TargetVerdict | None:
    """Judge the main result against the declared target by its 95 % interval; None when the run is not the
    declared main result, or when a search of it failed.

    The target holds only for the declared sets at the declared k: recall at a larger k is easier to reach. A failed
    search measured nothing, so the run measured fewer questions than the target is set for.
    """
    primary = gold.declaration.primary
    if record.sets != tuple(sorted(primary.sets)) or record.k != primary.k:
        return None
    main_result = record.summaries["primary"]["all"]
    if main_result.failed:
        return None
    return share_verdict(_hits(main_result), main_result.n, primary.target)


def _hits(summary: Summary) -> int:
    """Count the questions of a group whose answer section was found: the ones that ended in no kind of failure."""
    return summary.n - summary.misses - summary.empty - summary.failed


def _ran(summary: Summary) -> int:
    """Count the questions of a group whose search ran, the ones the shares are over."""
    return summary.n - summary.failed


def _value(share: float | None) -> str:
    """Write a share with three decimals, or "n/a" when no search of the group ran."""
    return "n/a" if share is None else f"{share:.3f}"


def _range(interval: Interval) -> str:
    """Write an interval as "low-high" with three decimals, like the other numbers of the summary."""
    return f"{interval.low:.3f}-{interval.high:.3f}"


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
