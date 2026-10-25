"""Command for a retrieval run of a gold set against the running stack.

    poetry run python -m evalharness.evaluation datasets/<course>/<gold set>.toml
    poetry run python -m evalharness.evaluation <declaration> --sets D --include-held-out

Run it from evaluation-harness/, like the tests. Without --sets the run takes the sets of the declared main result.
The run is written to runs/ as a new JSON file and its summary is printed. Exit code 0 when the run completed,
whatever its numbers; 2 when it was refused.
"""

import argparse
import asyncio
import logging
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import structlog

from evalharness.corpus_db import CorpusReader
from evalharness.datasets.retrieval_gold import GoldCase, GoldSet, GoldSetError, build_gold_set, load_declaration
from evalharness.evaluation.record import PassName, RunRecord, recorded_held_out_uses, write_record
from evalharness.evaluation.retrieval import RunRefused, bank_folder, run, select_cases
from evalharness.settings import HarnessSettings, StackSettings, load_stack_settings
from evalharness.transport import SutTransport

RUNS_DIR = Path("runs")


def main(argv: Sequence[str]) -> int:
    """Run the command and return its exit code."""
    parser = argparse.ArgumentParser(prog="python -m evalharness.evaluation")
    parser.add_argument("declaration", type=Path, help="the gold set's declaration file")
    parser.add_argument("--sets", nargs="+", help="exam sets to run (default: the sets of the declared main result)")
    parser.add_argument("--include-held-out", action="store_true", help="allow held-out sets in --sets")
    args = parser.parse_args(argv)

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
            f"  {'slice':<12}{'questions':>10}{'recall':>8}{'MRR':>7}{'precision':>11}{'irrelevant':>12}"
            f"{'hit':>6}{'miss':>6}{'empty':>6}{'fail':>6}"
        )
        table = record.summaries[name]
        for key in ["all", *sorted(key for key in table if key != "all")]:
            s = table[key]
            hits = s.n - s.misses - s.empty - s.failed
            print(
                f"  {key:<12}{s.n:>10}{s.recall:>8.3f}{s.mrr:>7.3f}{s.precision:>11.3f}{s.irrelevant:>12.3f}"
                f"{hits:>6}{s.misses:>6}{s.empty:>6}{s.failed:>6}"
            )
    main_result = record.summaries["primary"]["all"]
    if record.sets == tuple(sorted(primary.sets)) and record.k == primary.k:
        verdict = "met" if main_result.recall >= primary.target else "below"
        print(f"\nrecall@{record.k} = {main_result.recall:.3f} (n={main_result.n}), target {primary.target}: {verdict}")
    else:
        print(f"\nno target line: the target is for sets {' '.join(primary.sets)} at k={primary.k}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
