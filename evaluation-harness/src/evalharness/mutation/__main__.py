"""Command that runs the mutants of a catalog against the running stack, one at a time.

    poetry run python -m evalharness.mutation mutants/<course>.toml
    poetry run python -m evalharness.mutation mutants/<course>.toml --mutants lowest_score_0_88

Run it from evaluation-harness/, like the tests, with the stack running and serving the catalog's course as its local
course. The catalog's baseline runs first and once more at the end, around the chosen mutants: the first is what every
verdict is read against, the last shows the false alarms of each check. Every mutant run starts the app's container
again twice (with the mutant, then as configured), ingests the course twice when the mutant changes how it is cut into
passages, and writes a folder under runs/mutants/ with its record and the records of its checks. The command prints
the experiment id that builds the matrix:

    poetry run python -m evalharness.mutation.matrix <experiment id>

Exit code 0 when every mutant ran, valid or not; 2 when the command was refused before the app was changed; 1 when
the app did not come back after a mutant, so the stack needs a look before anything else runs on it.
"""

import argparse
import asyncio
import logging
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

import structlog

from evalharness.datasets.retrieval_gold import GoldSetError
from evalharness.evaluation.retrieval import RunRefused, check_ready, read_app_settings
from evalharness.mutation.catalog import CatalogError, CheckName, LoadedCatalog, load_catalog
from evalharness.mutation.course import AdminIngestedCourse, StoredCourse
from evalharness.mutation.oracles import Check, build_checks
from evalharness.mutation.record import MutantRunRecord
from evalharness.mutation.runner import Experiment, RestoreFailed, run_mutants
from evalharness.mutation.stack import ComposeStack, StackError
from evalharness.settings import HarnessSettings, load_stack_settings
from evalharness.transport import SutTransport

RUNS_DIR = Path("runs/mutants")


def main(argv: Sequence[str]) -> int:
    """Run the command and return its exit code."""
    parser = argparse.ArgumentParser(prog="python -m evalharness.mutation")
    parser.add_argument("catalog", type=Path, help="the mutant catalog file of a course")
    parser.add_argument("--mutants", nargs="+", help="ids of the mutants to run, in this order (default: all)")
    args = parser.parse_args(argv)

    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))
    settings = HarnessSettings()
    stack = load_stack_settings(settings.stack_env_file)
    try:
        loaded = load_catalog(args.catalog)
    except CatalogError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    catalog = loaded.catalog
    baseline = catalog.baseline_id()
    chosen = args.mutants or list(catalog.mutants)
    unknown = [mutant_id for mutant_id in chosen if mutant_id not in catalog.mutants]
    if unknown:
        parser.error(f"not in the catalog: {unknown}")
    ids = [baseline, *(mutant_id for mutant_id in chosen if mutant_id != baseline), baseline]
    if stack.local_course_id != catalog.course_id:
        print(
            f"refused: the catalog is for {catalog.course_id!r}, the stack's local course is {stack.local_course_id!r}",
            file=sys.stderr,
        )
        return 2
    commit, dirty = _harness_version()
    try:
        # The dataset loaders say what is wrong with a file by a ValueError; only they are read here.
        checks = build_checks(catalog, settings, stack, commit, dirty)
    except (RunRefused, GoldSetError, ValueError, FileNotFoundError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    course = AdminIngestedCourse(settings, stack, catalog.course_id, stack.admin_api_key)
    try:
        return asyncio.run(_run(args.catalog, ids, checks, settings, course, commit, dirty, loaded))
    except (RunRefused, StackError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2


async def _run(
    catalog_file: Path,
    ids: Sequence[str],
    checks: Mapping[CheckName, Check],
    settings: HarnessSettings,
    course: StoredCourse,
    commit: str,
    dirty: bool,
    loaded: LoadedCatalog,
) -> int:
    """Read the app's first state and the course's stored passages, find the stack, then run the mutants one after
    the other."""
    async with SutTransport(settings) as sut:
        first = await read_app_settings(sut)
        await check_ready(sut)
        git_commit = first.get("git_commit")
        stack = ComposeStack.discover(settings.stack_env_file, git_commit if isinstance(git_commit, str) else "")
        experiment_id = uuid.uuid4().hex
        experiment = Experiment(
            experiment_id, loaded, catalog_file.as_posix(), first, course.snapshot(), commit, dirty, RUNS_DIR
        )
        print(f"experiment {experiment_id}: {len(ids)} run(s); each starts the app twice", flush=True)
        try:
            await run_mutants(ids, experiment, stack, course, sut, checks, _print_run)
        except RestoreFailed as error:
            print(f"stopped: {error}", file=sys.stderr)
            return 1
    print(f"matrix: python -m evalharness.mutation.matrix {experiment_id}")
    return 0


def _print_run(record: MutantRunRecord) -> None:
    """Print one line per mutant run: valid or why not, then how each check went."""
    problem = record.problem()
    print(f"{record.mutant_id}: {'valid' if problem is None else 'INVALID: ' + problem}", flush=True)
    for name, result in record.checks.items():
        if result.tests is not None:
            outcome = ", ".join(f"{test}: {value}" for test, value in result.tests.items())
        elif result.status == "refused":
            outcome = f"refused ({result.refusal})"
        else:
            outcome = f"ran, {result.result_file}"
        print(f"  {name}: {outcome}")


def _harness_version() -> tuple[str, bool]:
    """Return the git commit of the checkout and whether the harness folder has changes not yet committed."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--", "."], capture_output=True, text=True, check=True)
    return commit, bool(status.stdout.strip())


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
