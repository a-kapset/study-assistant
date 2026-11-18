"""The checks a mutant run makes, built once from the catalog and then run against every mutant.

Each check is an existing measurement of the harness, unchanged: the retrieval run of a gold set, the paraphrase run
of a pair file, and the two api tests of the relations on search. A check leaves its own record or result file in
the folder of the mutant run and reports how it went; whether that outcome means the mutant was noticed is decided
later, from the records.
"""

import asyncio
import os
import subprocess
import sys
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from xml.etree import ElementTree

from evalharness.corpus_db import CorpusReader
from evalharness.datasets.paraphrase_set import ParaphraseSet, load_paraphrase_set
from evalharness.datasets.retrieval_gold import GoldCase, GoldSet, build_gold_set, load_declaration
from evalharness.evaluation import paraphrase_record, paraphrase_run, retrieval
from evalharness.evaluation import record as retrieval_record
from evalharness.mutation.catalog import Catalog, CheckName
from evalharness.mutation.record import CheckResult, TestOutcome
from evalharness.settings import HarnessSettings, StackSettings
from evalharness.transport import SutTransport

# A check takes the folder of the mutant run, leaves its result there and says how it went.
type Check = Callable[[Path], Awaitable[CheckResult]]

# The api tests of the relations on the number of hits and the lowest score, relative to the harness folder.
RELATION_TESTS = Path("tests/api/test_search_relations.py")
# Hypothesis draws the same questions for every mutant, so two mutant runs differ in the app only.
_RELATION_SEED = 0


def build_checks(
    catalog: Catalog, settings: HarnessSettings, stack: StackSettings, harness_commit: str, harness_dirty: bool
) -> dict[CheckName, Check]:
    """Prepare every check the catalog makes, reading its files now: a gold set, pair file or test file that cannot
    be used stops the command before the app is changed. Raises what the dataset loaders raise."""
    checks: dict[CheckName, Check] = {}
    if catalog.checks.retrieval_gold is not None:
        declaration_file = catalog.checks.retrieval_gold
        declaration = load_declaration(declaration_file)
        folder = retrieval.bank_folder(
            declaration, stack.local_course_id, stack.local_course_sources, settings.stack_env_file
        )
        gold = build_gold_set(declaration, folder)
        cases = retrieval.select_cases(gold, declaration.primary.sets, include_held_out=False)
        checks["retrieval_gold"] = _retrieval_check(
            gold, cases, declaration_file, settings, stack, harness_commit, harness_dirty
        )
    if catalog.checks.paraphrase is not None:
        pair_file = catalog.checks.paraphrase
        pairs = load_paraphrase_set(pair_file)
        checks["paraphrase"] = _paraphrase_check(pairs, pair_file, settings, stack, harness_commit, harness_dirty)
    if catalog.checks.search_relations:
        if not RELATION_TESTS.is_file():
            raise FileNotFoundError(f"{RELATION_TESTS} not found; run the command from evaluation-harness/")
        checks["search_relations"] = relation_check(
            [sys.executable, "-m", "pytest", str(RELATION_TESTS), "-m", "api", "-p", "no:cacheprovider", "-q"]
        )
    return checks


def _retrieval_check(
    gold: GoldSet,
    cases: Sequence[GoldCase],
    declaration_file: Path,
    settings: HarnessSettings,
    stack: StackSettings,
    harness_commit: str,
    harness_dirty: bool,
) -> Check:
    """Return the retrieval run of the gold set's main result as a check; its record goes into the run's folder."""

    async def check(folder: Path) -> CheckResult:
        """Make the retrieval run against the app as it now is and keep its record."""
        with CorpusReader(settings, stack) as corpus:
            async with SutTransport(settings) as sut:
                record = await retrieval.run(
                    gold,
                    cases,
                    sut,
                    corpus,
                    declaration_file=declaration_file.as_posix(),
                    held_out_use=None,
                    harness_commit=harness_commit,
                    harness_dirty=harness_dirty,
                )
        return CheckResult(status="ran", result_file=retrieval_record.write_record(record, folder).name)

    return check


def _paraphrase_check(
    pairs: ParaphraseSet,
    pair_file: Path,
    settings: HarnessSettings,
    stack: StackSettings,
    harness_commit: str,
    harness_dirty: bool,
) -> Check:
    """Return the paraphrase run of the pair file as a check; its record goes into the run's folder."""

    async def check(folder: Path) -> CheckResult:
        """Make the paraphrase run against the app as it now is and keep its record."""
        with CorpusReader(settings, stack) as corpus:
            async with SutTransport(settings) as sut:
                record = await paraphrase_run.run(
                    pairs,
                    sut,
                    corpus,
                    pair_file=pair_file.as_posix(),
                    harness_commit=harness_commit,
                    harness_dirty=harness_dirty,
                )
        return CheckResult(status="ran", result_file=paraphrase_record.write_record(record, folder).name)

    return check


def relation_check(command: Sequence[str]) -> Check:
    """Return the api tests run by ``command`` (a pytest call) as a check that reads their outcome per test.

    Hypothesis keeps the examples it saw in the run's own folder, so an example that failed on one mutant is not
    tried first on the next one, and every run starts from the same seed.
    """

    async def check(folder: Path) -> CheckResult:
        """Run the tests, keep pytest's output and its result file, and read the outcome of every test."""
        folder.mkdir(parents=True, exist_ok=True)
        result_file = folder / "search_relations.xml"
        log_file = folder / "search_relations.log"
        env = {**os.environ, "HYPOTHESIS_STORAGE_DIRECTORY": str(folder / "hypothesis")}
        full = [*command, f"--junitxml={result_file}", f"--hypothesis-seed={_RELATION_SEED}"]
        done = await asyncio.to_thread(subprocess.run, full, capture_output=True, text=True, env=env)
        log_file.write_text(done.stdout + done.stderr, encoding="utf-8")
        if not result_file.is_file():
            return CheckResult(
                status="refused", refusal=f"no_result: pytest exited with {done.returncode}; see {log_file.name}"
            )
        return CheckResult(status="ran", result_file=result_file.name, tests=read_test_outcomes(result_file))

    return check


def read_test_outcomes(result_file: Path) -> dict[str, TestOutcome]:
    """Read the outcome of every test from pytest's JUnit file.

    A test that failed is "failed" even when its teardown failed too; a test whose setup or collection failed is
    "error", because the relation it checks was never tried.
    """
    outcomes: dict[str, TestOutcome] = {}
    for case in ElementTree.parse(result_file).getroot().iter("testcase"):
        tags = {child.tag for child in case}
        outcome: TestOutcome = "passed"
        if "failure" in tags:
            outcome = "failed"
        elif "error" in tags:
            outcome = "error"
        elif "skipped" in tags:
            outcome = "skipped"
        outcomes[str(case.get("name"))] = outcome
    return outcomes
