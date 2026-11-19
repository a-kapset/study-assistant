"""API test of a mutant that changes how the course is cut into passages, on the running stack: the passages are
really made again under the mutant and made again as before afterwards.

One mutant run is made for the module, on the stack's local course, measured by the retrieval run of its gold set.
It starts the app twice and ingests the course twice, so on the first course the module takes some minutes.
"""

import asyncio
from pathlib import Path
from typing import NamedTuple

import pytest

from evalharness.evaluation.record import RunRecord
from evalharness.evaluation.retrieval import read_app_settings
from evalharness.mutation.catalog import Catalog, Checks, LoadedCatalog, Mutant
from evalharness.mutation.course import AdminIngestedCourse
from evalharness.mutation.oracles import build_checks
from evalharness.mutation.record import MutantRunRecord
from evalharness.mutation.runner import Experiment, run_mutant
from evalharness.mutation.stack import ComposeStack
from evalharness.settings import HarnessSettings, load_stack_settings
from evalharness.transport import SutTransport

# Short enough that every course of the stack has sections longer than one passage, so the passages must change.
SHORT_PASSAGES = Mutant(
    kind="configuration",
    component="ingestion",
    settings={"APP_INGEST__CHUNK_SIZE": "300", "APP_INGEST__CHUNK_OVERLAP": "30"},
    reingest=True,
    expected_symptom="passages of about two sentences",
    expected_killers=("retrieval_gold",),
    origin="invented",
)


class ReingestSeen(NamedTuple):
    """What the module's mutant run left: its record and folder, and the course's passages before and after it."""

    record: MutantRunRecord
    folder: Path
    corpus_before: str
    corpus_after: str


@pytest.fixture(scope="module")
def short_passages_run(tmp_path_factory: pytest.TempPathFactory) -> ReingestSeen:
    """Run the short-passage mutant once through the real runner, compose stack, admin ingestion and retrieval
    check, and read the course's stored passages before and after it."""
    settings = HarnessSettings()
    stack = load_stack_settings(settings.stack_env_file)
    catalog = Catalog(
        course_id=stack.local_course_id,
        checks=Checks(retrieval_gold=Path(f"datasets/{stack.local_course_id}/retrieval_gold_v0.toml")),
        mutants={"baseline": Mutant(kind="baseline", expected_symptom="nothing differs"), "short": SHORT_PASSAGES},
    )
    checks = build_checks(catalog, settings, stack, "evalharness_api_test", False)
    runs_dir = tmp_path_factory.mktemp("mutants")
    course = AdminIngestedCourse(settings, stack, stack.local_course_id, stack.admin_api_key)

    async def run() -> ReingestSeen:
        """Read the first state, run the mutant, read the passages after it."""
        corpus_before = course.snapshot()
        async with SutTransport(settings) as sut:
            first = await read_app_settings(sut)
            commit = first.get("git_commit")
            compose = ComposeStack.discover(settings.stack_env_file, commit if isinstance(commit, str) else "")
            loaded = LoadedCatalog(catalog, "evalharness_sha")
            experiment = Experiment(
                "evalharness_api_test", loaded, "in-test", first, corpus_before, "evalharness_api_test", False, runs_dir
            )
            record = await run_mutant("short", experiment, compose, course, sut, checks)
        (folder,) = runs_dir.iterdir()
        return ReingestSeen(record, folder, corpus_before, course.snapshot())

    return asyncio.run(run())


@pytest.mark.covers("US-019", "AC-2", "AC-3")
def test_a_passage_size_mutant_reaches_the_stored_course_and_the_course_comes_back(
    short_passages_run: ReingestSeen,
) -> None:
    """The short-passage mutant starts the app with APP_INGEST__CHUNK_SIZE=300 and ingests the course again. The run
    must be valid, the retrieval run made during it must name other stored passages than those before it, and after
    the run the course must hold its first passages again.

    A pass says that a setting of the ingestion goes through Docker Compose, the app and the admin endpoint into the
    database, proven by the retrieval run's own record, and that the restore ingests the course back. A failure
    points to the ingestion steps of runner.py or to course.py: an ingestion skipped or refused, or a restore that
    leaves the short passages for every later run. The unit test proves the same order on a fake course; this is the
    real stack under that fake's assumption.
    """
    seen = short_passages_run
    assert seen.record.problem() is None, f"the mutant run is invalid: {seen.record.problem()}"
    result = seen.record.checks["retrieval_gold"]
    assert result.status == "ran" and result.result_file, f"the retrieval run did not measure: {result}"
    retrieval = RunRecord.model_validate_json((seen.folder / result.result_file).read_bytes())
    assert retrieval.corpus_snapshot != seen.corpus_before, "the retrieval run measured the passages from before"
    assert seen.corpus_after == seen.corpus_before, "the course does not hold its first passages after the run"
