"""API tests of a mutant run on the running stack: the app is really started again with the mutant's setting, and it
is really back as configured afterwards.

One mutant run is made for the whole module, on the stack's local course: the number of hits set to one, measured by
the retrieval run of the course's gold set. It starts the app's container twice, so the module takes about half a
minute more than the searches themselves.
"""

import asyncio
from pathlib import Path
from typing import NamedTuple

import pytest
from pydantic import JsonValue

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

ONE_HIT = Mutant(
    kind="configuration",
    component="search",
    settings={"APP_SEARCH__DEFAULT_K": "1"},
    expected_symptom="every search returns one passage",
    expected_killers=("retrieval_gold",),
    origin="invented",
)


class MutantRunSeen(NamedTuple):
    """What the module's mutant run left: its record, its folder, and the app's state before and after it."""

    record: MutantRunRecord
    folder: Path
    settings_before: dict[str, JsonValue]
    settings_after: dict[str, JsonValue]
    ready_after: JsonValue


@pytest.fixture(scope="module")
def one_hit_run(tmp_path_factory: pytest.TempPathFactory) -> MutantRunSeen:
    """Run the one-hit mutant once through the real runner, the real compose stack and the real retrieval check,
    and read the app's settings and readiness before and after it."""
    settings = HarnessSettings()
    stack = load_stack_settings(settings.stack_env_file)
    catalog = Catalog(
        course_id=stack.local_course_id,
        checks=Checks(retrieval_gold=Path(f"datasets/{stack.local_course_id}/retrieval_gold_v0.toml")),
        mutants={"baseline": Mutant(kind="baseline", expected_symptom="nothing differs"), "one_hit": ONE_HIT},
    )
    checks = build_checks(catalog, settings, stack, "evalharness_api_test", False)
    runs_dir = tmp_path_factory.mktemp("mutants")
    course = AdminIngestedCourse(settings, stack, stack.local_course_id, stack.admin_api_key)

    async def run() -> MutantRunSeen:
        """Read the first state, run the mutant, read the state after it."""
        async with SutTransport(settings) as sut:
            before = await read_app_settings(sut)
            commit = before.get("git_commit")
            compose = ComposeStack.discover(settings.stack_env_file, commit if isinstance(commit, str) else "")
            loaded = LoadedCatalog(catalog, "evalharness_sha")
            experiment = Experiment(
                "evalharness_api_test",
                loaded,
                "in-test",
                before,
                course.snapshot(),
                "evalharness_api_test",
                False,
                runs_dir,
            )
            record = await run_mutant("one_hit", experiment, compose, course, sut, checks)
            after = await read_app_settings(sut)
            ready = await sut.request("GET", "/ready")
        (folder,) = runs_dir.iterdir()
        return MutantRunSeen(record, folder, before, after, ready.body_json)

    return asyncio.run(run())


@pytest.mark.covers("US-019", "AC-2")
def test_a_configuration_mutant_reaches_the_running_app(one_hit_run: MutantRunSeen) -> None:
    """The one-hit mutant is applied by starting the app's container again with APP_SEARCH__DEFAULT_K=1. The run
    must be valid, and the searches the retrieval run made during it must each have returned at most one passage.

    A pass says that a value of the catalog goes through Docker Compose into the container and changes what the app
    does, proven by the app's search responses, not only by the runner's own reading of the settings. A failure
    points to the apply path of stack.py and runner.py: values not handed to Compose, another service started
    again, or a setting name the app does not read. The unit tests prove the comparison on a fake stack; this is the
    real stack under that fake's assumption.
    """
    seen = one_hit_run
    assert seen.record.problem() is None, f"the mutant run is invalid: {seen.record.problem()}"
    result = seen.record.checks["retrieval_gold"]
    assert result.status == "ran" and result.result_file, f"the retrieval run did not measure: {result}"
    retrieval = RunRecord.model_validate_json((seen.folder / result.result_file).read_bytes())
    longest = max(len(case.passes["primary"].hits) for case in retrieval.cases)
    assert retrieval.k == 1 and longest <= 1, f"the app searched with k={retrieval.k}, up to {longest} passages"


@pytest.mark.covers("US-019", "AC-3")
def test_after_a_mutant_run_the_app_is_as_it_was_found(one_hit_run: MutantRunSeen) -> None:
    """After the one-hit mutant run, the app must report the same settings as before it and say that it is ready.

    A pass says that the restore of the real stack brings the app back with all of the stack's compose files and
    its own configuration, so whatever runs next measures the app as configured. A failure points to the restore
    path or to how stack.py finds the stack's compose files; the real case is an app started again with the main
    file only, which then calls the model provider without the proxy.
    """
    seen = one_hit_run
    assert seen.settings_after == seen.settings_before, "the app's settings differ from those before the mutant run"
    assert isinstance(seen.ready_after, dict) and seen.ready_after.get("status") == "ok", (
        f"the app is not ready after the run: {seen.ready_after}"
    )
