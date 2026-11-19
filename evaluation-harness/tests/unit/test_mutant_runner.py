"""Unit tests of a mutant run: a fake stack whose app reports the settings it was last started with, and fake
checks; never the real stack."""

import copy
from collections.abc import Mapping
from pathlib import Path

import httpx2
import pytest
from pydantic import HttpUrl, JsonValue

from evalharness.evaluation.retrieval import RunRefused
from evalharness.mutation.catalog import Catalog, CheckName, Checks, LoadedCatalog, Mutant
from evalharness.mutation.oracles import Check
from evalharness.mutation.record import RECORD_FILE, CheckResult, MutantRunRecord
from evalharness.mutation.runner import Experiment, RestoreFailed, run_mutant, run_mutants
from evalharness.settings import HarnessSettings
from evalharness.transport import SutTransport

FIRST: dict[str, JsonValue] = {
    "git_commit": "evalharness_commit",
    "search": {"default_k": 5, "min_score": 0.83},
    "ingest": {"chunk_size": 1000},
}
LOW_SCORE = Mutant(
    kind="configuration",
    component="search",
    settings={"APP_SEARCH__MIN_SCORE": "0.88"},
    expected_symptom="evalharness",
    expected_killers=("retrieval_gold",),
    origin="invented",
)
ONE_HIT = Mutant(
    kind="configuration",
    component="search",
    settings={"APP_SEARCH__DEFAULT_K": "1"},
    expected_symptom="evalharness",
    origin="invented",
)
SHORT_PASSAGES = Mutant(
    kind="configuration",
    component="ingestion",
    settings={"APP_INGEST__CHUNK_SIZE": "200"},
    reingest=True,
    expected_symptom="evalharness",
    origin="invented",
)
NO_MUTANT = Mutant(kind="baseline", expected_symptom="evalharness")
FIRST_CORPUS = "evalharness_corpus_first"


class FakeStack:
    """A stack whose app reports the first settings with the values it was last started with.

    With ``keeps_values`` a restart without values keeps the values of the restart before it, as an app started
    again without the stack's own configuration would.
    """

    def __init__(self, keeps_values: bool = False) -> None:
        """Start with the first settings and no restart seen."""
        self.keeps_values = keeps_values
        self.settings: dict[str, JsonValue] = copy.deepcopy(FIRST)
        self.restarts: list[dict[str, str]] = []

    def recreate_app(self, settings: Mapping[str, str]) -> None:
        """Remember the restart and set the reported settings from the first ones and the given values."""
        self.restarts.append(dict(settings))
        if not settings and self.keeps_values:
            return
        self.settings = copy.deepcopy(FIRST)
        for name, value in settings.items():
            group, key = name.removeprefix("APP_").lower().split("__")
            section = self.settings[group]
            assert isinstance(section, dict)
            section[key] = float(value) if "." in value else int(value)

    def serve(self, request: httpx2.Request) -> httpx2.Response:
        """Answer the app's settings request and its readiness request."""
        if request.url.path == "/v1/config":
            return httpx2.Response(200, json={"settings": self.settings})
        return httpx2.Response(200, json={"status": "ok", "checks": {"db": "ok"}})


class FakeCourse:
    """Stored passages that follow the chunk size the fake stack's app was last started with, but only once the
    course is ingested; with ``ingests`` false an ingestion is accepted and changes nothing."""

    def __init__(self, stack: FakeStack, ingests: bool = True) -> None:
        """Start with the first passages; remember the stack whose settings an ingestion reads."""
        self.stack = stack
        self.ingests = ingests
        self.stored = FIRST_CORPUS
        self.ingestions: list[JsonValue] = []

    async def ingest(self) -> None:
        """Make the passages again with the chunk size the app now reports."""
        ingest = self.stack.settings.get("ingest")
        size = ingest.get("chunk_size") if isinstance(ingest, dict) else None
        self.ingestions.append(size)
        if self.ingests:
            self.stored = FIRST_CORPUS if size == 1000 else f"evalharness_corpus_{size}"

    def snapshot(self) -> str:
        """Name the passages stored now."""
        return self.stored


def _experiment(runs_dir: Path) -> Experiment:
    """Return an experiment over a catalog with three mutants and the retrieval and paraphrase checks."""
    catalog = Catalog(
        course_id="evalharness_course",
        checks=Checks(retrieval_gold=Path("gold.toml"), paraphrase=Path("pairs.toml")),
        mutants={"baseline": NO_MUTANT, "low_score": LOW_SCORE, "one_hit": ONE_HIT, "short": SHORT_PASSAGES},
    )
    loaded = LoadedCatalog(catalog, "evalharness_sha")
    return Experiment("evalharness_experiment", loaded, "catalog.toml", FIRST, FIRST_CORPUS, "commit", False, runs_dir)


def _ran(name: str, made: list[str]) -> Check:
    """Return a check that notes it was made and reports a result."""

    async def check(folder: Path) -> CheckResult:
        """Note the check and return a result."""
        made.append(name)
        return CheckResult(status="ran", result_file=f"{name}.json")

    return check


async def _run(
    stack: FakeStack,
    checks: Mapping[CheckName, Check],
    runs_dir: Path,
    mutant_id: str = "low_score",
    course: FakeCourse | None = None,
) -> MutantRunRecord:
    """Run one mutant against the fake stack and the fake course."""
    settings = HarnessSettings(sut_base_url=HttpUrl("http://sut.test"))
    async with SutTransport(settings, transport=httpx2.MockTransport(stack.serve)) as sut:
        return await run_mutant(mutant_id, _experiment(runs_dir), stack, course or FakeCourse(stack), sut, checks)


def _written(runs_dir: Path) -> MutantRunRecord:
    """Read the one mutant run record written under ``runs_dir``."""
    (path,) = runs_dir.glob(f"*/{RECORD_FILE}")
    return MutantRunRecord.model_validate_json(path.read_bytes())


@pytest.mark.covers("US-019", "AC-3")
async def test_the_app_is_restored_when_a_check_raises(tmp_path: Path) -> None:
    """A check fails with an error of its own while the mutant is live. The run must still start the app again as
    configured and write its record before the error goes on.

    A pass says that no path through a mutant run leaves the app mutated. A failure points to the restore in
    runner.py, which must not depend on the checks finishing. Without it the next mutant, or the next ordinary run
    on the stack, would measure the mutant under another name.
    """

    async def broken(folder: Path) -> CheckResult:
        """Fail like a check with a bug."""
        raise RuntimeError("evalharness check bug")

    stack = FakeStack()
    with pytest.raises(RuntimeError, match="evalharness check bug"):
        await _run(stack, {"retrieval_gold": broken}, tmp_path)

    assert stack.restarts[-1] == {}, f"the last restart was {stack.restarts[-1]}, not the app as configured"
    assert stack.settings == FIRST, f"the app still reports {stack.settings}"
    assert _written(tmp_path).restored.problem is None, "the record does not show the restore"


@pytest.mark.covers("US-019", "AC-3")
async def test_a_restore_that_does_not_bring_the_first_settings_back_stops_the_command(tmp_path: Path) -> None:
    """The app, started again without values, keeps the mutant's lowest score. The run of the first mutant must be
    recorded as not restored, and the second mutant must never be applied.

    A pass says that a broken restore is seen from what the app reports, not taken from the command that was sent,
    and that nothing runs on a stack left changed. A failure points to the comparison after the restore or to the
    stop in runner.py. The real case: an app started again with only one of the stack's compose files.
    """
    stack = FakeStack(keeps_values=True)
    settings = HarnessSettings(sut_base_url=HttpUrl("http://sut.test"))
    done: list[MutantRunRecord] = []
    async with SutTransport(settings, transport=httpx2.MockTransport(stack.serve)) as sut:
        with pytest.raises(RestoreFailed, match="min_score"):
            await run_mutants(
                ["low_score", "one_hit"], _experiment(tmp_path), stack, FakeCourse(stack), sut, {}, done.append
            )

    assert {"APP_SEARCH__DEFAULT_K": "1"} not in stack.restarts, f"the second mutant ran: {stack.restarts}"
    assert _written(tmp_path).problem() is not None, "the record of the first mutant reads as valid"


@pytest.mark.covers("US-019", "AC-5")
async def test_a_refused_check_is_recorded_with_its_reason_and_the_next_check_runs(tmp_path: Path) -> None:
    """The first check refuses to measure, as the retrieval run does on a stack that is not ready. Its refusal and
    code must be in the record, and the second check must still run.

    A pass says that a refusal is an outcome of the mutant run, kept for the matrix. A failure points to the
    handling of refusals in runner.py. Without it a mutant that makes the stack refuse a measurement would end the
    run without a record, and the very effect of that mutant would be lost.
    """

    async def refusing(folder: Path) -> CheckResult:
        """Refuse like a run on a stack that is not ready."""
        raise RunRefused("stack_not_ready", "evalharness")

    made: list[str] = []
    record = await _run(FakeStack(), {"retrieval_gold": refusing, "paraphrase": _ran("paraphrase", made)}, tmp_path)

    refused = record.checks["retrieval_gold"]
    assert refused.status == "refused", f"the refused check is recorded as {refused.status}"
    assert refused.refusal is not None and refused.refusal.startswith("stack_not_ready"), (
        f"the refusal code is lost: {refused.refusal!r}"
    )
    assert made == ["paraphrase"], f"the check after the refusal did not run: {made}"


@pytest.mark.covers("US-019", "AC-2", "AC-3")
@pytest.mark.parametrize(
    ("ingests", "valid"),
    [
        pytest.param(True, True, id="passages made again"),
        pytest.param(False, False, id="ingestion that changed nothing"),
    ],
)
async def test_a_reingesting_mutant_is_valid_only_when_the_stored_passages_changed_and_came_back(
    tmp_path: Path, ingests: bool, valid: bool
) -> None:
    """A mutant that cuts the course into shorter passages runs against a fake course. When ingestion makes the
    passages again, the run must be valid, with one ingestion under the mutant's size and one under the first size.
    When ingestion is accepted but stores the old passages, the run must be invalid.

    A pass says that a changed passage size counts only when the stored passages really changed, and that they are
    made again as before after the run. A failure points to the ingestion steps or the passage comparison in
    runner.py. Without them a run would measure the old passages under the mutant's name, and the next mutant
    would run on passages of the wrong size.
    """
    stack = FakeStack()
    course = FakeCourse(stack, ingests=ingests)

    record = await _run(stack, {}, tmp_path, mutant_id="short", course=course)

    assert course.ingestions == [200, 1000], f"ingested with chunk sizes {course.ingestions}"
    assert (record.problem() is None) == valid, f"valid should be {valid}, problem: {record.problem()}"
