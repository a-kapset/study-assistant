"""One mutant run: apply the mutant, prove it is live, make the checks, restore the app, prove it is back.

The order follows a chaos experiment: the state is checked before the change, the change is made, the checks run,
and the change is undone whatever happened in between. The proofs are read from the app itself and from its
database, because a command that was accepted shows only what was asked for, not what the app runs with.
"""

import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pydantic import JsonValue

from evalharness.evaluation.retrieval import RunRefused, check_ready, read_app_settings
from evalharness.mutation.catalog import CheckName, LoadedCatalog, Mutant
from evalharness.mutation.course import IngestFailed, StoredCourse
from evalharness.mutation.oracles import Check
from evalharness.mutation.proof import apply_problem, changes, restore_problem
from evalharness.mutation.record import CheckResult, MutantRunRecord, Proof, run_folder, write_record
from evalharness.mutation.source import PreparedPatch
from evalharness.mutation.stack import Stack, StackError
from evalharness.transport import SutTransport


class RestoreFailed(RuntimeError):
    """The app is not back in its first state after a mutant run, so no further mutant may run; the record of the
    run that failed is written."""


@dataclass(frozen=True)
class Experiment:
    """What every mutant run of one command shares."""

    experiment_id: str
    loaded: LoadedCatalog
    catalog_file: str
    # The app's settings read before the first mutant; each run must leave the app with exactly these.
    first_settings: dict[str, JsonValue]
    # The hash of the course's stored passages before the first mutant; each run must leave the course with them.
    first_corpus: str
    harness_commit: str
    harness_dirty: bool
    runs_dir: Path
    # The prepared patch of every source mutant of the command, by mutant id.
    patches: Mapping[str, PreparedPatch] = field(default_factory=dict)


async def run_mutants(
    ids: Sequence[str],
    experiment: Experiment,
    stack: Stack,
    course: StoredCourse,
    sut: SutTransport,
    checks: Mapping[CheckName, Check],
    done: Callable[[MutantRunRecord], None],
) -> None:
    """Run the mutants in the given order and hand each record to ``done``.

    Stops at the first mutant after which the app did not come back (RestoreFailed is raised): every further run
    would measure a changed app and read it as its own mutant.
    """
    for mutant_id in ids:
        done(await run_mutant(mutant_id, experiment, stack, course, sut, checks))


async def run_mutant(
    mutant_id: str,
    experiment: Experiment,
    stack: Stack,
    course: StoredCourse,
    sut: SutTransport,
    checks: Mapping[CheckName, Check],
) -> MutantRunRecord:
    """Run one mutant of the catalog and write its record; raise RestoreFailed when the app did not come back.

    The checks run only when the app reports exactly the mutant's settings and, for a mutant that changes how the
    course is cut into passages, the stored passages changed. A check that refuses to measure is recorded with its
    reason and the next check runs. The app is restored in every case, also when a check raised.
    """
    mutant = experiment.loaded.catalog.mutants[mutant_id]
    started_at = datetime.now(UTC)
    run_id = uuid.uuid4().hex
    folder = run_folder(experiment.runs_dir, started_at, mutant_id, run_id)
    folder.mkdir(parents=True)
    applied = Proof(changed={}, problem="the run stopped before the mutant was applied")
    results: dict[CheckName, CheckResult] = {}
    patch = experiment.patches.get(mutant_id)
    try:
        applied = await _apply(stack, course, sut, mutant, experiment, patch)
        if applied.problem is None:
            for name, check in checks.items():
                try:
                    results[name] = await check(folder, mutant)
                except RunRefused as refusal:
                    results[name] = CheckResult(status="refused", refusal=str(refusal))
    finally:
        restored = await _restore(stack, course, sut, mutant, experiment, patch)
        record = MutantRunRecord(
            run_id=run_id,
            experiment_id=experiment.experiment_id,
            started_at=started_at,
            harness_commit=experiment.harness_commit,
            harness_dirty=experiment.harness_dirty,
            catalog_file=experiment.catalog_file,
            catalog_sha256=experiment.loaded.sha256,
            course_id=experiment.loaded.catalog.course_id,
            mutant_id=mutant_id,
            mutant=mutant,
            app_settings=experiment.first_settings,
            applied=applied,
            checks=results,
            restored=restored,
        )
        write_record(record, folder)
    if restored.problem is not None:
        raise RestoreFailed(f"{mutant_id}: the app is not back in its first state: {restored.problem}")
    return record


async def _apply(
    stack: Stack,
    course: StoredCourse,
    sut: SutTransport,
    mutant: Mutant,
    experiment: Experiment,
    patch: PreparedPatch | None,
) -> Proof:
    """Start the app again with the mutant's settings or its changed code files, ingest the course again when the
    mutant asks for it, and compare what the app reports, what its container holds and what is stored with the first
    readings."""
    if mutant.kind == "source" and patch is None:
        return Proof(changed={}, problem="the patch of the source mutant was not prepared")
    try:
        stack.recreate_app(mutant.settings, patch.override if patch is not None else None)
        now = await read_app_settings(sut)
    except (StackError, RunRefused) as error:
        return Proof(changed={}, problem=f"the app did not start with the mutant: {error}")
    changed = changes(experiment.first_settings, now)
    problem = apply_problem(mutant.settings, changed)
    if problem is None and patch is not None:
        problem = _files_problem(stack, patch, patched=True)
    if problem is None and mutant.reingest:
        try:
            await course.ingest()
        except IngestFailed as error:
            return Proof(changed=changed, problem=str(error))
        if course.snapshot() == experiment.first_corpus:
            problem = "the stored passages of the course did not change after it was ingested again"
    return Proof(changed=changed, problem=problem)


async def _restore(
    stack: Stack,
    course: StoredCourse,
    sut: SutTransport,
    mutant: Mutant,
    experiment: Experiment,
    patch: PreparedPatch | None,
) -> Proof:
    """Start the app again as the stack configures it, ingest the course again when the mutant changed it, and check
    that the settings, the code files a patch changed and the stored passages equal the first readings and that
    everything the app uses answers."""
    try:
        stack.recreate_app({})
        now = await read_app_settings(sut)
    except (StackError, RunRefused) as error:
        return Proof(changed={}, problem=f"the app did not start as configured: {error}")
    changed = changes(experiment.first_settings, now)
    problem = restore_problem(changed)
    if problem is None and patch is not None:
        problem = _files_problem(stack, patch, patched=False)
    if problem is None and mutant.reingest:
        try:
            await course.ingest()
        except IngestFailed as error:
            problem = str(error)
    if problem is None and course.snapshot() != experiment.first_corpus:
        problem = "the stored passages of the course differ from those before the first mutant"
    if problem is None:
        try:
            await check_ready(sut)
        except RunRefused as refusal:
            problem = str(refusal)
    return Proof(changed=changed, problem=problem)


def _files_problem(stack: Stack, patch: PreparedPatch, *, patched: bool) -> str | None:
    """Say which code files in the app's container are not the version expected now: the patched one while the
    mutant is applied, the image's own after the restore; None when every file is. The hash is read inside the
    container, because a mount that was asked for shows only the command, not the file the app runs."""
    wrong = []
    for file in patch.files:
        expected = file.patched_sha256 if patched else file.original_sha256
        try:
            actual = stack.file_sha256(file.container_path)
        except StackError as error:
            return f"the code files in the app's container could not be read: {error}"
        if actual != expected:
            wrong.append(file.container_path)
    if not wrong:
        return None
    return f"the app's container does not run the {'patched' if patched else 'original'} version of {wrong}"
