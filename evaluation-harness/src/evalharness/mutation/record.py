"""The record of one mutant run: which mutant, the proof that it was live and that the app came back, and the outcome
of every check as it came.

Each mutant run gets its own folder under runs/mutants/, with this record and the records the checks wrote, so a
run made on a mutant is never mistaken for a run of the app as it is configured. Whether a check killed the mutant is
not decided here: that is read later from these records, by a rule written before the first matrix.
"""

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from evalharness.mutation.catalog import CheckName, Mutant
from evalharness.mutation.proof import Change

RECORD_FILE = "mutant_run.json"

# The outcome of one api test as pytest reported it. "error": the test could not run (a fixture or the collection
# failed); it says nothing about the relation it checks.
type TestOutcome = Literal["passed", "failed", "error", "skipped"]


class _Strict(BaseModel):
    """Base for the record: an unknown key is an error, and a record never changes after it is made."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Proof(_Strict):
    """What the app's reported settings showed at one point of the run, compared with the first reading."""

    # Every setting that differed from the first reading.
    changed: dict[str, Change]
    # Why the app is not in the state the run claims at this point; None when it is.
    problem: str | None


class CheckResult(_Strict):
    """The outcome of one check in a mutant run, as it came."""

    # "ran": the check measured and left its result; "refused": the check declined to measure, and says why.
    status: Literal["ran", "refused"]
    # The check's own record or result file, in the folder of the mutant run.
    result_file: str | None = None
    # The refusal code and its detail, e.g. "stack_not_ready: …".
    refusal: str | None = None
    # Per api test, for a check made of api tests.
    tests: dict[str, TestOutcome] | None = None


class MutantRunRecord(_Strict):
    """Everything one mutant run did and saw."""

    run_id: str
    # The command that made the run: the matrix is built from the runs of one experiment, whose first no-mutant run
    # is the reference of every verdict. None in records written before runs kept it.
    experiment_id: str | None = None
    # In UTC.
    started_at: datetime
    # The harness's git commit, and whether its folder had changes not yet committed.
    harness_commit: str
    harness_dirty: bool
    # The catalog and the hash of its file: runs of different catalog versions are not put in one matrix.
    catalog_file: str
    catalog_sha256: str
    course_id: str
    mutant_id: str
    # The catalog entry as it was at the run: what changed, the faulty part, the expected symptom and killers.
    mutant: Mutant
    # The app's settings read before the mutant was applied; every mutant run of one experiment has the same.
    app_settings: dict[str, JsonValue]
    applied: Proof
    # By check name; empty when the mutant was not applied, so no check measured it.
    checks: dict[CheckName, CheckResult]
    restored: Proof

    def problem(self) -> str | None:
        """Say why the run cannot be read as a run of its mutant; None for a valid run."""
        if self.applied.problem is not None:
            return f"not applied: {self.applied.problem}"
        if self.restored.problem is not None:
            return f"not restored: {self.restored.problem}"
        return None


def run_folder(runs_dir: Path, started_at: datetime, mutant_id: str, run_id: str) -> Path:
    """Return the folder of one mutant run, named so that the folders sort by time."""
    return runs_dir / f"{started_at:%Y%m%dT%H%M%SZ}-{mutant_id}-{run_id[:8]}"


def write_record(record: MutantRunRecord, folder: Path) -> Path:
    """Write the record into the run's folder and return its path; an existing record is never replaced."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / RECORD_FILE
    with path.open("x", encoding="utf-8") as file:
        file.write(record.model_dump_json(indent=2))
    return path
