"""The mutant catalog of a course: the defects planted in the running app one at a time, read from a TOML file.

A mutant says what it changes, which part of the app that makes faulty, what is expected to happen and which checks
are expected to notice it. The expectations are written before the first run, so a check that misses a mutant it was
aimed at can be told apart from a check that was never meant to see it. The catalog also names the checks every
mutant run makes; a mutant may only expect checks the catalog makes.
"""

import hashlib
import re
import tomllib
from pathlib import Path
from typing import Literal, NamedTuple, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from evalharness.failures import FailureLabel
from evalharness.proxy.faults import FaultKind

# The checks a mutant run can make. "retrieval_gold": the retrieval run of a gold set's main result; "paraphrase":
# the paraphrase run of a pair file; "search_relations": the two api tests of the relations on the number of hits
# and the lowest score; "answer_run": the answer run of a Q&A set, every question asked more than once;
# "failure_labels": the labels the answer run gave its failed asks, read from its record without new calls.
type CheckName = Literal["retrieval_gold", "paraphrase", "search_relations", "answer_run", "failure_labels"]
# The part of the app a mutant makes faulty.
type Component = Literal["search", "embedding", "ingestion", "answer", "provider"]

# An app setting as the stack passes it in: the APP_ prefix, then groups and names joined by "__".
_SETTING = re.compile(r"APP_[A-Z0-9]+(_[A-Z0-9]+)*(__[A-Z0-9]+(_[A-Z0-9]+)*)*")
# A mutant id becomes part of a folder name.
_MUTANT_ID = re.compile(r"[a-z0-9_]+")


class CatalogError(ValueError):
    """The catalog file cannot be used; the message says which entry and why."""


class _Strict(BaseModel):
    """Base for the catalog: an unknown key is an error, and nothing changes after it is read."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Mutant(_Strict):
    """One planted defect, or the run without one that every mutant is compared with."""

    # "baseline": the app as it is configured, sent through the same steps and checks as a mutant;
    # "configuration": the app started again with some of its settings changed; "provider_fault": the app as
    # configured, while its model provider answers every call of the answer run with one failure; "source": the app
    # started again with files of its code changed by a patch.
    kind: Literal["baseline", "configuration", "provider_fault", "source"]
    # The part of the app the mutant makes faulty; none for the baseline.
    component: Component | None = None
    # The app settings the mutant changes, as the stack passes them in (e.g. APP_SEARCH__MIN_SCORE = "0.88").
    settings: dict[str, str] = {}
    # Whether the course must be ingested again for the settings to reach what the checks measure: a setting of how
    # sources are cut into passages changes nothing until the stored passages are made again. The runner then
    # ingests the course after starting the app with the mutant, and again after starting it as configured.
    reingest: bool = False
    # What is expected to happen, in words, written before the first run.
    expected_symptom: str
    # The failure the provider answers with, for a provider fault only.
    fault: FaultKind | None = None
    # The patch of the app's code, for a source mutant only: a git diff made from the repository root, its path
    # relative to the harness folder.
    patch: Path | None = None
    # The checks expected to notice the mutant, written before the first run; none for the baseline.
    expected_killers: tuple[CheckName, ...] = ()
    # The failure label the failed asks are expected to get, written before the first run; the matrix says whether
    # the labels name it. Only for a mutant that expects the failure labels to notice it.
    expected_label: FailureLabel | None = None
    # Where the idea of the defect came from: made up, or seen in a real codebase (then "reference" says where).
    origin: Literal["invented", "observed"] | None = None
    reference: str | None = None

    @model_validator(mode="after")
    def _fits_its_kind(self) -> Self:
        """Refuse a baseline that changes or expects anything, and a mutant that changes nothing or does not say
        what it breaks and where the idea came from."""
        if self.kind == "baseline":
            if (
                self.settings
                or self.reingest
                or self.component
                or self.expected_killers
                or self.origin
                or self.fault
                or self.expected_label
                or self.patch
            ):
                raise ValueError(
                    "a baseline changes no setting, ingests nothing, and names no component, fault, killer, label, "
                    "origin or patch"
                )
            return self
        if self.component is None or self.origin is None:
            raise ValueError("a mutant names the component it makes faulty and its origin")
        if self.expected_label is not None and "failure_labels" not in self.expected_killers:
            raise ValueError("a mutant that names an expected label expects the failure labels to notice it")
        if self.kind == "source":
            if self.patch is None or self.settings or self.reingest or self.fault is not None:
                raise ValueError(
                    "a source mutant names its patch and changes no setting, ingests nothing, plants no fault"
                )
            return self
        if self.patch is not None:
            raise ValueError("only a source mutant names a patch")
        if self.kind == "provider_fault":
            if self.fault is None or self.settings or self.reingest or self.component != "provider":
                raise ValueError("a provider fault names its fault, changes no setting and makes the provider faulty")
            return self
        if self.fault is not None:
            raise ValueError("only a provider fault names a fault")
        if not self.settings:
            raise ValueError("a configuration mutant changes at least one setting")
        bad = sorted(name for name in self.settings if not _SETTING.fullmatch(name))
        if bad:
            raise ValueError(f"not an app setting name (APP_<GROUP>__<NAME>): {bad}")
        return self


class AnswerCheck(_Strict):
    """The answer run a mutant run makes: its Q&A set and how many times every question is asked."""

    qa: Path
    repetitions: int = Field(ge=1)


class Checks(_Strict):
    """The checks every mutant run of the catalog makes, with the files they read (relative to the harness)."""

    retrieval_gold: Path | None = None
    paraphrase: Path | None = None
    search_relations: bool = False
    # The answer run; its record also gives the failure labels, so both checks come with it.
    answer_run: AnswerCheck | None = None

    def names(self) -> tuple[CheckName, ...]:
        """Return the names of the checks this catalog makes, in a fixed order."""
        made: list[CheckName] = []
        if self.retrieval_gold is not None:
            made.append("retrieval_gold")
        if self.paraphrase is not None:
            made.append("paraphrase")
        if self.search_relations:
            made.append("search_relations")
        if self.answer_run is not None:
            made.extend(("answer_run", "failure_labels"))
        return tuple(made)


class Catalog(_Strict):
    """The mutants of one course and the checks each of them goes through."""

    course_id: str
    checks: Checks
    # In the order of the file; the key is the mutant's id.
    mutants: dict[str, Mutant]

    @model_validator(mode="after")
    def _expectations_name_made_checks(self) -> Self:
        """Refuse an id that cannot be a folder name, a catalog without exactly one baseline (every verdict is read
        against it), a catalog that makes no check, and a mutant that expects a check the catalog does not make: that
        cell of the matrix would read as "not aimed at" without a sign."""
        bad_ids = sorted(mutant_id for mutant_id in self.mutants if not _MUTANT_ID.fullmatch(mutant_id))
        if bad_ids:
            raise ValueError(f"mutant ids may hold only a-z, 0-9 and _: {bad_ids}")
        baselines = [mutant_id for mutant_id, mutant in self.mutants.items() if mutant.kind == "baseline"]
        if len(baselines) != 1:
            raise ValueError(f"the catalog needs exactly one entry of kind baseline, has {baselines}")
        made = self.checks.names()
        if not made:
            raise ValueError("the catalog makes no check")
        for mutant_id, mutant in self.mutants.items():
            unknown = sorted(set(mutant.expected_killers) - set(made))
            if unknown:
                raise ValueError(f"{mutant_id}: expects checks the catalog does not make: {unknown}; made: {made}")
        return self

    def baseline_id(self) -> str:
        """Return the id of the catalog's one baseline entry."""
        return next(mutant_id for mutant_id, mutant in self.mutants.items() if mutant.kind == "baseline")


class LoadedCatalog(NamedTuple):
    """A catalog with the hash of its file, which every mutant run records so runs of different versions are not
    mixed in one matrix."""

    catalog: Catalog
    sha256: str


def load_catalog(path: Path) -> LoadedCatalog:
    """Read and check a catalog file; a file that cannot be used raises CatalogError with the reason."""
    content = path.read_bytes()
    try:
        catalog = Catalog.model_validate(tomllib.loads(content.decode("utf-8")))
    except (tomllib.TOMLDecodeError, ValidationError) as error:
        raise CatalogError(f"{path}: {error}") from error
    return LoadedCatalog(catalog, hashlib.sha256(content).hexdigest())
