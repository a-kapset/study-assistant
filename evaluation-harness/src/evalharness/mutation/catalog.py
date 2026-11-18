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

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

# The checks a mutant run can make. "retrieval_gold": the retrieval run of a gold set's main result; "paraphrase":
# the paraphrase run of a pair file; "search_relations": the two api tests of the relations on the number of hits
# and the lowest score.
type CheckName = Literal["retrieval_gold", "paraphrase", "search_relations"]
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
    # "configuration": the app started again with some of its settings changed.
    kind: Literal["baseline", "configuration"]
    # The part of the app the mutant makes faulty; none for the baseline.
    component: Component | None = None
    # The app settings the mutant changes, as the stack passes them in (e.g. APP_SEARCH__MIN_SCORE = "0.88").
    settings: dict[str, str] = {}
    # What is expected to happen, in words, written before the first run.
    expected_symptom: str
    # The checks expected to notice the mutant, written before the first run; none for the baseline.
    expected_killers: tuple[CheckName, ...] = ()
    # Where the idea of the defect came from: made up, or seen in a real codebase (then "reference" says where).
    origin: Literal["invented", "observed"] | None = None
    reference: str | None = None

    @model_validator(mode="after")
    def _fits_its_kind(self) -> Self:
        """Refuse a baseline that changes or expects anything, and a mutant that changes nothing or does not say
        what it breaks and where the idea came from."""
        if self.kind == "baseline":
            if self.settings or self.component or self.expected_killers or self.origin:
                raise ValueError("a baseline changes no setting and names no component, killer or origin")
            return self
        if not self.settings:
            raise ValueError("a configuration mutant changes at least one setting")
        if self.component is None or self.origin is None:
            raise ValueError("a mutant names the component it makes faulty and its origin")
        bad = sorted(name for name in self.settings if not _SETTING.fullmatch(name))
        if bad:
            raise ValueError(f"not an app setting name (APP_<GROUP>__<NAME>): {bad}")
        return self


class Checks(_Strict):
    """The checks every mutant run of the catalog makes, with the files they read (relative to the harness)."""

    retrieval_gold: Path | None = None
    paraphrase: Path | None = None
    search_relations: bool = False

    def names(self) -> tuple[CheckName, ...]:
        """Return the names of the checks this catalog makes, in a fixed order."""
        made: list[CheckName] = []
        if self.retrieval_gold is not None:
            made.append("retrieval_gold")
        if self.paraphrase is not None:
            made.append("paraphrase")
        if self.search_relations:
            made.append("search_relations")
        return tuple(made)


class Catalog(_Strict):
    """The mutants of one course and the checks each of them goes through."""

    course_id: str
    checks: Checks
    # In the order of the file; the key is the mutant's id.
    mutants: dict[str, Mutant]

    @model_validator(mode="after")
    def _expectations_name_made_checks(self) -> Self:
        """Refuse an id that cannot be a folder name, a catalog that makes no check, and a mutant that expects a
        check the catalog does not make: that cell of the matrix would read as "not aimed at" without a sign."""
        bad_ids = sorted(mutant_id for mutant_id in self.mutants if not _MUTANT_ID.fullmatch(mutant_id))
        if bad_ids:
            raise ValueError(f"mutant ids may hold only a-z, 0-9 and _: {bad_ids}")
        made = self.checks.names()
        if not made:
            raise ValueError("the catalog makes no check")
        for mutant_id, mutant in self.mutants.items():
            unknown = sorted(set(mutant.expected_killers) - set(made))
            if unknown:
                raise ValueError(f"{mutant_id}: expects checks the catalog does not make: {unknown}; made: {made}")
        return self


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
