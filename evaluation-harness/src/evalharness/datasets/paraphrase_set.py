"""Paraphrase pairs of a course: answerable questions of a Q&A set, each with a rewording that keeps its meaning and
its gold section but leaves out the course's own words for it (the question's key terms).

The pairs feed a metamorphic relation on search: a learner who does not know a term yet asks in other words, and
search should still find the passage that answers. The file names its Q&A set and states, before any run, how the
relation is judged. A pair is refused when the rewording is not what it claims to be, because a run on such a pair
would measure something else.
"""

import hashlib
import tomllib
from collections import Counter
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evalharness.datasets.check_qa_set import normal_text, term_occurs
from evalharness.datasets.qa_set import QaSet, load_qa_set


class _Strict(BaseModel):
    """Base for the set's records: an unknown or misspelt key is an error, not silently ignored."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RelationRules(_Strict):
    """How the two relations are judged, fixed before the first run."""

    # Answer kept: a change of recall whose interval stays closer to zero than this is not worth acting on.
    min_meaningful_change: float = Field(gt=0, lt=1)
    # Top-k stable: the part of the k passages two searches of a pair must share for the pair to count as stable.
    min_overlap: float = Field(gt=0, le=1)
    # Top-k stable: the share of stable pairs the search must reach.
    stable_share_min: float = Field(gt=0, le=1)


class _PairLine(_Strict):
    """One pair as written in the file: the Q&A case id and its rewording."""

    id: str = Field(min_length=1)
    paraphrase: str = Field(min_length=1)


class _SetFile(_Strict):
    """The file as written: its course, its Q&A set, the rules and the pairs."""

    course_id: str = Field(min_length=1)
    # Next to the pair file.
    qa_file: str = Field(min_length=1)
    relations: RelationRules
    pairs: tuple[_PairLine, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_ids(self) -> Self:
        """Refuse two pairs for the same question: their results would be counted twice."""
        repeated = sorted(case_id for case_id, n in Counter(pair.id for pair in self.pairs).items() if n > 1)
        if repeated:
            raise ValueError(f"questions with more than one pair: {repeated}")
        return self


class ParaphrasePair(_Strict):
    """One question, its rewording and the section that answers both."""

    case_id: str
    question: str
    paraphrase: str
    gold_section: str
    # Whether a key term occurs in the question, so the rewording removed it; when False the question was already
    # in plain words and the rewording only changed them. The run reports the two groups apart.
    term_removed: bool


class ParaphraseSet(_Strict):
    """The pairs of one course, the rules they are judged by, and a hash naming exactly this set."""

    course_id: str
    rules: RelationRules
    pairs: tuple[ParaphrasePair, ...]
    # The hash of the pair file and its Q&A set file together: two runs with the same value searched the same texts.
    sha256: str


def load_paraphrase_set(path: Path) -> ParaphraseSet:
    """Read a pair file and its Q&A set and join them; a pair that is not a valid rewording is refused.

    Refused: a Q&A set of another course, a pair whose question is not in the Q&A set or has no gold section, and a
    rewording that contains a key term of its question (matched at the start of a word, as the set check does).
    """
    content = path.read_bytes()
    raw = _SetFile.model_validate(tomllib.loads(content.decode("utf-8")))
    qa_path = path.parent / raw.qa_file
    qa = load_qa_set(qa_path)
    if qa.course_id != raw.course_id:
        raise ValueError(f"the pairs are for {raw.course_id!r}, the Q&A set {raw.qa_file} for {qa.course_id!r}")
    pairs = tuple(_pair(line, qa) for line in raw.pairs)
    digest = hashlib.sha256(content + b"\0" + qa_path.read_bytes()).hexdigest()
    return ParaphraseSet(course_id=raw.course_id, rules=raw.relations, pairs=pairs, sha256=digest)


def _pair(line: _PairLine, qa: QaSet) -> ParaphrasePair:
    """Join one pair line with its Q&A case, or say why the pair cannot be used."""
    case = next((case for case in qa.cases if case.id == line.id), None)
    if case is None:
        raise ValueError(f"{line.id}: no such question in the Q&A set")
    if case.gold_section is None:
        raise ValueError(f"{line.id}: the question is unanswerable, so search has nothing to keep")
    kept = [term for term in case.key_terms if term_occurs(normal_text(term), normal_text(line.paraphrase))]
    if kept:
        raise ValueError(f"{line.id}: the rewording still contains the key terms {kept}")
    question = normal_text(case.question)
    return ParaphrasePair(
        case_id=case.id,
        question=case.question,
        paraphrase=line.paraphrase,
        gold_section=case.gold_section,
        term_removed=any(term_occurs(normal_text(term), question) for term in case.key_terms),
    )
