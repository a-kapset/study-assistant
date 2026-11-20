"""Which stage of the app the request traces point to for each mutant, and whether that is the part made faulty.

Every ask of a mutant run is compared with the asks of the same question in the run without a mutant, stage by
stage in the order an ask passes them. The first stage whose description is none of the reference's descriptions for
that question is the stage the ask names: the reference is allowed its own variation between repetitions. A mutant
is attributed to the stage most of its differing asks name. The verdict compares that stage with the part of the app
the catalog says the mutant makes faulty; a part that has no span in an ask's trace (how the course is cut into
passages happens before any ask) cannot be found from the trace at all, whatever stage the trace points to.

A replayed model call that had no recording gets an error from the proxy instead of the model's reply. That error
is the proxy's, not the mutant's, so an ask whose first difference is at the model provider or the answer step while
its own call or the reference's call for its question missed the recordings says nothing: it is "not measured".

A failed ask is compared like any other: unlike a judged cell, which leaves out what the stack failed on, the
attribution reads how a request went wrong, and a failure is often exactly the mutant's effect (a search declined
because the embedding server is unreachable). A slip of the stack on a few asks is outweighed by the majority rule
and shows in the run without a mutant at the end. Only an ask whose trace could not be read is left out.
"""

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, get_args

from evalharness.evaluation.answer_record import AnswerCaseRecord
from evalharness.mutation.catalog import Component
from evalharness.stages import STAGES, StageName
from evalharness.stats.intervals import Interval, share_interval

# What one ask says: the stage it names, "none" when every stage is as in the reference, "not_measured" when a
# missed recording may have made the difference, "left_out" when the ask or its question has no trace to compare.
type AskOutcome = StageName | Literal["none", "not_measured", "left_out"]
# What a mutant's asks together say.
type MutantStage = StageName | Literal["several", "none", "not_measured"]
type AttributionVerdict = Literal["right", "wrong", "several", "none", "not_attributable", "not_measured"]

# The stage whose span shows the work of each part of the app; None for a part with no span in an ask's trace.
FAULTY_STAGE: dict[Component, StageName | None] = {
    "embedding": "embedding",
    "search": "search",
    "ingestion": None,
    "answer": "generate",
    "provider": "provider",
}
# Stages that come after the model call: a reply missing from the recordings changes them by itself.
_AFTER_MODEL_CALL: frozenset[StageName] = frozenset({"provider", "generate"})
# Verdicts that leave a mutant out of the accuracy: the trace could not, or was not allowed to, show the part.
_NOT_COUNTED: frozenset[AttributionVerdict] = frozenset({"not_attributable", "not_measured"})


@dataclass(frozen=True)
class Attribution:
    """What the asks of one mutant run name, and the verdict against the part the mutant makes faulty."""

    stage: MutantStage
    # How many asks gave each outcome, for the report: the verdict alone hides how clear it was.
    counts: Mapping[AskOutcome, int]
    verdict: AttributionVerdict


@dataclass(frozen=True)
class Accuracy:
    """How many of the counted mutants of one part were attributed to its own stage."""

    right: int
    counted: int
    # None when no mutant of the part could be counted.
    interval: Interval | None


def named_stage(ask: Mapping[StageName, str], reference: Sequence[Mapping[StageName, str]]) -> StageName | None:
    """Return the first stage, in the order an ask passes them, whose description is none of the reference's; None
    when every stage matches some reference ask of the question."""
    for stage in STAGES:
        if all(ask[stage] != seen[stage] for seen in reference):
            return stage
    return None


def ask_outcomes(cases: Sequence[AnswerCaseRecord], reference: Sequence[AnswerCaseRecord]) -> list[AskOutcome]:
    """Return what every ask of a mutant run names against the reference run; an ask without stages, or whose question
    has no reference ask with stages, is left out. Failure labels are not read: a failed ask is compared too."""
    by_case: defaultdict[str, list[AnswerCaseRecord]] = defaultdict(list)
    for case in reference:
        if case.stages is not None:
            by_case[case.case_id].append(case)
    outcomes: list[AskOutcome] = []
    for case in cases:
        seen = by_case.get(case.case_id, [])
        if case.stages is None or not seen:
            outcomes.append("left_out")
            continue
        stage = named_stage(case.stages, [ask.stages for ask in seen if ask.stages is not None])
        if stage in _AFTER_MODEL_CALL and (_missed(case) or any(_missed(ask) for ask in seen)):
            outcomes.append("not_measured")
        else:
            outcomes.append(stage if stage is not None else "none")
    return outcomes


def _missed(case: AnswerCaseRecord) -> bool:
    """Whether a model call of the ask found no recording in the proxy and so got the proxy's error."""
    return "miss" in case.model_calls


def attribute(outcomes: Sequence[AskOutcome], noise: int, component: Component | None) -> Attribution:
    """Attribute one run from the outcomes of its asks.

    ``noise`` is how many asks named a stage in the run without a mutant at the end of the experiment: differences
    the app shows on its own. A run whose asks name stages no more often than that is attributed to "none" (or "not
    measured" when more asks than that could not be measured). Otherwise the run is attributed to the stage named by
    more than half of the asks that named one, or to "several" when no stage has that majority. ``component`` is the
    part the mutant makes faulty; None for a run without a mutant.
    """
    counts = Counter(outcomes)
    named = [stage for outcome in outcomes for stage in STAGES if outcome == stage]
    stage: MutantStage
    if len(named) <= noise:
        stage = "not_measured" if counts["not_measured"] > noise else "none"
    else:
        top, n = Counter(named).most_common(1)[0]
        stage = top if 2 * n > len(named) else "several"
    return Attribution(stage=stage, counts=dict(counts), verdict=verdict(component, stage))


def verdict(component: Component | None, stage: MutantStage) -> AttributionVerdict:
    """Compare the attributed stage with the stage of the faulty part."""
    if stage == "not_measured":
        return "not_measured"
    if component is not None and FAULTY_STAGE[component] is None:
        return "not_attributable"
    if stage == "none":
        return "none"
    if stage == "several":
        return "several"
    return "right" if component is not None and stage == FAULTY_STAGE[component] else "wrong"


def accuracy(verdicts: Sequence[tuple[Component, AttributionVerdict]]) -> dict[Component | Literal["all"], Accuracy]:
    """Return, per part of the app and over all parts, the share of counted mutants attributed to their own stage,
    with its Wilson interval. A mutant is the unit: the asks of one mutant share one fault, so they are not
    independent and would make the interval look far narrower than the evidence is."""
    groups: dict[Component | Literal["all"], list[AttributionVerdict]] = defaultdict(list)
    for component, value in verdicts:
        groups[component].append(value)
        groups["all"].append(value)
    result: dict[Component | Literal["all"], Accuracy] = {}
    for part in (*get_args(Component.__value__), "all"):
        counted = [value for value in groups.get(part, []) if value not in _NOT_COUNTED]
        right = sum(value == "right" for value in counted)
        result[part] = Accuracy(right, len(counted), share_interval(right, len(counted)) if counted else None)
    return result
