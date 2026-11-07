"""Unit tests of the paraphrase relation: how one pair is classified and how a group of pairs is judged, on made-up
scores instead of a run."""

import pytest

from evalharness.oracles.retrieval import CaseScore, score_case
from evalharness.relations.paraphrase import (
    PairOutcome,
    RelationSummary,
    answer_kept_verdict,
    overlap,
    pair_outcome,
    summarise,
    top_k_stable_verdict,
)

HIT = score_case("1.1", ["1.1"], 5)
MISS = score_case("1.1", ["2.1"], 5)
FAILED = score_case("1.1", None, 5)


@pytest.mark.covers("US-015", "AC-4")
def test_overlap_counts_the_passages_both_searches_returned_over_the_k_places_in_any_order() -> None:
    """Compares two top-5 lists that share three passages in another order, and two lists cut short to two
    passages each that share both.

    A pass means the overlap is the number of shared passages divided by k, whatever their order: 3/5 and 2/5. A
    failure means the stable share would count a reordered but equal list as a change (order counted), or a short
    list as fully stable (divided by its own length), and the ranking-stability verdict would mislead.
    """
    shared_in_other_order = overlap(["a", "b", "c", "d", "e"], ["c", "x", "a", "y", "b"], 5)
    short_lists = overlap(["a", "b"], ["b", "a"], 5)

    assert (shared_in_other_order, short_lists) == (0.6, 0.4), (
        f"overlap of 3 shared passages in other order {shared_in_other_order}, of 2 short lists {short_lists}"
    )


@pytest.mark.covers("US-015", "AC-2")
@pytest.mark.parametrize(
    ("question", "rewording", "found"),
    [(HIT, HIT, "kept"), (HIT, MISS, "lost"), (MISS, HIT, "gained"), (MISS, MISS, "neither"), (HIT, FAILED, "failed")],
)
def test_a_pair_is_classified_by_which_of_its_searches_found_the_gold_section(
    question: CaseScore, rewording: CaseScore, found: str
) -> None:
    """Classifies a pair for every combination of the question's and the rewording's result.

    A pass means a pair is "lost" only when the question found the gold section and the rewording did not, and a
    pair with a failed search is set apart. A failure means the counts the verdict is computed from are wrong, e.g.
    lost and gained swapped, which would turn a search that loses answers into one that gains them.
    """
    outcome = pair_outcome(question, rewording, ["c1"], ["c1"], 5)

    assert outcome.found == found, f"question {question.verdict}, rewording {rewording.verdict}: got {outcome.found}"


def _summary(kept: int, lost: int, gained: int, neither: int = 0, stable: int = 0, failed: int = 0) -> RelationSummary:
    """Build the counts of a group of pairs."""
    pairs = kept + lost + gained + neither + failed
    return RelationSummary(
        pairs=pairs, failed=failed, kept=kept, lost=lost, gained=gained, neither=neither, stable=stable
    )


@pytest.mark.covers("US-015", "AC-2")
@pytest.mark.parametrize(
    ("lost", "gained", "verdict"),
    [(0, 0, "holds"), (0, 3, "holds"), (2, 1, "not_confirmed"), (11, 2, "violated")],
)
def test_answer_kept_is_violated_only_when_rewordings_lose_answers_beyond_chance(
    lost: int, gained: int, verdict: str
) -> None:
    """Judges 82 pairs with no change, with three gains, with two losses and one gain, and with the eleven losses
    and two gains of the first real run.

    A pass means the relation holds when nothing was lost or rewordings found more, stays "not confirmed" when a
    few losses could be chance, and is violated when the interval of the change lies below zero. A failure means
    gains are counted as losses, or a small loss within noise is called a violation, or a clear loss passes.
    """
    summary = _summary(kept=82 - lost - gained, lost=lost, gained=gained)

    assert answer_kept_verdict(summary, 0.05) == verdict, f"{lost} lost, {gained} gained of 82"


@pytest.mark.covers("US-015", "AC-4")
@pytest.mark.parametrize(("stable", "verdict"), [(70, "holds"), (60, "not_confirmed"), (22, "violated")])
def test_top_k_stable_is_judged_by_the_interval_of_the_stable_share(stable: int, verdict: str) -> None:
    """Judges 82 pairs with 70, 60 and 22 stable pairs against a target share of 0.7.

    A pass means 70 of 82 (0.85, lower end 0.76) holds, 60 of 82 (0.73, lower end 0.63) is not confirmed, and 22 of
    82 is violated. A failure means the verdict is taken from the measured share alone, so a share just above the
    target on 82 pairs would be called proven.
    """
    summary = _summary(kept=82, lost=0, gained=0, stable=stable)

    assert top_k_stable_verdict(summary, 0.7) == verdict, f"{stable} of 82 stable"


@pytest.mark.covers("US-015", "AC-2", "AC-4")
def test_a_failed_search_leaves_both_relations_without_a_verdict() -> None:
    """Summarises three kept, stable pairs and one pair with a failed search.

    A pass means the failed pair is counted apart and neither relation gets a verdict, as in the answer run. A
    failure means a run where the app returned errors could still report "holds", although it measured fewer
    pairs than it claims.
    """
    outcomes = [PairOutcome(found="kept", overlap=1.0)] * 3 + [PairOutcome(found="failed", overlap=None)]
    summary = summarise(outcomes, 0.6)

    verdicts = (answer_kept_verdict(summary, 0.05), top_k_stable_verdict(summary, 0.7))

    assert verdicts == (None, None), f"failed {summary.failed}: verdicts {verdicts}"
