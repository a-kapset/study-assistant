"""Unit tests of the retrieval scores: hand-computed examples of the rules that decide what a search is worth."""

import pytest

from evalharness.oracles.retrieval import score_case, summarise

# The answer section of every example below; "1.2.4" is one of its subsections, "1.20" only looks like one.
GOLD = "1.2"
K = 5


@pytest.mark.covers("US-007", "AC-3")
@pytest.mark.parametrize(
    ("section", "counts"),
    [
        pytest.param("1.2", True, id="the-section"),
        pytest.param("1.2.4", True, id="a-subsection"),
        pytest.param("1.20", False, id="an-id-with-the-same-start"),
        pytest.param("1", False, id="the-parent-chapter"),
        pytest.param(None, False, id="text-before-any-heading"),
    ],
)
def test_a_hit_counts_only_for_the_answer_section_and_its_subsections(section: str | None, counts: bool) -> None:
    """Scores one search whose only hit comes from the given section and checks whether it was counted as found.

    A pass means a question is credited exactly when a passage of its answer section, or of a subsection, was found.
    A failure means the metric credits a wrong section (for example "1.20" for "1.2") or misses a right one, so every
    recall in a run would be too high or too low. Nothing else in the harness or the app checks this rule.
    """
    score = score_case(GOLD, [section], K)

    assert (score.verdict == "hit") is counts, f"a hit from section {section!r} for answer section {GOLD}: {score}"


@pytest.mark.covers("US-007", "AC-3")
def test_the_rank_is_the_position_in_the_returned_list() -> None:
    """Scores a search whose answer section comes second in the returned list and expects rank 2 and 0.5.

    A pass means the rank and the reciprocal rank describe the order the hits arrived in, which is the order the
    user and the answer step read them in. A failure means MRR no longer measures that order (for example the
    positions are counted from 0 or the hits are sorted again), so a ranking change would be scored wrongly.
    """
    score = score_case(GOLD, ["3.1", "1.2", "1.2.4"], K)

    assert (score.first_relevant_rank, score.reciprocal_rank) == (2, 0.5), (
        f"answer section second in the list must give rank 2 and reciprocal rank 0.5, got {score}"
    )


@pytest.mark.covers("US-007", "AC-4")
@pytest.mark.parametrize(
    ("sections", "verdict"),
    [
        pytest.param(None, "failed", id="no-result"),
        pytest.param([], "empty", id="nothing-above-the-lowest-score"),
        pytest.param(["3.1", "4.2"], "miss", id="only-other-sections"),
    ],
)
def test_each_kind_of_failure_gets_its_own_verdict(sections: list[str | None] | None, verdict: str) -> None:
    """Scores a failed search, an empty one and one that found only other sections, and checks each verdict.

    A pass means a run report can say how many questions failed because the search broke, because the threshold
    cut everything off, or because the ranking found the wrong sections. These need different fixes, so a failure
    here means the report would send the developer to the wrong place.
    """
    score = score_case(GOLD, sections, K)

    assert score.verdict == verdict, f"search result {sections!r} must be {verdict!r}, got {score.verdict!r}"


@pytest.mark.covers("US-007", "AC-4")
def test_failed_and_empty_searches_stay_in_the_group_with_zero() -> None:
    """Averages one found question, one empty search and one failed search, and expects recall and MRR of 1/3.

    A pass means a question whose search gave the user nothing lowers the result as much as a wrong answer does.
    A failure means such questions are dropped from the count, so a broken or over-strict search would show a
    better number than a working one. This is the rule that keeps outages and threshold mistakes visible.
    """
    scores = [score_case(GOLD, ["1.2"], K), score_case(GOLD, [], K), score_case(GOLD, None, K)]

    summary = summarise(scores)

    assert (summary.n, summary.recall, summary.mrr) == (3, 1 / 3, 1 / 3), (
        f"one found question of three must give n 3, recall 1/3 and MRR 1/3, got {summary}"
    )


@pytest.mark.covers("US-007", "AC-4")
def test_shares_are_counted_over_k_places_and_empty_places_count_as_neither() -> None:
    """Scores a search that returned two hits of five places, one from the answer section and one from another.

    It expects precision 1/5 and irrelevant 1/5: the three places the threshold left empty are neither. A failure
    means precision is counted over the returned hits only (a short list would look precise) or empty places are
    counted as wrong sections (an empty search would look like one full of noise).
    """
    score = score_case(GOLD, ["1.2", "3.1"], K)

    assert (score.precision, score.irrelevant) == (1 / K, 1 / K), (
        f"one right and one wrong hit in {K} places must give precision and irrelevant 1/{K}, got {score}"
    )
