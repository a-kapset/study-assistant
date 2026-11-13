"""The search store's statements against a real PostgreSQL with pgvector: which passages a search reads and in
what order, by meaning and by meaning and words together, and how a course's chunks and vectors are counted per
embedding space.

The unit tests of the search rules use a store that returns given passages best first and given counts; these
tests check that the real statements produce exactly that. Courses get their rows through a real ingestion, and a
new embedding space through the migration run, the way the stack gets them.
"""

import math
import shutil
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from opentelemetry.trace import NoOpTracerProvider
from psycopg import sql

from study_assistant.courses.models import CourseManifest
from study_assistant.courses.registry import load_course
from study_assistant.db import PostgresDatabase
from study_assistant.ingest.chunker import ChunkSize
from study_assistant.ingest.repository import PostgresCorpusStore
from study_assistant.ingest.service import CourseIngestor
from study_assistant.retrieval.repository import Coverage, HybridQuery, PostgresSearchStore

if TYPE_CHECKING:
    from .conftest import FixedLengthEmbedder, MigratedDatabase

from .conftest import APP_ROLE, APP_ROLE_PASSWORD

pytestmark = [pytest.mark.database, pytest.mark.anyio]

# The repository's own course folder: app/tests/integration/<this file> -> repository root.
REPO_COURSES = Path(__file__).resolve().parents[3] / "courses"
SIZE = ChunkSize(size=1000, overlap=100)
DEMO_CHUNKS = 12
# A second course with the demo course's files, so both courses hold passages with the same texts and scores.
OTHER_COURSE = "demo_copy"
# An invented revision of the configured model that only this file configures, so its space is new here.
NEW_REVISION = "fedcba9876543210fedcba9876543210fedcba98"


@pytest.fixture
async def search_store(postgres: MigratedDatabase) -> AsyncIterator[PostgresSearchStore]:
    """The app's search store, connected as the app's role through the app's connection pool."""
    database = PostgresDatabase(postgres.settings(APP_ROLE, APP_ROLE_PASSWORD), NoOpTracerProvider())
    await database.open()
    try:
        yield PostgresSearchStore(database)
    finally:
        await database.close()


async def _ingest_two_courses(store: PostgresCorpusStore, embedder: FixedLengthEmbedder, root: Path) -> None:
    """Ingest the demo course and a copy of it under another id, both from folders under ``root``."""
    demo = load_course(REPO_COURSES, "demo")
    for course_id in ("demo", OTHER_COURSE):
        shutil.copytree(REPO_COURSES / "demo", root / course_id)
        manifest: CourseManifest = demo.model_copy(update={"id": course_id})
        await CourseIngestor(store, embedder, root, SIZE).ingest(manifest)


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors, computed here without the database."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


@pytest.mark.covers("US-006", "AC-1")
async def test_nearest_returns_the_k_closest_passages_of_the_requested_course_with_their_similarity(
    postgres: MigratedDatabase,
    store: PostgresCorpusStore,
    search_store: PostgresSearchStore,
    embedder: FixedLengthEmbedder,
    tmp_path: Path,
) -> None:
    """Ingests the demo course and a copy of it with the same texts, then asks for the 5 passages of the demo
    course closest to a question.

    Expects demo chunks only, and the scores of the 5 most similar demo chunks, computed here from the same vectors,
    highest first. Catches a statement without the course filter (the copy has passages with the same scores), the
    wrong sort direction, a missing limit, and a distance reported as the score.
    """
    await _ingest_two_courses(store, embedder, tmp_path)
    with postgres.owner_connection() as conn:
        demo_chunks: dict[str, str] = dict(
            conn.execute("SELECT chunk_id, text FROM chunks WHERE course_id = 'demo'").fetchall()
        )
    query = await embedder.aembed_query("How many dice does a player roll on the first turn?")
    passage_vectors = await embedder.aembed_documents(list(demo_chunks.values()))
    expected = sorted((_cosine(query, vector) for vector in passage_vectors), reverse=True)[:5]

    passages = await search_store.nearest("demo", embedder.space, query, k=5)

    foreign = [p.chunk_id for p in passages if p.chunk_id not in demo_chunks]
    assert foreign == [], f"the search of the demo course returned chunks of another course: {foreign}"
    scores = [p.score for p in passages]
    assert scores == pytest.approx(expected, abs=1e-6), f"expected the top-5 similarities {expected}, got {scores}"


@pytest.mark.covers("US-006", "AC-5")
async def test_coverage_counts_the_courses_chunks_and_those_with_a_vector_in_the_given_space(
    postgres: MigratedDatabase,
    store: PostgresCorpusStore,
    search_store: PostgresSearchStore,
    embedder: FixedLengthEmbedder,
    tmp_path: Path,
) -> None:
    """Ingests the demo course and its copy, then configures a new revision of the model, so the migration run
    prepares a new, still empty space.

    Expects 12 chunks with 12 vectors for the demo course in the first space and 12 chunks with 0 vectors in the
    new one. Catches counts that include the other course, vectors counted in another space's table, and chunks
    counted as embedded without a vector, any of which would let a search run over part of a course.
    """
    await _ingest_two_courses(store, embedder, tmp_path)
    new_space = embedder.with_settings(embedder.settings.model_copy(update={"revision": NEW_REVISION})).space
    assert not postgres.space_registered(new_space), "another test already prepared this space"
    postgres.migrate(embedder.settings.model_copy(update={"revision": NEW_REVISION}))

    counted = (
        await search_store.coverage("demo", embedder.space),
        await search_store.coverage("demo", new_space),
    )

    expected = (Coverage(DEMO_CHUNKS, DEMO_CHUNKS), Coverage(DEMO_CHUNKS, 0))
    assert counted == expected, f"expected {expected} for the first and the new space, got {counted}"


# A question vector the tests choose themselves. Against the stored vectors (the text length, then ones) it ranks
# shorter chunks as closer, so the order by meaning is known before a search.
ONES = [1.0] * 384


def _hybrid(text: str, min_score: float = -1.0, k: int = DEMO_CHUNKS) -> HybridQuery:
    """A hybrid query for ``text`` with the chosen vector, ranking every demo chunk before fusion."""
    return HybridQuery(vector=ONES, text=text, min_score=min_score, k=k, window=DEMO_CHUNKS, rrf_constant=60)


async def _demo_by_meaning(postgres: MigratedDatabase, embedder: FixedLengthEmbedder) -> list[tuple[str, float]]:
    """The demo chunks with their similarity to the chosen vector, in the search's order: most similar first, equal
    similarities by chunk id."""
    with postgres.owner_connection() as conn:
        rows = conn.execute("SELECT chunk_id, text FROM chunks WHERE course_id = 'demo'").fetchall()
    texts: dict[str, str] = dict(rows)
    vectors = await embedder.aembed_documents(list(texts.values()))
    scored = [(chunk_id, _cosine(ONES, vector)) for chunk_id, vector in zip(texts, vectors, strict=True)]
    return sorted(scored, key=lambda pair: (-pair[1], pair[0]))


def _word_only_in(postgres: MigratedDatabase, chunk_id: str) -> str:
    """Return a word of the chunk that no other demo chunk holds, written so a question turns it into itself."""
    with postgres.owner_connection() as conn:
        inside = sql.SQL("SELECT text_search FROM chunks WHERE chunk_id = {}").format(chunk_id).as_string(conn)
        others = (
            sql.SQL("SELECT text_search FROM chunks WHERE course_id = 'demo' AND chunk_id <> {}")
            .format(chunk_id)
            .as_string(conn)
        )
        statement = sql.SQL(
            "SELECT word FROM ts_stat({inside}) WHERE word ~ '^[a-z]{{4,}}$'"
            " AND plainto_tsquery('english', word)::text = quote_literal(word)"
            " EXCEPT SELECT word FROM ts_stat({others}) ORDER BY 1 LIMIT 1"
        ).format(inside=inside, others=others)
        row = conn.execute(statement).fetchone()
    assert row is not None, f"chunk {chunk_id} holds no word of its own; the demo course changed"
    return str(row[0])


@pytest.mark.covers("US-018", "AC-3")
async def test_hybrid_puts_first_a_passage_that_holds_the_questions_word_but_is_last_by_meaning(
    postgres: MigratedDatabase,
    store: PostgresCorpusStore,
    search_store: PostgresSearchStore,
    embedder: FixedLengthEmbedder,
    tmp_path: Path,
) -> None:
    """Ingests the demo course, takes the chunk that is last by meaning for the chosen vector, and asks for one hit
    with a word only that chunk holds.

    Expects that chunk as the hit: first by words and last by meaning beats first by meaning alone. Catches a hybrid
    search whose ranking by words is lost, empty or not added in, which would return the hits of the search by
    meaning.
    """
    await _ingest_two_courses(store, embedder, tmp_path)
    last_by_meaning = (await _demo_by_meaning(postgres, embedder))[-1][0]
    word = _word_only_in(postgres, last_by_meaning)

    fused = await search_store.hybrid("demo", embedder.space, _hybrid(word, k=1))

    hits = [p.chunk_id for p in fused.passages]
    assert hits == [last_by_meaning], f"the chunk holding '{word}' should come first, got {hits}"


@pytest.mark.covers("US-018", "AC-4")
async def test_hybrid_leaves_out_a_passage_below_the_lowest_score_even_when_it_holds_the_questions_word(
    postgres: MigratedDatabase,
    store: PostgresCorpusStore,
    search_store: PostgresSearchStore,
    embedder: FixedLengthEmbedder,
    tmp_path: Path,
) -> None:
    """Ingests the demo course, sets the lowest score just above the chunk that is last by meaning, and searches
    with a word only that chunk holds.

    Expects every hit to reach the lowest score, and that chunk not to be a hit. Catches a ranking by words made
    from all chunks of the course: a question that shares any word with the course would then never be empty, and a
    question outside the course would get passages as if the course answered it.
    """
    await _ingest_two_courses(store, embedder, tmp_path)
    by_meaning = await _demo_by_meaning(postgres, embedder)
    last, last_score = by_meaning[-1]
    min_score = (last_score + min(score for _, score in by_meaning if score > last_score)) / 2
    word = _word_only_in(postgres, last)

    fused = await search_store.hybrid("demo", embedder.space, _hybrid(word, min_score=min_score))

    below = [
        (p.chunk_id, p.score) for p in fused.passages if p.score is None or p.score < min_score or p.chunk_id == last
    ]
    assert below == [], f"passages below the lowest score {min_score} entered through their words: {below}"


@pytest.mark.covers("US-018", "AC-3")
async def test_hybrid_ranks_by_words_the_passages_that_hold_any_word_of_the_question(
    postgres: MigratedDatabase,
    store: PostgresCorpusStore,
    search_store: PostgresSearchStore,
    embedder: FixedLengthEmbedder,
    tmp_path: Path,
) -> None:
    """Ingests the demo course and asks a question of two words, each held by a different single chunk.

    Expects both chunks in the ranking by words. Catches a ranking that needs every word of the question in one
    passage: questions rarely have all their words in one passage, so the ranking by words would be empty for most
    of them and the hybrid search would quietly equal the search by meaning.
    """
    await _ingest_two_courses(store, embedder, tmp_path)
    by_meaning = await _demo_by_meaning(postgres, embedder)
    question = f"{_word_only_in(postgres, by_meaning[0][0])} {_word_only_in(postgres, by_meaning[-1][0])}"

    fused = await search_store.hybrid("demo", embedder.space, _hybrid(question))

    assert fused.by_words == 2, f"'{question}' should find 2 passages by words, found {fused.by_words}"


@pytest.mark.covers("US-018", "AC-5")
async def test_hybrid_hits_of_a_smaller_request_are_the_first_hits_of_a_larger_one(
    postgres: MigratedDatabase,
    store: PostgresCorpusStore,
    search_store: PostgresSearchStore,
    embedder: FixedLengthEmbedder,
    tmp_path: Path,
) -> None:
    """Ingests the demo course and picks a chunk at the fourth place or lower by meaning whose id sorts after the
    first chunk's; a word of its own makes it first by words. Asks for 3 hits, then for all 12.

    Expects the 3 hits to be the first 3 of the 12. Catches rankings cut to the number of hits asked for: with 3,
    the picked chunk would lose its place by meaning, tie with the first chunk and fall behind it by id.
    """
    await _ingest_two_courses(store, embedder, tmp_path)
    by_meaning = [chunk_id for chunk_id, _ in await _demo_by_meaning(postgres, embedder)]
    picked = next((c for c in by_meaning[3:] if c > by_meaning[0]), None)
    assert picked is not None, "no chunk below the third place sorts after the first; the demo course changed"
    word = _word_only_in(postgres, picked)

    small = await search_store.hybrid("demo", embedder.space, _hybrid(word, k=3))
    large = await search_store.hybrid("demo", embedder.space, _hybrid(word, k=DEMO_CHUNKS))

    first, expected = [p.chunk_id for p in small.passages], [p.chunk_id for p in large.passages][:3]
    assert first == expected, f"3 hits should be the first 3 of 12 ({expected}), got {first}"


@pytest.mark.covers("US-018", "AC-9")
async def test_lexical_returns_only_the_requested_courses_passages_holding_a_word_of_the_question_without_a_score(
    postgres: MigratedDatabase,
    store: PostgresCorpusStore,
    search_store: PostgresSearchStore,
    embedder: FixedLengthEmbedder,
    tmp_path: Path,
) -> None:
    """Ingests the demo course and a copy of it with the same texts, takes two demo chunks with a word of their own
    each, and searches the demo course by words for a question holding both words and a word no chunk holds.

    Expects exactly those two chunks, each without a score. Catches a search by words that reads another course
    (each chunk would come twice, once per course), one that needs every word of the question in a passage (it
    would find nothing), and a rank by words passed on as a similarity score.
    """
    await _ingest_two_courses(store, embedder, tmp_path)
    by_meaning = await _demo_by_meaning(postgres, embedder)
    first, last = by_meaning[0][0], by_meaning[-1][0]
    question = f"{_word_only_in(postgres, first)} {_word_only_in(postgres, last)} zyxwvut"

    passages = await search_store.lexical("demo", question, DEMO_CHUNKS)

    found = sorted((p.chunk_id, p.score) for p in passages)
    expected = sorted([(first, None), (last, None)])
    assert found == expected, f"'{question}' should find {expected} by words, found {found}"


@pytest.mark.covers("US-018", "AC-9")
async def test_lexical_finds_nothing_for_a_question_of_stop_words_only(
    store: PostgresCorpusStore,
    search_store: PostgresSearchStore,
    embedder: FixedLengthEmbedder,
    tmp_path: Path,
) -> None:
    """Ingests the demo course and searches it by words for a question made only of words the course language
    ignores ("what is it").

    Expects no passage and no error. Catches a statement that fails on the empty word query such a question gives
    (a learner's short question would end in a 500 while the embedding server is down) and one that then matches
    every passage.
    """
    await _ingest_two_courses(store, embedder, tmp_path)

    passages = await search_store.lexical("demo", "what is it", DEMO_CHUNKS)

    assert passages == [], f"a question of stop words found {[p.chunk_id for p in passages]}"
