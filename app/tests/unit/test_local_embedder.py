"""Tests of the local embedder with the real model files from the repository's ``models/`` folder.

The model files are not in the repository. Without them the tests that need the model fail and name the folder
they looked in; they are never skipped, so a missing model cannot look like a green run.
"""

import json
import math
import re
from pathlib import Path

import pytest

from study_assistant.embed.local import EmbedderLoadError, LocalEmbedder, model_path
from study_assistant.settings import EmbeddingSettings

# The repository root: app/tests/unit/<this file> is three levels below it.
MODELS = Path(__file__).resolve().parents[3] / "models"
# Vectors made once by sentence-transformers on the same model revision, outside the app (texts are invented).
REFERENCE = json.loads((Path(__file__).parent / "data" / "embedder_reference.json").read_text(encoding="utf-8"))
# Equal vectors give 1.0. The tolerance only allows rounding differences between machines; a lost or swapped
# prefix gives far less.
MIN_COSINE = 0.9999


def _settings(**changes: object) -> EmbeddingSettings:
    """Settings of the reference model, read from the repository's ``models/`` folder, with ``changes`` applied."""
    values: dict[str, object] = {
        "model": REFERENCE["model"],
        "revision": REFERENCE["revision"],
        "dimension": 384,
        "passage_prefix": "passage: ",
        "query_prefix": "query: ",
        "model_dir": MODELS,
    }
    return EmbeddingSettings.model_validate(values | changes)


def _cosine(first: list[float], second: list[float]) -> float:
    """How close two vectors point: 1.0 for the same direction."""
    dot = sum(a * b for a, b in zip(first, second, strict=True))
    return dot / math.sqrt(sum(a * a for a in first) * sum(b * b for b in second))


@pytest.fixture(scope="module")
def embedder() -> LocalEmbedder:
    """The embedder on the real model, loaded once for the module because loading takes several seconds."""
    return LocalEmbedder.load(_settings())


@pytest.mark.covers("US-004", "AC-22")
def test_missing_model_files_are_named(tmp_path: Path) -> None:
    """Points the model folder at an empty directory and expects a load error that names the missing folder.

    Catches a load that fails with a library error about a download or a repository name instead, which does
    not say where the files were expected.
    """
    settings = _settings(model_dir=tmp_path)

    with pytest.raises(EmbedderLoadError, match=re.escape(str(model_path(settings)))):
        LocalEmbedder.load(settings)


def test_wrong_dimension_stops_the_load() -> None:
    """Configures 383 dimensions for a model that gives 384 and expects a load error that shows both lengths.

    Catches a load that accepts the model anyway: the vector table is created with the configured length, so
    the first ingest would fail far from the cause, inside the database.
    """
    with pytest.raises(EmbedderLoadError, match=r"length 384, but the configured dimension is 383"):
        LocalEmbedder.load(_settings(dimension=383))


@pytest.mark.covers("US-004", "AC-21")
def test_vectors_match_the_reference_implementation(embedder: LocalEmbedder) -> None:
    """Embeds the reference passages in one batch and the reference query, and expects each vector to match.

    The texts include a Russian one and one longer than the model's 512-token limit. Catches a lost, swapped
    or doubled passage or query prefix, and a library upgrade that silently changes the vectors: the space id
    does not include library versions, so new vectors would be mixed with stored ones without any error.
    """
    passages = embedder.embed_documents([item["text"] for item in REFERENCE["passages"]])
    queries = [embedder.embed_query(item["text"]) for item in REFERENCE["queries"]]
    expected = [item["vector"] for item in REFERENCE["passages"] + REFERENCE["queries"]]

    cosines = [_cosine(got, want) for got, want in zip(passages + queries, expected, strict=True)]

    assert min(cosines) >= MIN_COSINE, cosines
