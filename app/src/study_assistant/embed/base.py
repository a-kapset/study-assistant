"""Embedding spaces and what ingestion and search need from an embedder.

Two vectors can only be compared when the same model computed them in the same way. An embedding space names
the model, its revision and the passage and query prefixes, all taken from the configuration. Every stored
vector belongs to one space, and each space has its own table, so vectors of different spaces are never mixed
or counted together. How the vectors are computed (the embedding server, our client and their libraries) is not
part of the space: a change there is accepted only when a comparison with reference vectors shows that the
vectors stayed the same.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Protocol, Self

from study_assistant.settings import EmbeddingSettings

# Characters of the model name kept in a space id, so that table and index names built from the id stay well
# below PostgreSQL's limit of 63 characters.
_NAME_LENGTH = 24
# Hex digits of the identity hash kept in a space id: far more than needed to tell apart the few spaces of one
# database.
_HASH_LENGTH = 12


@dataclass(frozen=True)
class EmbeddingSpace:
    """Everything that decides which vector a text gets; two spaces are the same only when all of it is equal."""

    model: str
    revision: str
    passage_prefix: str
    query_prefix: str

    @classmethod
    def from_settings(cls, settings: EmbeddingSettings) -> Self:
        """Name the space of the configured model and prefixes."""
        return cls(
            model=settings.model,
            revision=settings.revision,
            passage_prefix=settings.passage_prefix,
            query_prefix=settings.query_prefix,
        )

    @property
    def id(self) -> str:
        """A short id that is safe in table names: the model's name, then a hash of all identity fields.

        For example ``multilingual_e5_small_1a2b3c4d5e6f``. The name is there for people; the hash changes
        when any field changes, so two different spaces do not share an id. The fields are hashed as a JSON
        list, so no prefix text can make two different field lists look the same.
        """
        fields = [
            self.model,
            self.revision,
            self.passage_prefix,
            self.query_prefix,
        ]
        digest = hashlib.sha256(json.dumps(fields).encode("utf-8")).hexdigest()[:_HASH_LENGTH]
        model_name = self.model.rsplit("/", 1)[-1].lower()
        name = re.sub(r"[^a-z0-9]+", "_", model_name)[:_NAME_LENGTH].strip("_")
        return f"{name}_{digest}"

    @property
    def vector_table(self) -> str:
        """The name of the table that holds this space's vectors: "vectors_" and the space id.

        For example ``vectors_multilingual_e5_small_1a2b3c4d5e6f``. The migration run creates the table; ingestion
        writes to it and the course endpoints count its rows, all through this one name.
        """
        return f"vectors_{self.id}"


class Embedder(Protocol):
    """What ingestion, search and the readiness check need from an embedding model: the space it computes, its
    vectors, and whether it can answer now.

    The embedder reports its own space, so vectors are always stored under the space that really made them,
    never under one taken from somewhere else. The methods are asynchronous because every call waits for the
    embedding server; the names of the embedding methods match the asynchronous methods of LangChain's embeddings
    interface.
    """

    @property
    def space(self) -> EmbeddingSpace:
        """The embedding space of every vector this embedder returns."""
        ...

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Turn course passages into vectors, one per text, in the same order; the passage prefix is added."""
        ...

    async def aembed_query(self, text: str) -> list[float]:
        """Turn one search query into a vector; the query prefix is added."""
        ...

    async def ping(self) -> None:
        """Return when the embedding model can compute vectors now; raise an error when it cannot."""
        ...
