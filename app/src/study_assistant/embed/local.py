"""A local embedding model run by sentence-transformers through LangChain's ``HuggingFaceEmbeddings``.

The model's own files decide how vectors are made (mean pooling, normalization, the 512-token limit), so this
module only chooses the model files, adds the passage and query prefixes and checks the model before it is used.
"""

from pathlib import Path
from typing import Self

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from study_assistant.embed.base import EmbeddingSpace
from study_assistant.settings import EmbeddingSettings

# How vectors are made: sentence-transformers on the model's own configuration. Raise the number with any change
# in how this module calls it: vectors made before and after the change belong to different embedding spaces.
EMBEDDER_VERSION = "sentence-transformers/1"
# The file that makes a folder a sentence-transformers model: it lists the steps from tokens to one vector.
MODULES_FILE = "modules.json"
# Any text works; the load check only needs the length of one vector.
_PROBE_TEXT = "length check"


class EmbedderLoadError(Exception):
    """The model files cannot be used: they are missing, or the model does not give vectors of the configured length."""


def model_path(settings: EmbeddingSettings) -> Path:
    """Where the files of the configured model revision are kept: ``<model_dir>/<model id>/<revision>``.

    Each revision has its own folder, so the files of two revisions never overwrite each other.
    """
    return settings.model_dir / settings.model / settings.revision


class LocalEmbedder(Embeddings):
    """Turns course passages and queries into vectors of one embedding space, on the CPU of this machine."""

    def __init__(self, space: EmbeddingSpace, model: Embeddings) -> None:
        """Keep a loaded model; ``load`` builds it from the configured files."""
        self.space = space
        self._model = model

    @classmethod
    def load(cls, settings: EmbeddingSettings) -> Self:
        """Load the configured model files; missing files or vectors of another length are an ``EmbedderLoadError``."""
        path = model_path(settings)
        if not (path / MODULES_FILE).is_file():
            raise EmbedderLoadError(f"model files not found at {path} (no {MODULES_FILE})")
        model = HuggingFaceEmbeddings(
            model_name=str(path),
            # The CPU is named, so the library never picks another device by itself, and only the local files
            # are read, so a missing file is never downloaded from somewhere else in its place.
            model_kwargs={"device": "cpu", "local_files_only": True},
            # No normalization option: the model's own steps (modules.json) already scale every vector to length 1.
            encode_kwargs={"prompt": settings.passage_prefix},
            query_encode_kwargs={"prompt": settings.query_prefix},
        )
        width = len(model.embed_query(_PROBE_TEXT))
        if width != settings.dimension:
            raise EmbedderLoadError(
                f"{settings.model}: the model gives vectors of length {width}, "
                f"but the configured dimension is {settings.dimension}"
            )
        return cls(EmbeddingSpace.from_settings(settings, EMBEDDER_VERSION), model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Turn course passages into vectors, one per text, in the same order; the passage prefix is added."""
        return self._model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Turn one search query into a vector; the query prefix is added."""
        return self._model.embed_query(text)
