"""A local embedding model run by ONNX Runtime on the model's own ONNX export and its tokenizer, without torch.

The model gives one vector per token. A text's vector is the mean of the vectors of its real tokens (padding is
left out by the attention mask), scaled to length 1. This is what the model's reference implementation does
(sentence-transformers with mean pooling and normalization), and a test compares our vectors with vectors it
produced.
"""

from pathlib import Path
from typing import Self

import numpy as np
import onnxruntime as ort
import structlog
from langchain_core.embeddings import Embeddings
from numpy.typing import NDArray
from tokenizers import Encoding, Tokenizer

from study_assistant.embed.base import EmbeddingSpace
from study_assistant.settings import EmbeddingSettings

# How vectors are made from the model's output: ONNX Runtime, truncation at the model's limit, mean pooling over
# the attention mask, scaling to length 1. Raise the number with any change in that chain: vectors made before
# and after the change belong to different embedding spaces.
EMBEDDER_VERSION = "onnx-mean-l2/1"
# The tokenizer file in the model's repository; the weights file is configured.
TOKENIZER_FILE = "tokenizer.json"
# Longest input in tokens, special tokens included. The model has 512 positions and fails on a longer input,
# so longer texts are cut, as the reference implementation does.
MAX_TOKENS = 512
# Texts sent to the model in one run; bounds the memory one run needs.
BATCH_SIZE = 32


class EmbedderLoadError(Exception):
    """The model files cannot be used: one is missing, or the model does not give vectors of the configured length."""


def model_file(settings: EmbeddingSettings, file: str) -> Path:
    """Where a file of the configured model revision is kept: ``<model_dir>/<model id>/<revision>/<file>``.

    Each revision has its own folder, so the files of two revisions never overwrite each other.
    """
    return settings.model_dir / settings.model / settings.revision / file


class LocalEmbedder(Embeddings):
    """Turns course passages and queries into vectors of one embedding space, on the CPU of this machine."""

    def __init__(self, space: EmbeddingSpace, session: ort.InferenceSession, tokenizer: Tokenizer) -> None:
        """Keep a loaded model and its tokenizer; ``load`` builds them from the configured files."""
        self.space = space
        self._session = session
        self._tokenizer = tokenizer
        # The graph's own input names: some exports also take token type ids, others do not.
        self._inputs = {node.name for node in session.get_inputs()}
        self._log: structlog.typing.FilteringBoundLogger = structlog.get_logger(__name__)

    @classmethod
    def load(cls, settings: EmbeddingSettings) -> Self:
        """Load the configured model files; a missing file or vectors of another length is an ``EmbedderLoadError``."""
        weights = model_file(settings, settings.weights_file)
        vocabulary = model_file(settings, TOKENIZER_FILE)
        for name, path in ((settings.weights_file, weights), (TOKENIZER_FILE, vocabulary)):
            if not path.is_file():
                raise EmbedderLoadError(f"{name}: model file not found at {path}")
        # The CPU is named explicitly, so the runtime never picks another device by itself.
        session = ort.InferenceSession(str(weights), providers=["CPUExecutionProvider"])
        width = session.get_outputs()[0].shape[-1]
        if width != settings.dimension:
            raise EmbedderLoadError(
                f"{settings.weights_file}: the model gives vectors of length {width}, "
                f"but the configured dimension is {settings.dimension}"
            )
        tokenizer = Tokenizer.from_file(str(vocabulary))
        tokenizer.enable_truncation(max_length=MAX_TOKENS)
        # Padding fills shorter texts of a batch up to the longest one. The attention mask hides those
        # positions from the model and from the mean, so the padding id never changes a vector.
        tokenizer.enable_padding()
        return cls(EmbeddingSpace.from_settings(settings, EMBEDDER_VERSION), session, tokenizer)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Turn course passages into vectors, one per text, in the same order; the passage prefix is added."""
        return self._embed([self.space.passage_prefix + text for text in texts])

    def embed_query(self, text: str) -> list[float]:
        """Turn one search query into a vector; the query prefix is added."""
        return self._embed([self.space.query_prefix + text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches and log how many of them were too long and were cut."""
        vectors: list[list[float]] = []
        cut = 0
        for start in range(0, len(texts), BATCH_SIZE):
            encodings = self._tokenizer.encode_batch(texts[start : start + BATCH_SIZE])
            cut += sum(1 for encoding in encodings if encoding.overflowing)
            vectors.extend(self._vectors(encodings))
        if cut:
            self._log.warning("embedding_input_cut", texts=cut, of=len(texts), max_tokens=MAX_TOKENS)
        return vectors

    def _vectors(self, encodings: list[Encoding]) -> list[list[float]]:
        """Run the model on one batch: the mean of each text's real token vectors, scaled to length 1."""
        mask = np.array([encoding.attention_mask for encoding in encodings], dtype=np.int64)
        feed = {
            "input_ids": np.array([encoding.ids for encoding in encodings], dtype=np.int64),
            "attention_mask": mask,
            "token_type_ids": np.array([encoding.type_ids for encoding in encodings], dtype=np.int64),
        }
        inputs = {name: array for name, array in feed.items() if name in self._inputs}
        tokens: NDArray[np.float32] = self._session.run(None, inputs)[0]
        weights = mask[:, :, np.newaxis].astype(np.float32)
        means = (tokens * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-9)
        lengths = np.linalg.norm(means, axis=1, keepdims=True)
        unit: list[list[float]] = (means / np.maximum(lengths, 1e-12)).tolist()
        return unit
