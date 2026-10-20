"""Make the reference vectors for the embedding server check (tests/integration/test_embedding_server_reference.py).

The vectors come from the model's reference implementation, sentence-transformers on CPU, with the pinned model
revision downloaded from Hugging Face. The check then sends the same texts through the app's own client to the
running embedding server and expects the same vectors. Run it in a throwaway container, never in the app's
environment (the command is in app/README.md, "Reference vectors"). The texts are invented and course-neutral; a
Russian text and one longer than the model's limit of 512 tokens are among them, because both are cut or tokenized
in ways that can differ between implementations.

Make new vectors only for another model, revision or prefixes; never after an upgrade of the server image or of
the client libraries, because those are exactly the changes the check must catch.
"""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

LONG = " ".join(f"Rule {i} says the river token moves {i % 7 + 1} steps to the left." for i in range(90))
PASSAGES = [
    "Tiles are drawn face down.",
    "At the start of every round each player draws three cards from the shared deck, "
    "keeps two of them and passes the third card to the player on the left. A player "
    "who cannot pass a card skips the trading phase of that round.",
    "Каждый игрок в начале раунда берёт три карточки из общей колоды и оставляет две.",
    LONG,
]
QUERIES = ["How many cards does a player keep?"]

# The defaults are the model the stack runs now; for another one pass the APP_EMBEDDING__* values from .env.
parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--model", default="intfloat/multilingual-e5-small")
parser.add_argument("--revision", default="614241f622f53c4eeff9890bdc4f31cfecc418b3")
parser.add_argument("--passage-prefix", default="passage: ")
parser.add_argument("--query-prefix", default="query: ")
parser.add_argument("--out", type=Path, default=Path("embedder_reference.json"))
args = parser.parse_args()

model = SentenceTransformer(args.model, revision=args.revision, device="cpu")
# All passages in one call, as the check sends them in one batch through the app's client.
passages = model.encode(PASSAGES, prompt=args.passage_prefix, normalize_embeddings=True, batch_size=32)
queries = model.encode(QUERIES, prompt=args.query_prefix, normalize_embeddings=True, batch_size=32)
long_tokens = len(model.tokenizer(args.passage_prefix + LONG)["input_ids"])

result = {
    "model": args.model,
    "revision": args.revision,
    "passage_prefix": args.passage_prefix,
    "query_prefix": args.query_prefix,
    # How the vectors were made, so a later difference can be traced to its source.
    "reference": {
        "sentence-transformers": version("sentence-transformers"),
        "transformers": version("transformers"),
        "torch": torch.__version__,
        "max_seq_length": model.max_seq_length,
    },
    "long_text_tokens_before_cut": long_tokens,
    "passages": [{"text": t, "vector": v.tolist()} for t, v in zip(PASSAGES, passages, strict=True)],
    "queries": [{"text": t, "vector": v.tolist()} for t, v in zip(QUERIES, queries, strict=True)],
}
args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"written {args.out}: {len(PASSAGES)} passages, {len(QUERIES)} queries; long text: {long_tokens} tokens")
print(result["reference"])
