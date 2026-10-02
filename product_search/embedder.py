"""
Text -> vector, using a small open embedding model that runs locally.

The same model must embed BOTH the products (done once, offline, on the GPU
because 890k products is a lot of text) and the shopper's query (done per
request, on the CPU, one short string -- a few milliseconds). Vectors are
L2-normalised, so cosine similarity is just a dot product.

bge-small-en-v1.5 is trained so that a *query* and a *passage* that answers it
land near each other. It wants a fixed instruction prefix on queries (but not
on documents) -- forgetting it measurably hurts quality.
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name=DEFAULT_MODEL, device=None, max_seq_length=128, query_prefix=QUERY_PREFIX):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.query_prefix = query_prefix
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.max_seq_length = max_seq_length      # product text is short; longer only costs time
        if self.device == "cuda":
            self.model.half()                           # fp16: ~2x faster, no measurable quality loss for retrieval
        self.dim = self.model.get_sentence_embedding_dimension()

    def _encode(self, texts, batch_size, show_progress):
        vecs = self.model.encode(
            list(texts), batch_size=batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=show_progress,
        )
        return vecs.astype(np.float32, copy=False)

    def encode_documents(self, texts, batch_size=256, show_progress=False):
        return self._encode(texts, batch_size, show_progress)

    def encode_queries(self, queries, batch_size=64, show_progress=False):
        return self._encode([self.query_prefix + q for q in queries], batch_size, show_progress)
