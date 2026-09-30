"""
Persistent, semantically-retrieved memory -- the same mean-pooled-GloVe
cosine-similarity technique from week5-production-ai/07_agent_memory.py,
now actually persisted to disk (data/memory.json) so facts survive
between separate runs of the assistant, not just within one session.
"""

import json
import os
import numpy as np
import gensim.downloader as api

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_MEMORY_FILE = os.path.join(_DATA_DIR, "memory.json")

_wv = None  # lazy-loaded -- avoid the ~66MB GloVe download/load unless memory is actually used


def _get_wv():
    global _wv
    if _wv is None:
        _wv = api.load("glove-wiki-gigaword-50")
    return _wv


def _embed(text):
    wv = _get_wv()
    words = [w.strip(".,!?").lower() for w in text.split()]
    vectors = [wv[w] for w in words if w in wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(wv.vector_size)


def _cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class Memory:
    def __init__(self):
        os.makedirs(_DATA_DIR, exist_ok=True)
        self.facts = self._load()

    def _load(self):
        if os.path.exists(_MEMORY_FILE):
            with open(_MEMORY_FILE, encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save(self):
        with open(_MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(self.facts, f, indent=2)

    def remember(self, fact):
        self.facts.append(fact)
        self._save()

    def forget_all(self):
        self.facts = []
        self._save()

    def retrieve_relevant(self, query, k=3, min_score=0.75):
        """Only returns memories that clear a similarity floor -- an empty/irrelevant
        context is better than injecting noise the model might latch onto."""
        if not self.facts:
            return []
        query_emb = _embed(query)
        scored = [(_cosine_similarity(query_emb, _embed(fact)), fact) for fact in self.facts]
        scored.sort(key=lambda x: -x[0])
        return [(score, fact) for score, fact in scored[:k] if score >= min_score]
