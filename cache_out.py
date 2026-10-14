"""
Cache Out — semantic cache for inference gateways.

Same question twice should not mean two GPU passes. Cache Out sits in
front of the model and returns cached completions for semantically
identical prompts:

- exact match: normalized text key (casefold, whitespace-collapsed)
- fuzzy match: token-set Jaccard similarity >= threshold (default 0.98 —
  near-identical only; creative prompts must not cross-hit)
- TTL: entries expire (default 3600s)
- bounded: LRU eviction at max_entries
- honest stats: hits, misses, hit_rate — the number that justifies the cache

Public API:
    SemanticCache(threshold=0.98, ttl=3600, max_entries=1000)
    .get(prompt) -> str | None
    .put(prompt, completion)
    .stats() -> {hits, misses, hit_rate, entries, evictions}

Deliberately dependency-free: token-set Jaccard instead of embeddings.
Embedding similarity is a swap-in (same interface), not a requirement —
and it keeps the cache itself faster than the model it shields.
"""
from __future__ import annotations

import hashlib
import re
import time
from collections import OrderedDict

__version__ = "0.1.0"

_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    """Casefold + collapse whitespace — the exact-match key."""
    return " ".join(text.casefold().split())


def tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.casefold()))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class SemanticCache:
    def __init__(self, threshold: float = 0.98, ttl: float = 3600.0,
                 max_entries: int = 1000):
        if not 0.5 < threshold <= 1.0:
            raise ValueError("threshold must be in (0.5, 1.0]")
        self.threshold = threshold
        self.ttl = ttl
        self.max_entries = max_entries
        self._exact: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._fuzzy: list[tuple[float, set[str], str]] = []  # (ts, toks, completion)
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    # ---------------------------------------------------------------- core

    def get(self, prompt: str) -> str | None:
        now = time.time()
        key = normalize(prompt)

        hit = self._exact.get(key)
        if hit and now - hit[0] <= self.ttl:
            self._exact.move_to_end(key)
            self.hits += 1
            return hit[1]
        if hit:  # expired
            del self._exact[key]

        ptoks = tokens(prompt)
        for ts, toks, _key, completion in self._fuzzy:
            if now - ts > self.ttl:
                continue
            if jaccard(ptoks, toks) >= self.threshold:
                self.hits += 1
                return completion
        self.misses += 1
        return None

    def put(self, prompt: str, completion: str) -> None:
        now = time.time()
        key = normalize(prompt)
        self._exact[key] = (now, completion)
        self._exact.move_to_end(key)
        self._fuzzy.append((now, tokens(prompt), key, completion))
        self._evict_if_needed(now)

    def _evict_if_needed(self, now: float) -> None:
        # drop expired fuzzy entries first
        self._fuzzy = [(ts, t, k, c) for (ts, t, k, c) in self._fuzzy
                       if now - ts <= self.ttl]
        overflow = len(self._exact) - self.max_entries
        for _ in range(max(0, overflow)):
            old_key, _ = self._exact.popitem(last=False)  # LRU
            self._fuzzy = [(ts, t, k, c) for (ts, t, k, c) in self._fuzzy
                           if k != old_key]  # evicted entry dies in BOTH stores
            self.evictions += 1
        while len(self._fuzzy) > self.max_entries:
            self._fuzzy.pop(0)  # oldest
            self.evictions += 1

    # ---------------------------------------------------------------- stats

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": self.hits / total if total else 0.0,
                "entries": len(self._exact), "evictions": self.evictions}
