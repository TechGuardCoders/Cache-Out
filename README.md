# Cache Out

**Semantic cache for inference gateways — same question twice should not mean two GPU passes.**

Sits in front of the model; returns cached completions for semantically identical prompts.

```python
from cache_out import SemanticCache

cache = SemanticCache(threshold=0.98, ttl=3600, max_entries=1000)

cache.put("explain the KV cache budget for batch 4", "4.29 GB")
cache.get("explain the KV cache budget for batch 4.")   # '4.29 GB' — fuzzy hit
cache.get("write a poem about the ocean")                # None — miss, go to model
cache.stats()   # {'hits': 1, 'misses': 1, 'hit_rate': 0.5, ...}
```

## Design decisions

- **Token-set Jaccard, not embeddings** — dependency-free, and the cache stays faster than the model it shields. Embedding similarity is a swap-in with the same interface, not a requirement.
- **Default threshold 0.98** — near-identical only. A creative prompt must never cross-hit a cached answer; a wrong cache hit is worse than a cache miss.
- **LRU with dual-store consistency** — an evicted entry dies in both the exact and fuzzy stores (tests caught the bug where it survived in fuzzy).
- **Honest stats** — `hit_rate` is the number that justifies the cache's existence.

## Part of the TechGuardCoders portfolio

- **Triage** decides *which* tier; Cache Out decides *whether* — check the cache before routing
- **Ball Knowledge** is the natural host: cache in front of the BackendChain
- **statsheet** proves the hit rate is real, not anecdotal

## Tests

```bash
python -m pytest tests/ -q   # 17 tests
```

Tests caught two real bugs during the build: evicted entries surviving in the fuzzy store, and eviction double-counting.
