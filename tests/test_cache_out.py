"""Cache Out tests — hits, misses, thresholds, TTL, eviction, honesty."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from cache_out import SemanticCache, normalize, tokens, jaccard


# ---------------------------------------------------------------- helpers

def test_normalize_collapses_case_and_space():
    assert normalize("  Hello   WORLD ") == "hello world"


def test_tokens_strips_punctuation():
    assert tokens("Hello, world!") == {"hello", "world"}


def test_jaccard_identical_is_one():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_jaccard_disjoint_is_zero():
    assert jaccard({"a"}, {"b"}) == 0.0


# ---------------------------------------------------------------- exact

def test_exact_hit():
    c = SemanticCache()
    c.put("What is 2+2?", "4")
    assert c.get("What is 2+2?") == "4"


def test_exact_hit_after_normalization():
    c = SemanticCache()
    c.put("What is 2+2?", "4")
    assert c.get("what   is 2+2?") == "4"


def test_miss_returns_none():
    c = SemanticCache()
    assert c.get("unrelated") is None


# ---------------------------------------------------------------- fuzzy

def test_fuzzy_hit_near_identical():
    c = SemanticCache()
    c.put("explain the KV cache budget for batch 4", "4.29 GB")
    assert c.get("explain the KV cache budget for batch 4.") == "4.29 GB"


def test_fuzzy_does_not_cross_hit_different_prompts():
    c = SemanticCache()
    c.put("explain the KV cache budget", "4.29 GB")
    assert c.get("write a poem about the ocean") is None


def test_low_threshold_crosses_hits_more():
    c = SemanticCache(threshold=0.6)
    c.put("the quick brown fox jumps", "A")
    assert c.get("the quick brown fox runs") == "A"


def test_default_threshold_is_conservative():
    c = SemanticCache()
    c.put("the quick brown fox jumps", "A")
    # 'jumps' vs 'runs' -> jaccard 4/6 = 0.67 < 0.98 -> miss
    assert c.get("the quick brown fox runs") is None


# ---------------------------------------------------------------- ttl

def test_ttl_expiry(monkeypatch):
    c = SemanticCache(ttl=10)
    t = {"now": 1000.0}
    monkeypatch.setattr("cache_out.time.time", lambda: t["now"])
    c.put("q", "a")
    t["now"] = 1005.0
    assert c.get("q") == "a"
    t["now"] = 1011.0
    assert c.get("q") is None


# ---------------------------------------------------------------- eviction

def test_lru_eviction_at_max_entries():
    c = SemanticCache(max_entries=2)
    c.put("a", "1")
    c.put("b", "2")
    c.put("c", "3")  # evicts 'a'
    assert c.get("a") is None
    assert c.get("c") == "3"
    assert c.evictions == 1


def test_recently_used_survives():
    c = SemanticCache(max_entries=2)
    c.put("a", "1")
    c.put("b", "2")
    c.get("a")          # touch a -> b is now LRU
    c.put("c", "3")
    assert c.get("a") == "1"
    assert c.get("b") is None


# ---------------------------------------------------------------- stats

def test_stats_hit_rate():
    c = SemanticCache()
    c.put("q", "a")
    c.get("q")
    c.get("q")
    c.get("other")
    s = c.stats()
    assert s["hits"] == 2
    assert s["misses"] == 1
    assert s["hit_rate"] == pytest.approx(2 / 3)


def test_stats_empty_cache():
    assert SemanticCache().stats()["hit_rate"] == 0.0


def test_invalid_threshold_raises():
    with pytest.raises(ValueError):
        SemanticCache(threshold=0.2)
