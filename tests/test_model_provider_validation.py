"""validate_model self-heals a stale OpenRouter model cache.

A model listed on OpenRouter AFTER the <=24h disk pricing cache was written
(e.g. a fresh release) must not be rejected as unknown. validate_model forces
exactly ONE live refresh on a cache-miss and re-checks before reporting
not-found. A genuinely invalid id still fails cleanly, with no fetch loop.
"""
from __future__ import annotations

import agent.model_provider as mp


def _isolate(monkeypatch):
    """Pin the module state so validate_model touches neither disk nor network
    except through the monkeypatched fetch. A non-None pricing memo short-
    circuits _get_openrouter_live_pricing (no disk load / lazy fetch)."""
    monkeypatch.setattr(mp, "_OPENROUTER_PRICING_LIVE", {}, raising=False)


def test_stale_cache_refreshes_and_accepts_new_model(monkeypatch):
    _isolate(monkeypatch)
    # Cache predates kimi-k3.
    monkeypatch.setattr(mp, "_OPENROUTER_ALL_MODEL_IDS", {"moonshotai/kimi-k2"})
    calls = {"n": 0}

    def fake_fetch(*, timeout=10.0, cache_path=None, write_cache=True):
        calls["n"] += 1
        # A live refresh discovers the freshly-listed model.
        mp._OPENROUTER_ALL_MODEL_IDS = {"moonshotai/kimi-k2", "moonshotai/kimi-k3"}
        return {}

    monkeypatch.setattr(mp, "fetch_openrouter_pricing", fake_fetch)

    ok, hint = mp.validate_model("openrouter", "moonshotai/kimi-k3")
    assert ok is True
    assert hint == ""
    assert calls["n"] == 1  # exactly one self-heal refresh


def test_genuinely_invalid_model_fails_after_one_refresh(monkeypatch):
    _isolate(monkeypatch)
    monkeypatch.setattr(mp, "_OPENROUTER_ALL_MODEL_IDS", {"moonshotai/kimi-k2"})
    calls = {"n": 0}

    def fake_fetch(*, timeout=10.0, cache_path=None, write_cache=True):
        calls["n"] += 1
        # Even fresh, the bogus id does not exist.
        mp._OPENROUTER_ALL_MODEL_IDS = {"moonshotai/kimi-k2", "moonshotai/kimi-k3"}
        return {}

    monkeypatch.setattr(mp, "fetch_openrouter_pricing", fake_fetch)

    ok, hint = mp.validate_model("openrouter", "moonshotai/kimi-k9-nope")
    assert ok is False
    assert "was not found on OpenRouter" in hint
    assert "kimi" in hint  # suggestions drawn from the FRESH id set
    assert calls["n"] == 1  # bounded — a single extra fetch, no loop


def test_refresh_network_failure_falls_through_to_hint(monkeypatch):
    _isolate(monkeypatch)
    monkeypatch.setattr(mp, "_OPENROUTER_ALL_MODEL_IDS", {"moonshotai/kimi-k2"})
    calls = {"n": 0}

    def boom(*, timeout=10.0, cache_path=None, write_cache=True):
        calls["n"] += 1
        raise OSError("network down")

    monkeypatch.setattr(mp, "fetch_openrouter_pricing", boom)

    ok, hint = mp.validate_model("openrouter", "moonshotai/kimi-k3")
    assert ok is False
    assert "was not found on OpenRouter" in hint
    assert calls["n"] == 1  # attempted once, then gave up gracefully


def test_no_double_refresh_when_first_fetch_was_fresh(monkeypatch):
    _isolate(monkeypatch)
    # No id set yet -> the is-None branch fetches (fetched_fresh=True); a miss
    # must NOT trigger a second refresh.
    monkeypatch.setattr(mp, "_OPENROUTER_ALL_MODEL_IDS", None)
    calls = {"n": 0}

    def fake_fetch(*, timeout=10.0, cache_path=None, write_cache=True):
        calls["n"] += 1
        mp._OPENROUTER_ALL_MODEL_IDS = {"moonshotai/kimi-k2"}
        return {}

    monkeypatch.setattr(mp, "fetch_openrouter_pricing", fake_fetch)

    ok, hint = mp.validate_model("openrouter", "moonshotai/kimi-k3")
    assert ok is False
    assert calls["n"] == 1  # the initial populate only; no redundant self-heal


def test_cached_model_present_does_not_refresh(monkeypatch):
    _isolate(monkeypatch)
    monkeypatch.setattr(
        mp, "_OPENROUTER_ALL_MODEL_IDS", {"moonshotai/kimi-k3"}
    )
    calls = {"n": 0}

    def fake_fetch(*, timeout=10.0, cache_path=None, write_cache=True):
        calls["n"] += 1
        return {}

    monkeypatch.setattr(mp, "fetch_openrouter_pricing", fake_fetch)

    ok, hint = mp.validate_model("openrouter", "moonshotai/kimi-k3")
    assert ok is True and hint == ""
    assert calls["n"] == 0  # already known — no network at all
