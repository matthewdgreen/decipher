"""validate_model self-heals a stale OpenRouter model cache.

A model listed on OpenRouter AFTER the <=24h disk pricing cache was written
(e.g. a fresh release) must not be rejected as unknown. validate_model forces
exactly ONE live refresh on a cache-miss and re-checks before reporting
not-found. A genuinely invalid id still fails cleanly, with no fetch loop.
"""
from __future__ import annotations

import pytest

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


# ---------------------------------------------------------------------------
# Transient rate-limit retry (K3/OpenRouter 429 incident, 2026-07-17)
# ---------------------------------------------------------------------------

OPENROUTER_429 = (
    "Error code: 429 - {'error': {'message': 'Provider returned error', "
    "'code': 429, 'metadata': {'raw': 'moonshotai/kimi-k3 is temporarily "
    "rate-limited upstream. Please retry shortly', 'provider_name': "
    "'Moonshot AI', 'retry_after_seconds': 1, "
    '\'headers\': {"Retry-After": "1"}}}}'
)
QUOTA_429 = (
    "Error code: 429 - {'error': {'message': 'You exceeded your current "
    "quota', 'type': 'insufficient_quota', 'code': 'insufficient_quota'}}"
)


def _no_sleep(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(mp.time, "sleep", lambda s: sleeps.append(s))
    return sleeps


def test_retryable_detection():
    assert mp.is_retryable_rate_limit_error(
        mp.ModelProviderError(OPENROUTER_429)) is True
    assert mp.is_retryable_rate_limit_error(
        mp.ModelProviderError(QUOTA_429)) is False  # billing: never retry
    assert mp.is_retryable_rate_limit_error(
        mp.ModelProviderError("Rate limit exceeded, slow down")) is True
    assert mp.is_retryable_rate_limit_error(
        mp.ModelProviderError("connection reset by peer")) is False


def test_parse_retry_after_seconds():
    assert mp.parse_retry_after_seconds(
        mp.ModelProviderError(OPENROUTER_429)) == 1.0
    assert mp.parse_retry_after_seconds(
        mp.ModelProviderError("Retry-After: 20")) == 20.0
    assert mp.parse_retry_after_seconds(
        mp.ModelProviderError("some other error")) is None
    # Clamped to 60s so a hostile/huge header cannot stall a run.
    assert mp.parse_retry_after_seconds(
        mp.ModelProviderError("retry_after_seconds': 9999")) == 60.0


def test_retry_succeeds_after_transient_429(monkeypatch):
    sleeps = _no_sleep(monkeypatch)
    calls = {"n": 0}
    retries: list[tuple[int, float]] = []

    def send():
        calls["n"] += 1
        if calls["n"] <= 2:
            raise mp.ModelProviderError(OPENROUTER_429)
        return "ok"

    out = mp.call_with_rate_limit_retry(
        send, on_retry=lambda a, d, e: retries.append((a, d)))
    assert out == "ok"
    assert calls["n"] == 3
    # Schedule (2, 5) wins over the 1s Retry-After (max of the two).
    assert sleeps == [2.0, 5.0]
    assert retries == [(1, 2.0), (2, 5.0)]


def test_retry_honors_longer_retry_after(monkeypatch):
    sleeps = _no_sleep(monkeypatch)
    calls = {"n": 0}

    def send():
        calls["n"] += 1
        if calls["n"] == 1:
            raise mp.ModelProviderError("429 rate limit. Retry-After: 20")
        return "ok"

    assert mp.call_with_rate_limit_retry(send) == "ok"
    assert sleeps == [20.0]  # provider's wait exceeds the 2s schedule slot


def test_persistent_429_reraises_after_bounded_retries(monkeypatch):
    sleeps = _no_sleep(monkeypatch)
    calls = {"n": 0}

    def send():
        calls["n"] += 1
        raise mp.ModelProviderError(OPENROUTER_429)

    with pytest.raises(mp.ModelProviderError):
        mp.call_with_rate_limit_retry(send)
    assert calls["n"] == 4          # initial + 3 bounded retries, then give up
    assert sleeps == [2.0, 5.0, 10.0]


def test_insufficient_quota_fails_fast(monkeypatch):
    sleeps = _no_sleep(monkeypatch)
    calls = {"n": 0}

    def send():
        calls["n"] += 1
        raise mp.ModelProviderError(QUOTA_429)

    with pytest.raises(mp.ModelProviderError):
        mp.call_with_rate_limit_retry(send)
    assert calls["n"] == 1 and sleeps == []   # no futile billing retries
