"""Tests for Slice F (F-live) valence + emotional-trend detection.

Pure-math coverage of `VoiceProfile.valence()` and
`VoiceProfileStore.valence_trend()` — no Inworld backend required. Profiles
are constructed directly with controlled `received_monotonic` timestamps and
the trend detector is driven with an injected `now` so the lookback window is
deterministic.
"""

from reachy_mini_conversation_app.voice_profile import (
    ClassLabel,
    VoiceProfile,
    VoiceProfileStore,
)


def _profile(label: str | None, confidence: float, received: float) -> VoiceProfile:
    """Build a VoiceProfile carrying a single emotion label at a given time."""
    emotion = [ClassLabel(label=label, confidence=confidence)] if label else []
    return VoiceProfile(emotion=emotion, received_monotonic=received)


def _store_with(profiles: list[VoiceProfile]) -> VoiceProfileStore:
    """Return a store pre-populated with `profiles` (oldest first)."""
    store = VoiceProfileStore()
    for p in profiles:
        store.update(p)
    return store


# --- VoiceProfile.valence() -------------------------------------------------


def test_valence_confidence_weighted():
    """Valence is the label anchor scaled by the label's confidence."""
    assert _profile("happy", 1.0, 0.0).valence() == 1.0
    assert _profile("sad", 0.5, 0.0).valence() == -0.5
    assert _profile("angry", 1.0, 0.0).valence() == -0.6


def test_valence_none_when_no_emotion():
    """A profile with no emotion array yields no valence."""
    assert VoiceProfile().valence() is None


def test_valence_none_for_unmapped_label():
    """Unmapped labels (e.g. Inworld's "unclear") are None, not neutral 0."""
    assert _profile("unclear", 0.9, 0.0).valence() is None


def test_valence_case_insensitive():
    """Label lookup is case-insensitive."""
    assert _profile("HAPPY", 1.0, 0.0).valence() == 1.0


# --- VoiceProfileStore.valence_trend() --------------------------------------


def test_trend_none_below_min_samples():
    """Fewer than min_samples mappable readings → no trend."""
    now = 1000.0
    store = _store_with([_profile("sad", 1.0, now - i) for i in range(4)])
    assert store.valence_trend(now=now) is None


def test_trend_none_when_stable():
    """A steady emotional register produces no shift."""
    now = 1000.0
    store = _store_with([_profile("calm", 0.8, now - (50 - i)) for i in range(6)])
    assert store.valence_trend(now=now) is None


def test_trend_downward_shift():
    """Upbeat earlier, sad recently → a 'down' shift labelled sad."""
    now = 1000.0
    prior = [_profile("happy", 1.0, now - (60 - i)) for i in range(4)]  # older
    recent = [_profile("sad", 1.0, now - (10 - i)) for i in range(3)]  # newer
    store = _store_with(prior + recent)
    trend = store.valence_trend(now=now)
    assert trend is not None
    assert trend["direction"] == "down"
    assert trend["recent_label"] == "sad"
    assert trend["prior_mean"] == 1.0
    assert trend["recent_mean"] == -1.0


def test_trend_upward_shift():
    """Sad earlier, happy recently → an 'up' shift labelled happy."""
    now = 1000.0
    prior = [_profile("sad", 1.0, now - (60 - i)) for i in range(4)]
    recent = [_profile("happy", 1.0, now - (10 - i)) for i in range(3)]
    store = _store_with(prior + recent)
    trend = store.valence_trend(now=now)
    assert trend is not None
    assert trend["direction"] == "up"
    assert trend["recent_label"] == "happy"


def test_trend_single_blip_does_not_fire():
    """One mild sad reading among calm ones stays under the delta gate."""
    now = 1000.0
    prior = [_profile("calm", 1.0, now - (60 - i)) for i in range(4)]
    recent = [
        _profile("calm", 1.0, now - 12),
        _profile("calm", 1.0, now - 8),
        _profile("sad", 0.4, now - 4),
    ]
    store = _store_with(prior + recent)
    assert store.valence_trend(now=now) is None


def test_trend_excludes_stale_readings_outside_lookback():
    """Readings older than lookback_seconds can't serve as the baseline."""
    now = 10000.0
    stale = [_profile("happy", 1.0, now - 5000 - i) for i in range(4)]  # > 600s
    recent = [_profile("sad", 1.0, now - (10 - i)) for i in range(3)]
    store = _store_with(stale + recent)
    assert store.valence_trend(now=now, lookback_seconds=600.0) is None


def test_trend_skips_unmappable_readings():
    """Unmappable (valence None) readings are dropped before windowing."""
    now = 1000.0
    profiles = [
        _profile("happy", 1.0, now - 60),
        _profile("unclear", 0.9, now - 55),  # dropped
        _profile("happy", 1.0, now - 50),
        _profile("happy", 1.0, now - 45),
        _profile("happy", 1.0, now - 40),
        _profile("sad", 1.0, now - 12),
        _profile("unclear", 0.9, now - 9),  # dropped
        _profile("sad", 1.0, now - 6),
        _profile("sad", 1.0, now - 3),
    ]
    store = _store_with(profiles)
    trend = store.valence_trend(now=now)
    assert trend is not None
    assert trend["direction"] == "down"
    assert trend["samples"] == 7  # 9 minus the 2 unmappable
