"""Tests for the F-live state-block trend line.

Exercises `BaseRealtimeHandler._voice_trend_summary` end-to-end (real store →
`valence_trend` → phrasing) with a duck-typed `self` — the method only reads
`self.deps.voice_profile_store`, so no full realtime handler is constructed.
Profiles use the real `time.monotonic()` clock (the method calls
`valence_trend()` with no injectable `now`), spaced within the lookback window.
"""

import time
from types import SimpleNamespace

from reachy_mini_conversation_app.base_realtime import BaseRealtimeHandler
from reachy_mini_conversation_app.voice_profile import (
    ClassLabel,
    VoiceProfile,
    VoiceProfileStore,
)


def _profile(label: str, confidence: float, received: float) -> VoiceProfile:
    return VoiceProfile(
        emotion=[ClassLabel(label=label, confidence=confidence)],
        received_monotonic=received,
    )


def _summary_for(profiles: list[VoiceProfile]) -> str | None:
    store = VoiceProfileStore()
    for p in profiles:
        store.update(p)
    fake_self = SimpleNamespace(deps=SimpleNamespace(voice_profile_store=store))
    return BaseRealtimeHandler._voice_trend_summary(fake_self)


def test_summary_none_without_store():
    """No store → no trend line."""
    fake_self = SimpleNamespace(deps=SimpleNamespace(voice_profile_store=None))
    assert BaseRealtimeHandler._voice_trend_summary(fake_self) is None


def test_summary_none_without_shift():
    """A steady register → no trend line."""
    base = time.monotonic()
    assert _summary_for([_profile("calm", 0.8, base) for _ in range(6)]) is None


def test_summary_downward_to_sad():
    """A real drop into sadness reads 'sadder' and tells Bemo to ease off."""
    base = time.monotonic()
    prior = [_profile("happy", 1.0, base) for _ in range(4)]
    recent = [_profile("sad", 1.0, base) for _ in range(3)]
    line = _summary_for(prior + recent)
    assert line is not None
    assert "sadder" in line
    assert "ease off" in line


def test_summary_upward_with_negative_label_says_brighter_not_sadder():
    """Codex P2 regression: an upward shift must not reuse a negative label."""
    # High-confidence sad easing to low-confidence sad raises valence while the
    # dominant recent_label stays "sad"; the line must read "brighter ... match
    # that lift", never "sadder ... match that lift".
    base = time.monotonic()
    prior = [_profile("sad", 0.95, base) for _ in range(4)]  # strongly sad
    recent = [_profile("sad", 0.3, base) for _ in range(3)]  # barely sad
    line = _summary_for(prior + recent)
    assert line is not None
    assert "brighter" in line
    assert "match that lift" in line
    assert "sadder" not in line


def test_summary_downward_with_positive_label_says_flatter_not_brighter():
    """Mirror case: a down shift labelled "happy" must not say "brighter"."""
    # Happy easing to less-happy is a downward shift while recent_label stays
    # "happy"; the line must read "flatter ... ease off", not "brighter".
    base = time.monotonic()
    prior = [_profile("happy", 1.0, base) for _ in range(4)]
    recent = [_profile("happy", 0.3, base) for _ in range(3)]
    line = _summary_for(prior + recent)
    assert line is not None
    assert "flatter" in line
    assert "ease off" in line
    assert "brighter" not in line
