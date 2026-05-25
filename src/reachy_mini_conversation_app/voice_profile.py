"""Speaker voice-profile store.

Captures Inworld's per-utterance voice classification (age, emotion, pitch,
vocal style, accent) from realtime transcription events and exposes it to the
rest of the conv app — tools, custom profiles, sidecar modules — via either:

  - `deps.voice_profile_store` on `ToolDependencies` (preferred for tools)
  - `get_voice_profile_store()` module singleton (preferred for standalone code
    running in the same Python process)

The store is thread-safe and keeps a bounded history so future modules can
reason about trends (rising pitch, sustained emotion, etc).

See https://docs.inworld.ai/stt/voice-profiles for the source feature; we
mirror its category set and confidence semantics 1:1.

Usage example:

    from reachy_mini_conversation_app.voice_profile import get_voice_profile_store

    store = get_voice_profile_store()
    profile = store.get_current()
    if profile and profile.top_emotion() == "sad":
        # adapt response, play a tender emotion, route to a human, etc.
        ...
"""
from __future__ import annotations
import time
import threading
from typing import Any
from dataclasses import field, dataclass


__all__ = [
    "ClassLabel",
    "VoiceProfile",
    "VoiceProfileStore",
    "get_voice_profile_store",
    "parse_voice_profile",
]


@dataclass(frozen=True)
class ClassLabel:
    """A single (label, confidence) classification entry.

    `confidence` is in [0.0, 1.0]; arrays of these are returned by Inworld
    sorted by descending confidence.
    """

    label: str
    confidence: float


@dataclass(frozen=True)
class VoiceProfile:
    """One snapshot of Inworld voice classification for a single utterance.

    Each category is an array sorted by descending confidence. Arrays may be
    empty if Inworld couldn't classify that category for the utterance — always
    check before accessing `[0]`. The `top_*()` helpers return None safely.
    """

    age: list[ClassLabel] = field(default_factory=list)
    emotion: list[ClassLabel] = field(default_factory=list)
    pitch: list[ClassLabel] = field(default_factory=list)
    vocal_style: list[ClassLabel] = field(default_factory=list)
    accent: list[ClassLabel] = field(default_factory=list)
    # The full original Inworld payload, in case future modules need a field
    # we haven't lifted into a dataclass member yet.
    raw: dict[str, Any] = field(default_factory=dict)
    # `time.monotonic()` when this profile was received. Use deltas, not wall
    # time, for "how long since the speaker last sounded sad" style queries.
    received_monotonic: float = 0.0

    def top_age(self) -> str | None:
        """Return the most-confident age label, or None if none was returned."""
        return self.age[0].label if self.age else None

    def top_emotion(self) -> str | None:
        """Return the most-confident emotion label, or None."""
        return self.emotion[0].label if self.emotion else None

    def top_pitch(self) -> str | None:
        """Return the most-confident pitch label, or None."""
        return self.pitch[0].label if self.pitch else None

    def top_vocal_style(self) -> str | None:
        """Return the most-confident vocal-style label, or None."""
        return self.vocal_style[0].label if self.vocal_style else None

    def top_accent(self) -> str | None:
        """Return the most-confident accent (BCP-47 locale), or None."""
        return self.accent[0].label if self.accent else None


def _parse_label_array(items: Any) -> list[ClassLabel]:
    """Coerce Inworld's `[{label, confidence}, ...]` arrays into ClassLabels.

    Handles both camelCase (`confidence`) and snake_case (`confidence`) keys.
    Robust to missing fields and odd types — returns whatever it can.
    """
    if not isinstance(items, list):
        return []
    out: list[ClassLabel] = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        label = entry.get("label")
        conf = entry.get("confidence")
        if not isinstance(label, str):
            continue
        try:
            conf_f = float(conf) if conf is not None else 0.0
        except (TypeError, ValueError):
            conf_f = 0.0
        out.append(ClassLabel(label=label, confidence=conf_f))
    return out


def parse_voice_profile(payload: Any) -> VoiceProfile | None:
    """Build a `VoiceProfile` from a raw Inworld `voiceProfile` payload.

    Returns None if `payload` isn't a dict or has no usable fields.
    Accepts both camelCase (`vocalStyle`) and snake_case (`vocal_style`).
    """
    if not isinstance(payload, dict):
        return None
    vocal_style_raw = payload.get("vocalStyle") or payload.get("vocal_style") or []
    return VoiceProfile(
        age=_parse_label_array(payload.get("age")),
        emotion=_parse_label_array(payload.get("emotion")),
        pitch=_parse_label_array(payload.get("pitch")),
        vocal_style=_parse_label_array(vocal_style_raw),
        accent=_parse_label_array(payload.get("accent")),
        raw=payload,
        received_monotonic=time.monotonic(),
    )


class VoiceProfileStore:
    """Thread-safe store for the latest speaker voice profile + bounded history.

    Producer side: the Inworld realtime handler calls `update()` whenever a
    `voiceProfile` payload arrives on a transcription event.

    Consumer side: any tool/module calls `get_current()` (for the most recent)
    or `get_history()` (for trend analysis).
    """

    DEFAULT_MAX_HISTORY = 50

    def __init__(self, max_history: int = DEFAULT_MAX_HISTORY) -> None:
        """Initialize an empty store with capped history length."""
        self._lock = threading.Lock()
        self._current: VoiceProfile | None = None
        self._history: list[VoiceProfile] = []
        self._max_history = max_history

    def update(self, profile: VoiceProfile) -> None:
        """Record a new voice profile snapshot. O(1) amortized."""
        with self._lock:
            self._current = profile
            self._history.append(profile)
            if len(self._history) > self._max_history:
                # Drop oldest. O(n) here but n is small (~50).
                self._history.pop(0)

    def get_current(self) -> VoiceProfile | None:
        """Return the most recently observed profile, or None if none yet."""
        with self._lock:
            return self._current

    def get_history(self) -> list[VoiceProfile]:
        """Return a snapshot of recent profiles, oldest first. Safe to iterate."""
        with self._lock:
            return list(self._history)

    def clear(self) -> None:
        """Forget everything. Useful at session boundaries."""
        with self._lock:
            self._current = None
            self._history.clear()


_global_store: VoiceProfileStore = VoiceProfileStore()


def get_voice_profile_store() -> VoiceProfileStore:
    """Return the process-wide `VoiceProfileStore` singleton.

    Use this from any code that isn't on the tool-call path (where
    `deps.voice_profile_store` is preferred). The singleton always exists; it
    just stays empty if voice-profile isn't enabled on the active backend.
    """
    return _global_store
