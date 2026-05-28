import json
import os
import sys
import time
import uuid
import base64
import random
import asyncio
import logging
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Final, Tuple, ClassVar, Optional
from datetime import datetime

import numpy as np
import gradio as gr
from openai import AsyncOpenAI
from fastrtc import AdditionalOutputs, wait_for_item, audio_to_int16
from pydantic import Field, BaseModel
from numpy.typing import NDArray
from scipy.signal import resample
from openai.types.realtime import (
    RealtimeAudioConfigParam,
    RealtimeToolsConfigParam,
    RealtimeFunctionToolParam,
    RealtimeAudioConfigOutputParam,
    RealtimeResponseCreateParamsParam,
    RealtimeSessionCreateRequestParam,
)
from websockets.exceptions import ConnectionClosedError
from openai.resources.realtime.realtime import AsyncRealtimeConnection

from reachy_mini_conversation_app.config import (
    config,
    get_default_voice_for_backend,
    get_available_voices_for_backend,
)
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.conversation_handler import ConversationHandler
from reachy_mini_conversation_app.tools.background_tool_manager import (
    ToolCallRoutine,
    ToolNotification,
    BackgroundToolManager,
)


logger = logging.getLogger(__name__)

_RESPONSE_DONE_TIMEOUT: Final[float] = 30.0
_RESPONSE_REJECTION_RETRY_DELAY: Final[float] = 0.5
_IDLE_THRESHOLD_SECONDS: Final[float] = 60.0


def _load_capture_store() -> Any:
    """Locate and import bemo-reachy's _capture_store, or return None.

    Branch #6 Phase 1: conversational memory capture lives in the
    workspace's tools/ dir (pointed to by REACHY_MINI_EXTERNAL_TOOLS_
    DIRECTORY). The conv app stays usable without it — capture hooks
    no-op when this returns None.
    """
    tools_dir = os.environ.get("REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY")
    if not tools_dir:
        return None
    tools_dir = os.path.abspath(os.path.expanduser(tools_dir))
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    try:
        import _capture_store  # type: ignore[import-not-found]
        return _capture_store
    except ImportError:
        return None


_CAPTURE_STORE = _load_capture_store()


def _load_memory_store() -> Any:
    """Locate and import bemo-reachy's _memory_store, or return None.

    Slice C: the entities/relationships graph lives in _memory_store and
    backs the recall/remember tools. The state-block builder reads from
    it to surface the entity roster (so the LLM walks in knowing which
    pets belong to which household members and never confuses a cat for
    a dog). Same path-discovery pattern as _load_capture_store.
    """
    tools_dir = os.environ.get("REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY")
    if not tools_dir:
        return None
    tools_dir = os.path.abspath(os.path.expanduser(tools_dir))
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    try:
        import _memory_store  # type: ignore[import-not-found]
        return _memory_store
    except ImportError:
        return None


_MEMORY_STORE = _load_memory_store()


def _load_speaker_state() -> Any:
    """Locate and import bemo-reachy's _speaker_state, or return None.

    Slice D: read_dossier(is_current_speaker=True) caches who Bemo is
    talking with; the state-block builder reads it to inject a
    RELATIONSHIP CONTEXT block. Same path-discovery pattern as
    _load_memory_store (the tools dir is already on sys.path by the time
    this runs).
    """
    tools_dir = os.environ.get("REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY")
    if not tools_dir:
        return None
    tools_dir = os.path.abspath(os.path.expanduser(tools_dir))
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    try:
        import _speaker_state  # type: ignore[import-not-found]
        return _speaker_state
    except ImportError:
        return None


_SPEAKER_STATE = _load_speaker_state()


# Slice D: affective self_note kinds split into two trust tiers.
#   STEERING — tone/handling metadata Bemo acts on but must NEVER speak
#     ("don't recite that you're being warm because he's your creator").
#   SHAREABLE — warm history she's free to bring up. Lumping these under
#     the never-quote guardrail made her refuse to tell her own birthday
#     in live testing, so memorable moments live here, not in steering.
# affinity_update is omitted from both — it's consolidator history, not
# live context.
_STEERING_RENDER = (
    ("interaction_cue",  "Cues"),
    ("caution",          "Handle carefully"),
)
_SHAREABLE_RENDER = (
    ("memorable_moment", "Shared history"),
)


def _render_affective_group(
    affective: dict[str, list[dict[str, Any]]],
    render_spec: tuple[tuple[str, str], ...],
    max_per_kind: int,
) -> list[str]:
    """Render one trust tier's notes as 'Label: a; b; c' lines."""
    out: list[str] = []
    for kind, label in render_spec:
        notes = affective.get(kind, [])
        if not notes:
            continue
        rendered = "; ".join(n["content"] for n in notes[:max_per_kind])
        out.append(f"{label}: {rendered}")
    return out


def _format_relationship_context(
    name: str,
    affective: dict[str, list[dict[str, Any]]],
    *,
    max_per_kind: int = 3,
) -> str | None:
    """Render Bemo's affective notes about the current speaker.

    Two tiers: a steering block she acts on but never quotes (guardrail
    layer 2; tool description + instructions.txt are layers 1 and 3), and
    a shareable 'Shared history' block she MAY bring up warmly. Returns
    None when there's nothing to show, so the caller omits it entirely.
    """
    if not affective:
        return None
    lines = [f"--- RELATIONSHIP CONTEXT: {name} ---"]

    steering = _render_affective_group(
        affective, _STEERING_RENDER, max_per_kind
    )
    if steering:
        lines.append(
            "(Steering only — act on these to shape your tone and choices. "
            "Do NOT quote, paraphrase, or read them aloud.)"
        )
        lines.extend(steering)

    shareable = _render_affective_group(
        affective, _SHAREABLE_RENDER, max_per_kind
    )
    if shareable:
        lines.append(
            "(Shared history — yours to bring up warmly when it fits. "
            "These you CAN talk about.)"
        )
        lines.extend(shareable)

    # Only the header rendered → nothing useful; omit the block.
    if len(lines) == 1:
        return None
    return "\n".join(lines)


_KIND_PLURALS = {
    "person": "people",
    "pet": "pets",
    "place": "places",
    "organization": "organizations",
    "event": "events",
    "topic": "topics",
    "thing": "things",
}


def _format_entity_roster(
    entities: list[dict[str, Any]],
    *,
    max_per_kind: int = 12,
    max_total: int = 40,
) -> str:
    """Render entities as a per-kind comma list for the state block.

    Example: "people: Jason, Amanda Hannah, Rolf Eriksen · pets: Amelia
    (dog), Beedi (dog), Cece (dog), Gingy (cat), Nut (cat)"

    Pets show their species in parens when extra.species is set, so the
    LLM never has to guess which entity is a dog vs a cat. Truncated at
    `max_per_kind` per group and `max_total` overall; the rest are
    summarized as "(+N more)".
    """
    if not entities:
        return ""
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for e in entities:
        by_kind.setdefault(e.get("kind", "thing"), []).append(e)
    parts: list[str] = []
    total = 0
    # Stable ordering: person → pet → place → organization → … alphabetical fallback
    kind_order = ["person", "pet", "place", "organization", "event", "topic", "thing"]
    seen = set(by_kind.keys())
    ordered_kinds = [k for k in kind_order if k in seen] + sorted(seen - set(kind_order))
    for kind in ordered_kinds:
        group = by_kind[kind]
        label = _KIND_PLURALS.get(kind, kind)
        rendered: list[str] = []
        for e in group[:max_per_kind]:
            name = e.get("name", "")
            extra = e.get("extra") or {}
            species = extra.get("species") if isinstance(extra, dict) else None
            if kind == "pet" and species:
                rendered.append(f"{name} ({species})")
            else:
                rendered.append(name)
            total += 1
            if total >= max_total:
                break
        if len(group) > max_per_kind:
            rendered.append(f"(+{len(group) - max_per_kind} more)")
        parts.append(f"{label}: {', '.join(rendered)}")
        if total >= max_total:
            break
    return " · ".join(parts)


def _resolve_enrich_bin() -> Optional[Path]:
    """Path to scripts/bemo-enrich derived from the external tools dir.

    Convention: tools dir is <workspace>/tools, enricher script is
    <workspace>/scripts/bemo-enrich. Returns None if either piece is
    missing.
    """
    tools_dir = os.environ.get("REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY")
    if not tools_dir:
        return None
    enrich = Path(os.path.expanduser(tools_dir)).resolve().parent / "scripts" / "bemo-enrich"
    return enrich if enrich.exists() else None


_BEMO_ENRICH_BIN = _resolve_enrich_bin()


def _resolve_scene_cooldown_seconds() -> float:
    """Read REACHY_MINI_SCENE_COOLDOWN_SECONDS or default to 180s (3 min)."""
    import os
    raw = os.environ.get("REACHY_MINI_SCENE_COOLDOWN_SECONDS")
    if raw is None:
        return 180.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning(
            "Invalid REACHY_MINI_SCENE_COOLDOWN_SECONDS=%r; using default 180s",
            raw,
        )
        return 180.0


_SCENE_COOLDOWN_SECONDS: Final[float] = _resolve_scene_cooldown_seconds()
_SCENE_STALE_SECONDS: Final[float] = 600.0  # 10 minutes
_SCENE_NOTABLE_PROMPT: Final[str] = (
    "Briefly, in one sentence: what's notable, new, or visually "
    "interesting in this scene? Mention objects, people, clothing, "
    "lighting, or anything that stands out. If it's just a typical "
    "empty room with nothing remarkable, reply exactly: nothing notable."
)


class InputTranscriptChunksByItem(BaseModel):
    """Current item_id and its accumulated deltas. Only one item at a time."""

    item_id: str | None = None
    deltas: list[str] = Field(default_factory=list)


def to_realtime_tools_config(tool_specs: list[dict[str, Any]]) -> RealtimeToolsConfigParam:
    """Convert app tool specs to the OpenAI-compatible realtime session shape."""
    realtime_tools: RealtimeToolsConfigParam = []
    for spec in tool_specs:
        tool_type = spec.get("type")
        name = spec.get("name")
        description = spec.get("description")
        parameters = spec.get("parameters", {})

        if tool_type != "function" or not isinstance(name, str):
            raise ValueError(f"Unsupported realtime tool spec: {spec!r}")

        realtime_tool = RealtimeFunctionToolParam(
            type="function",
            name=name,
            parameters=parameters,
        )
        if isinstance(description, str):
            realtime_tool["description"] = description
        realtime_tools.append(realtime_tool)
    return realtime_tools


class BaseRealtimeHandler(ConversationHandler, ABC):
    """Shared realtime stream handler for OpenAI-compatible client APIs."""

    BACKEND_PROVIDER: ClassVar[str]
    SAMPLE_RATE: ClassVar[int]
    REFRESH_CLIENT_ON_RECONNECT: ClassVar[bool]
    AUDIO_INPUT_COST_PER_1M: ClassVar[float]
    AUDIO_OUTPUT_COST_PER_1M: ClassVar[float]
    TEXT_INPUT_COST_PER_1M: ClassVar[float]
    TEXT_OUTPUT_COST_PER_1M: ClassVar[float]
    IMAGE_INPUT_COST_PER_1M: ClassVar[float]

    _REQUIRED_PROVIDER_CONFIG: ClassVar[tuple[str, ...]] = (
        "BACKEND_PROVIDER",
        "SAMPLE_RATE",
        "REFRESH_CLIENT_ON_RECONNECT",
        "AUDIO_INPUT_COST_PER_1M",
        "AUDIO_OUTPUT_COST_PER_1M",
        "TEXT_INPUT_COST_PER_1M",
        "TEXT_OUTPUT_COST_PER_1M",
        "IMAGE_INPUT_COST_PER_1M",
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Require concrete providers to declare their provider configuration."""
        super().__init_subclass__(**kwargs)
        missing = [name for name in cls._REQUIRED_PROVIDER_CONFIG if name not in cls.__dict__]
        if missing:
            raise TypeError(f"{cls.__name__} must define provider config class variable(s): {', '.join(missing)}")

    def __init__(
        self,
        deps: ToolDependencies,
        gradio_mode: bool = False,
        instance_path: Optional[str] = None,
        startup_voice: Optional[str] = None,
    ):
        """Initialize the handler."""
        sample_rate = self.SAMPLE_RATE
        super().__init__(
            expected_layout="mono",
            output_sample_rate=sample_rate,
            input_sample_rate=sample_rate,
        )

        self.deps = deps
        # Let external tools request a fresh session.update when state the
        # state-block reads changes (e.g. mood_snapshot writes).
        self.deps.refresh_session_instructions = self.refresh_session_instructions

        self.output_sample_rate = sample_rate
        self.input_sample_rate = sample_rate

        self.client: AsyncOpenAI
        self.connection: AsyncRealtimeConnection | None = None
        self.output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()

        self.last_activity_time = asyncio.get_event_loop().time()
        self.start_time = asyncio.get_event_loop().time()
        self.is_idle_tool_call = False
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path
        self._voice_override: str | None = self._normalize_startup_voice(startup_voice)
        self._realtime_connect_query: dict[str, str] = {}

        # Debouncing for partial transcripts
        self.partial_transcript_task: asyncio.Task[None] | None = None
        self.partial_debounce_delay = 0.5  # seconds
        self.input_transcript_chunks_by_item = InputTranscriptChunksByItem()

        # Internal lifecycle flags
        self._connected_event: asyncio.Event = asyncio.Event()

        # Background tool manager
        self.tool_manager = BackgroundToolManager()

        # Cost tracking
        self.cumulative_cost: float = 0.0

        # Response-in-progress guard: the Realtime API only allows one active
        # response per conversation at a time.  A dedicated worker task
        # (_response_sender_loop) dequeues and sends one request at a time
        self._pending_responses: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._response_done_event: asyncio.Event = asyncio.Event()
        self._response_done_event.set()
        self._response_started_or_rejected_event: asyncio.Event = asyncio.Event()
        self._last_response_rejected: bool = False
        self._turn_user_done_at: float | None = None
        self._turn_response_created_at: float | None = None
        self._turn_first_audio_at: float | None = None

        # Embodied vision (branch #5): caches the most recent SmolVLM2
        # observation of the room. Value is (text, monotonic_timestamp)
        # when a notable observation is current, or None when the last
        # scan returned "nothing notable" (or no scan has run yet). The
        # state block reads this; idle/startup triggers refresh it.
        self._latest_scene_observation: tuple[str, float] | None = None
        self._last_scene_scan_at: float | None = None
        self._scene_scan_lock = asyncio.Lock()

        # Realtime face-ID (feat/realtime-face-id Phase 2). Mirrors the scene
        # cache above. The recognition cache ts uses the SAME clock as
        # _face_recognition_summary's stale check (asyncio loop time), or the
        # 600 s stale gate silently breaks.
        self._latest_face_recognition: tuple[int, str, float] | None = None
        self._last_face_scan_at: float | None = None
        self._face_scan_lock = asyncio.Lock()
        self._session_recognized_ids: set[int] = set()
        self._face_unrecognized_present: bool = False
        self._user_turn_count: int = 0
        # Share the SAME set object with deps so the enroll/correct tools and
        # the recognizer mutate one gray-zone continuity set. Attached here, by
        # the set; the connect-path reset clears IN PLACE (never rebinds) so
        # this identity survives reconnects.
        self.deps.session_recognized_ids = self._session_recognized_ids

    @staticmethod
    def _sanitize_tool_result_for_model(tool_name: str, tool_result: dict[str, Any]) -> dict[str, Any]:
        """Remove bulky transport-only fields before echoing tool output back to the model."""
        if tool_name == "camera" and "b64_im" in tool_result:
            sanitized = dict(tool_result)
            sanitized.pop("b64_im", None)
            sanitized["image_attached"] = True
            return sanitized
        return tool_result

    def _normalize_startup_voice(self, voice: str | None) -> str | None:
        """Return a valid persisted startup voice for this backend, or None."""
        return self._resolve_backend_voice(voice, source="persisted startup voice")

    def _resolve_backend_voice(
        self,
        voice: str | None,
        *,
        source: str,
        fallback: str | None = None,
    ) -> str | None:
        """Return a backend-supported voice, optionally falling back when unsupported."""
        available_voices = get_available_voices_for_backend(self.BACKEND_PROVIDER)
        voice_value = (voice or "").strip()
        if not voice_value:
            return fallback

        voice_by_lowercase = {candidate.lower(): candidate for candidate in available_voices}
        normalized_voice = voice_by_lowercase.get(voice_value.lower())
        if normalized_voice is not None:
            return normalized_voice

        if voice:
            logger.warning(
                "Ignoring unsupported %s %r for backend=%r; expected one of %s",
                source,
                voice,
                self.BACKEND_PROVIDER,
                available_voices,
            )
        return fallback

    def _response_done_timeout(self) -> float:
        """Return the response completion timeout."""
        return _RESPONSE_DONE_TIMEOUT

    def _connection_closed_errors(self) -> tuple[type[BaseException], ...]:
        """Return websocket closure exceptions handled as reconnectable/ignorable."""
        return (ConnectionClosedError,)

    @abstractmethod
    def _get_session_instructions(self) -> str:
        """Return session instructions for this backend."""

    async def _read_latest_mood(self) -> dict[str, Any] | None:
        """Best-effort read of Bemo's latest mood snapshot.

        Returns None unless the external tools dir has _memory_store.py
        loaded (its `latest_mood_snapshot()` is the contract). Soft
        dependency: the conv app stays usable without external memory.
        """
        try:
            import _memory_store  # type: ignore[import-not-found]
        except ImportError:
            return None
        try:
            return await _memory_store.latest_mood_snapshot()
        except Exception:
            logger.exception("latest_mood_snapshot failed")
            return None

    def _voice_profile_summary(self) -> str | None:
        """Format the most recent user VoiceProfile as a one-line summary."""
        store = self.deps.voice_profile_store
        if store is None:
            return None
        profile = store.get_current()
        if profile is None:
            return None
        bits: list[str] = []
        emotion = profile.top_emotion()
        if emotion:
            bits.append(emotion)
        pitch = profile.top_pitch()
        if pitch:
            bits.append(f"{pitch} pitch")
        style = profile.top_vocal_style()
        if style:
            bits.append(style)
        return ", ".join(bits) if bits else None

    def _voice_trend_summary(self) -> str | None:
        """Format a short-window emotional-shift line for the state block.

        Slice F (F-live): the orthogonal complement to the instantaneous
        "User's voice right now" line. That line says how they *sound*; this
        one says how they just *changed* over the last few turns vs. earlier
        in the conversation. The shift is the signal that turns "she sounds
        sad" into "you said something that landed badly — back off" — the
        documented soften/back-off loop. Returns None on non-Inworld backends
        (empty store) or when no notable shift is detected.
        """
        store = self.deps.voice_profile_store
        if store is None:
            return None
        trend = store.valence_trend()
        if trend is None:
            return None
        # Per-label adjectives apply ONLY to the down direction — they are all
        # negative emotions. An UP shift can still carry a negative
        # `recent_label` (e.g. high-confidence "sad" easing to low-confidence
        # "sad" raises valence while the dominant label stays "sad"), so reusing
        # the label would emit "trended sadder ... match that lift." For UP we
        # always use the generic brightening word; for DOWN we fall back to
        # "flatter" when the recent label is neutral/calm (a cooling from upbeat
        # has no negative label but is still a drop).
        down_adjectives = {
            "sad": "sadder",
            "tender": "more tender",
            "angry": "more on edge",
            "disgusted": "more put-off",
            "fearful": "more anxious",
        }
        label = (trend.get("recent_label") or "").lower()
        if trend["direction"] == "down":
            adj = down_adjectives.get(label, "flatter")
            return (
                f"User's voice has trended {adj} over the last few turns "
                "(vs. earlier in this chat) — soften, slow down, and ease off "
                "the topic you just raised."
            )
        return (
            "User's voice has trended brighter over the last few turns "
            "(vs. earlier in this chat) — you can match that lift."
        )

    async def _build_state_block(self) -> str:
        """Return the CURRENT STATE block appended to session instructions.

        Always emits the mood line (even "uncharted") so the LLM is
        prompted to write a fresh mood early. Voice line is omitted on
        backends that don't populate VoiceProfileStore (HF, OpenAI,
        Gemini); only Inworld feeds it.

        Branch #6 Phase 1: also emits a directive line, a "Recently:"
        recap from the prior episode summary, and a "Phrases you've
        leaned on" line when Bemo has been overusing anything across
        sessions. The directive tells her these signals override the
        personality file's topical priming — without it, the personality
        file keeps cueing the same openings.
        """
        mood = await self._read_latest_mood()
        voice_summary = self._voice_profile_summary()

        lines = ["--- CURRENT STATE ---"]
        # Phase 1 directive: state block overrides personality-file priming.
        lines.append(
            "Treat this block as authoritative for this session's opening. "
            "If a 'Recently:' line names a topic you and the person were "
            "just discussing, pick up there — do NOT fall back to the "
            "stock topical suggestions in your personality file."
        )
        if mood is None:
            lines.append(
                "Your mood right now: uncharted "
                "(you haven't reflected on a mood yet — "
                "consider writing one with reflect(kind='mood_snapshot'))"
            )
        else:
            minutes = round(mood["age_seconds"] / 60.0)
            stale = " — possibly stale" if mood["age_seconds"] > 3 * 3600 else ""
            lines.append(
                f"Your mood right now: {mood['content']} "
                f"(written ~{minutes} min ago{stale})"
            )
        if voice_summary is not None:
            lines.append(f"User's voice right now: {voice_summary}")
        trend_summary = self._voice_trend_summary()
        if trend_summary is not None:
            lines.append(trend_summary)
        scene_summary = self._scene_observation_summary()
        if scene_summary is not None:
            lines.append(f"What you can see right now: {scene_summary}")

        # Slice C (Slice B preview): entity roster grouped by kind. Walking
        # in with "Jason's pets: Gingy (cat), Nut (cat), Amelia (dog), …"
        # eliminates the category-confusion failure mode where Bemo, asked
        # about "the dogs", picked the freshest entities in working memory
        # (Gingy and Nut, who are cats) and hallucinated their species.
        if _MEMORY_STORE is not None:
            try:
                entities = await asyncio.to_thread(
                    _MEMORY_STORE.list_all_entities_sync
                )
                roster = _format_entity_roster(entities)
                if roster:
                    lines.append(f"Known entities — {roster}")
            except Exception:
                logger.exception("entity roster build failed")

        # Slice D: RELATIONSHIP CONTEXT for the current speaker. When Bemo
        # has identified who she's talking to (via read_dossier with
        # is_current_speaker=True), inject her affective notes about them
        # so her tone is steered every turn — without her having to re-read
        # the dossier. Steering-only; the block itself says "don't quote."
        if _SPEAKER_STATE is not None and _MEMORY_STORE is not None:
            try:
                speaker = await asyncio.to_thread(
                    _SPEAKER_STATE.get_current_speaker
                )
                if speaker is not None:
                    affective = await asyncio.to_thread(
                        _MEMORY_STORE.get_affective_notes_sync, speaker["name"]
                    )
                    rel_block = _format_relationship_context(
                        speaker["name"], affective
                    )
                    if rel_block:
                        lines.append(rel_block)
            except Exception:
                logger.exception("relationship context build failed")

        # Phase 1: prior-episode recap + cross-session anti-pattern hint.
        if _CAPTURE_STORE is not None:
            try:
                recent = await asyncio.to_thread(
                    _CAPTURE_STORE.recent_episode_summaries, 1
                )
                if recent:
                    summary = (recent[0].get("summary") or "").strip()
                    vibe = (recent[0].get("vibe") or "").strip()
                    if summary:
                        vibe_tail = f" (vibe: {vibe})" if vibe else ""
                        lines.append(f"Recently: {summary}{vibe_tail}")
            except Exception:
                logger.exception("recent_episode_summaries failed")
            try:
                patterns = await asyncio.to_thread(
                    _CAPTURE_STORE.recent_patterns, min_count=5, days=7
                )
                if patterns:
                    top = patterns[:3]
                    rendered = " · ".join(
                        f'"{p["pattern"]}" ({p["count"]}×)' for p in top
                    )
                    lines.append(
                        f"Phrases you've leaned on lately: {rendered} — vary them."
                    )
            except Exception:
                logger.exception("recent_patterns failed")

        block = "\n".join(lines)
        logger.info("State block: %s", block.replace("\n", " | "))
        return block

    def _scene_observation_summary(self) -> str | None:
        """Return the cached scene observation if fresh and non-empty.

        Suppresses stale observations (>10 min) and the "nothing
        notable" case (cached as None) so the state block stays
        honest about whether Bemo currently sees anything worth
        mentioning.
        """
        cached = self._latest_scene_observation
        if cached is None:
            return None
        text, ts = cached
        if asyncio.get_event_loop().time() - ts > _SCENE_STALE_SECONDS:
            return None
        return text

    async def _run_scene_observation(self, *, force: bool = False) -> None:
        """Run a SmolVLM2 scan of the current camera frame.

        Bails out silently when:
          - no camera_worker / no vision_processor (non-vision sessions)
          - cooldown not elapsed (unless force=True)
          - no frame available
          - already scanning (concurrent guard)

        On success, caches the result and triggers a session.update
        refresh so the new observation reaches the LLM right away.
        Stores None when the model returns "nothing notable" — the
        state block then omits the scene line entirely.
        """
        if self.deps.camera_worker is None or self.deps.vision_processor is None:
            return
        if self._scene_scan_lock.locked():
            return
        now = asyncio.get_event_loop().time()
        if (
            not force
            and self._last_scene_scan_at is not None
            and now - self._last_scene_scan_at < _SCENE_COOLDOWN_SECONDS
        ):
            return

        async with self._scene_scan_lock:
            frame = self.deps.camera_worker.get_latest_frame()
            if frame is None:
                logger.debug("Scene observation skipped: no frame available")
                return
            self._last_scene_scan_at = asyncio.get_event_loop().time()
            try:
                result = await asyncio.to_thread(
                    self.deps.vision_processor.process_image,
                    frame,
                    _SCENE_NOTABLE_PROMPT,
                )
            except Exception:
                logger.exception("Scene observation failed")
                return

            text = (result or "").strip()
            normalized = text.lower().rstrip(".!?,;: ")
            if not text or normalized == "nothing notable":
                self._latest_scene_observation = None
                logger.info("Scene observation: (nothing notable)")
            else:
                self._latest_scene_observation = (
                    text,
                    asyncio.get_event_loop().time(),
                )
                logger.info("Scene observation: %s", text)

        # Refresh the session so the new state-block line is visible
        # immediately. Refresh is a no-op when no connection is live.
        try:
            await self.refresh_session_instructions()
        except Exception:
            logger.exception(
                "session refresh after scene observation failed; "
                "next natural session.update will pick it up"
            )

    async def _resolve_full_instructions(self) -> str:
        """Base instructions plus the dynamic state block."""
        base = self._get_session_instructions()
        block = await self._build_state_block()
        if not block:
            return base
        return f"{base}\n\n{block}"

    async def refresh_session_instructions(self) -> None:
        """Re-issue session.update with freshly built instructions.

        Called by tools (e.g. reflect on mood_snapshot) when state the
        state block reads has changed. No-op when no live connection.
        """
        if self.connection is None:
            return
        try:
            text = await self._resolve_full_instructions()
            voice = self.get_current_voice()
            await self.connection.session.update(
                session=RealtimeSessionCreateRequestParam(
                    type="realtime",
                    instructions=text,
                    audio=RealtimeAudioConfigParam(
                        output=RealtimeAudioConfigOutputParam(voice=voice),
                    ),
                ),
            )
            logger.info("Refreshed session instructions (state block updated)")
        except Exception:
            logger.exception("refresh_session_instructions failed")

    @abstractmethod
    def _get_session_voice(self, default: str | None = None) -> str:
        """Return the configured session voice for this backend."""

    @abstractmethod
    def _get_active_tool_specs(self) -> list[dict[str, Any]]:
        """Return active tool specs for the current session dependencies."""

    @abstractmethod
    def _get_session_config(self, tool_specs: list[dict[str, Any]]) -> RealtimeSessionCreateRequestParam:
        """Return the backend-specific realtime session config."""

    async def _wait_for_output_item(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Wait for the next output item."""
        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    def _mark_activity(self, reason: str) -> None:
        """Record non-idle conversation activity for the idle timer."""
        self.last_activity_time = asyncio.get_event_loop().time()
        logger.debug("last activity time updated to %s (%s)", self.last_activity_time, reason)

    def _maybe_update_voice_profile(self, event: Any) -> None:
        """Extract Inworld's `voiceProfile` extra from a transcription event.

        The OpenAI SDK's event classes are Pydantic models with `extra='allow'`,
        so backend-specific extras like Inworld's voice classification ride
        through and are reachable via the event's `model_extra` or as direct
        attributes. Inworld nests this data under `providerData` (the Inworld
        catch-all for non-OpenAI fields) on transcription.completed events.
        Other backends don't emit voiceProfile, so this is a no-op for them.
        """
        if self.deps.voice_profile_store is None:
            return

        def _dig(container: Any) -> Any:
            """Look for voiceProfile/voice_profile under attrs or dict keys."""
            if container is None:
                return None
            # attribute access (pydantic model fields)
            for key in ("voiceProfile", "voice_profile"):
                val = getattr(container, key, None)
                if val is not None:
                    return val
            # dict access
            if isinstance(container, dict):
                for key in ("voiceProfile", "voice_profile"):
                    if container.get(key) is not None:
                        return container[key]
            return None

        # 1. top-level (per Inworld docs the field is at event root, but in
        #    practice it's been observed under providerData — check both).
        payload: Any = _dig(event)
        if payload is None:
            extras = getattr(event, "model_extra", None)
            if isinstance(extras, dict):
                payload = _dig(extras)

        # 2. inside providerData (where Inworld actually stashes it).
        if payload is None:
            provider_data = getattr(event, "providerData", None) or getattr(event, "provider_data", None)
            if provider_data is None:
                extras = getattr(event, "model_extra", None)
                if isinstance(extras, dict):
                    provider_data = extras.get("providerData") or extras.get("provider_data")
            payload = _dig(provider_data)
            # providerData may itself nest under `stt` or `transcription` — try one layer deeper.
            if payload is None and provider_data is not None:
                for nest_key in ("stt", "transcription"):
                    inner = (
                        getattr(provider_data, nest_key, None)
                        if not isinstance(provider_data, dict)
                        else provider_data.get(nest_key)
                    )
                    payload = _dig(inner)
                    if payload is not None:
                        break

        if payload is None:
            # No voice profile on this event — normal on non-Inworld backends or
            # when Inworld couldn't classify the utterance. Stay silent.
            return
        # Pydantic may wrap nested objects as models; convert to plain dict.
        if hasattr(payload, "model_dump"):
            try:
                payload = payload.model_dump()
            except Exception:
                pass
        from reachy_mini_conversation_app.voice_profile import parse_voice_profile

        profile = parse_voice_profile(payload)
        if profile is None:
            return
        self.deps.voice_profile_store.update(profile)
        logger.info(
            "VoiceProfile: age=%s emotion=%s pitch=%s style=%s accent=%s",
            profile.top_age(),
            profile.top_emotion(),
            profile.top_pitch(),
            profile.top_vocal_style(),
            profile.top_accent(),
        )

    def copy(self) -> "BaseRealtimeHandler":
        """Create a copy of the handler."""
        return type(self)(
            self.deps,
            self.gradio_mode,
            self.instance_path,
            startup_voice=self._voice_override,
        )

    async def change_voice(self, voice: str) -> str:
        """Change only the voice and restart the session."""
        default_voice = get_default_voice_for_backend(self.BACKEND_PROVIDER)
        resolved_voice = self._resolve_backend_voice(voice, source="requested voice", fallback=default_voice)
        self._voice_override = resolved_voice
        if getattr(self, "client", None) is not None:
            try:
                await self._restart_session()
                return f"Voice changed to {resolved_voice}."
            except Exception as e:
                logger.warning("Failed to restart session for voice change: %s", e)
                return "Voice change failed. Will take effect on next connection."
        return "Voice changed. Will take effect on next connection."

    def get_current_voice(self) -> str:
        """Return the voice currently selected for this handler."""
        default_voice = get_default_voice_for_backend(self.BACKEND_PROVIDER)
        voice = self._voice_override or self._get_session_voice(default=default_voice)
        return self._resolve_backend_voice(voice, source="session voice", fallback=default_voice) or default_voice

    async def apply_personality(self, profile: str | None) -> str:
        """Apply a new personality (profile) at runtime if possible.

        - Updates the global config's selected profile for subsequent calls.
        - If a realtime connection is active, sends a session.update with the
          freshly resolved instructions so the change takes effect immediately.

        Returns a short status message for UI feedback.
        """
        try:
            # Update the in-process config value and env
            from reachy_mini_conversation_app.config import config as _config
            from reachy_mini_conversation_app.config import set_custom_profile

            set_custom_profile(profile)
            logger.info(
                "Set custom profile to %r (config=%r)", profile, getattr(_config, "REACHY_MINI_CUSTOM_PROFILE", None)
            )

            try:
                instructions = await self._resolve_full_instructions()
                voice = self.get_current_voice()
            except BaseException as e:  # catch SystemExit from prompt loader without crashing
                logger.error("Failed to resolve personality content: %s", e)
                return f"Failed to apply personality: {e}"

            # Attempt a live update first, then force a full restart to ensure it sticks
            if self.connection is not None:
                try:
                    await self.connection.session.update(
                        session=RealtimeSessionCreateRequestParam(
                            type="realtime",
                            instructions=instructions,
                            audio=RealtimeAudioConfigParam(
                                output=RealtimeAudioConfigOutputParam(
                                    voice=voice,
                                ),
                            ),
                        ),
                    )
                    logger.info("Applied personality via live update: %s", profile or "built-in default")
                except Exception as e:
                    logger.warning("Live update failed; will restart session: %s", e)

                # Force a real restart to guarantee the new instructions/voice
                try:
                    await self._restart_session()
                    return "Applied personality and restarted realtime session."
                except Exception as e:
                    logger.warning("Failed to restart session after apply: %s", e)
                    return "Applied personality. Will take effect on next connection."
            else:
                logger.info(
                    "Applied personality recorded: %s (no live connection; will apply on next session)",
                    profile or "built-in default",
                )
                return "Applied personality. Will take effect on next connection."
        except Exception as e:
            logger.error("Error applying personality '%s': %s", profile, e)
            return f"Failed to apply personality: {e}"

    async def _emit_debounced_partial(self, transcript: str, item_id: str, sequence_counter: int) -> None:
        """Emit partial transcript after debounce delay."""
        try:
            await asyncio.sleep(self.partial_debounce_delay)

            input_transcript = self.input_transcript_chunks_by_item
            if input_transcript.item_id == item_id and len(input_transcript.deltas) - 1 == sequence_counter:
                await self.output_queue.put(AdditionalOutputs({"role": "user_partial", "content": transcript}))
                logger.debug(f"Debounced partial emitted: {transcript}")
        except asyncio.CancelledError:
            logger.debug("Debounced partial cancelled")
            raise

    def _record_partial_transcript_delta(
        self,
        input_transcript: InputTranscriptChunksByItem,
        item_id: str,
        delta: str,
    ) -> None:
        """Record a suffix delta for a partial transcript."""
        if input_transcript.item_id != item_id:
            input_transcript.item_id = item_id
            input_transcript.deltas = [delta]
        else:
            input_transcript.deltas.append(delta)

    def _compute_response_cost(self, usage: Any) -> float:
        """Compute response cost using this backend's pricing."""
        inp = getattr(usage, "input_token_details", None)
        out = getattr(usage, "output_token_details", None)
        cost = 0.0
        if inp:
            cost += (getattr(inp, "audio_tokens", 0) or 0) * self.AUDIO_INPUT_COST_PER_1M / 1e6
            cost += (getattr(inp, "text_tokens", 0) or 0) * self.TEXT_INPUT_COST_PER_1M / 1e6
            cost += (getattr(inp, "image_tokens", 0) or 0) * self.IMAGE_INPUT_COST_PER_1M / 1e6
        if out:
            cost += (getattr(out, "audio_tokens", 0) or 0) * self.AUDIO_OUTPUT_COST_PER_1M / 1e6
            cost += (getattr(out, "text_tokens", 0) or 0) * self.TEXT_OUTPUT_COST_PER_1M / 1e6
        return cost

    async def _prepare_startup_credentials(self) -> None:
        """Let providers collect any startup credentials they need."""

    def _persist_credentials_if_needed(self) -> None:
        """Let providers persist credentials after a successful session update."""

    async def start_up(self) -> None:
        """Start the handler with minimal retries on unexpected websocket closure."""
        await self._prepare_startup_credentials()
        self.client = await self._build_realtime_client()

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                await self._run_realtime_session()
                # Normal exit from the session, stop retrying
                return
            except self._connection_closed_errors() as e:
                # Abrupt close (e.g., "no close frame received or sent") → retry
                logger.warning("Realtime websocket closed unexpectedly (attempt %d/%d): %s", attempt, max_attempts, e)
                if attempt < max_attempts:
                    if self.REFRESH_CLIENT_ON_RECONNECT:
                        self.client = await self._build_realtime_client()
                    # exponential backoff with jitter
                    base_delay = 2 ** (attempt - 1)  # 1s, 2s, 4s, 8s, etc.
                    jitter = random.uniform(0, 0.5)
                    delay = base_delay + jitter
                    logger.info("Retrying in %.1f seconds...", delay)
                    await asyncio.sleep(delay)
                    continue
                raise
            finally:
                # never keep a stale reference
                self.connection = None
                try:
                    self._connected_event.clear()
                except Exception:
                    pass

    async def _restart_session(self) -> None:
        """Force-close the current session and start a fresh one in background.

        Does not block the caller while the new session is establishing.
        """
        try:
            if self.connection is not None:
                try:
                    await self.connection.close()
                except Exception:
                    pass
                finally:
                    self.connection = None

            # Ensure we have a client (start_up must have run once)
            if getattr(self, "client", None) is None:
                logger.warning("Cannot restart: realtime client not initialized yet.")
                return

            # Fire-and-forget new session and wait briefly for connection
            try:
                self._connected_event.clear()
            except Exception:
                pass
            if self.REFRESH_CLIENT_ON_RECONNECT:
                self.client = await self._build_realtime_client()
            asyncio.create_task(self._run_realtime_session(), name="realtime-session-restart")
            try:
                await asyncio.wait_for(self._connected_event.wait(), timeout=5.0)
                logger.info("Realtime session restarted and connected.")
            except asyncio.TimeoutError:
                logger.warning("Realtime session restart timed out; continuing in background.")
        except Exception as e:
            logger.warning("_restart_session failed: %s", e)

    async def _safe_response_create(self, **kwargs: Any) -> None:
        """Enqueue a response.create() kwargs for the sender worker _response_sender_loop().

        This method never blocks the caller.
        """
        await self._pending_responses.put(kwargs)

    async def _response_sender_loop(self) -> None:
        """Dedicated worker that sends ``response.create()`` calls serially.

        This logic was designed to comply with the response.create() docstring specification for event ordering:
        https://github.com/openai/openai-python/blob/3e0c05b84a2056870abf3bd6a5e7849020209cc3/src/openai/resources/realtime/realtime.py#L649C1-L651C30

        For each queued request the worker:
        1. Waits until no response is active (_response_done_event).
        2. Sends response.create().
        3. Waits until the receiver observes response.created or a rejection.
        4. Waits for the response cycle to complete (response.done).
        5. If the server rejected with active_response, retries from step 1.
        """
        while self.connection:
            try:
                kwargs = await self._pending_responses.get()
            except asyncio.CancelledError:
                return

            sent = False
            max_retries = 5
            attempts = 0
            while not sent and self.connection and attempts < max_retries:
                try:
                    await asyncio.wait_for(
                        self._response_done_event.wait(),
                        timeout=self._response_done_timeout(),
                    )
                except asyncio.TimeoutError:
                    logger.debug("Timed out waiting for previous response to finish; forcing ahead")
                    self._response_done_event.set()

                if not self.connection:
                    break

                self._last_response_rejected = False
                self._response_started_or_rejected_event.clear()
                try:
                    await self.connection.response.create(**kwargs)
                except Exception as e:
                    logger.debug("_response_sender_loop: send failed: %s", e)
                    self._response_done_event.set()
                    break

                try:
                    await asyncio.wait_for(
                        self._response_started_or_rejected_event.wait(),
                        timeout=self._response_done_timeout(),
                    )
                except asyncio.TimeoutError:
                    logger.debug("Timed out waiting for response.created or response rejection")

                # Check if the receiver loop observed an asynchronous rejection.
                if self._last_response_rejected:
                    attempts += 1
                    if attempts >= max_retries:
                        logger.debug("response.create rejected %d times; giving up", attempts)
                        break
                    logger.debug("response.create was rejected; retrying (%d/%d)", attempts, max_retries)
                    await asyncio.sleep(_RESPONSE_REJECTION_RETRY_DELAY)
                    continue

                try:
                    await asyncio.wait_for(
                        self._response_done_event.wait(),
                        timeout=self._response_done_timeout(),
                    )
                except asyncio.TimeoutError:
                    logger.debug("Timed out waiting for response.done; assuming response completed")
                    self._response_done_event.set()
                    break

                sent = True

    async def _handle_tool_result(self, bg_tool: ToolNotification) -> None:
        """Process the result of a tool call."""
        if bg_tool.error is not None:
            logger.error("Tool '%s' (id=%s) failed with error: %s", bg_tool.tool_name, bg_tool.id, bg_tool.error)
            tool_result = {"error": bg_tool.error}
            tool_result_for_model = tool_result
        elif bg_tool.result is not None:
            tool_result = bg_tool.result
            tool_result_for_model = (
                self._sanitize_tool_result_for_model(bg_tool.tool_name, tool_result)
                if isinstance(tool_result, dict)
                else tool_result
            )
            logger.info(
                "Tool '%s' (id=%s) executed successfully.",
                bg_tool.tool_name,
                bg_tool.id,
            )
            logger.debug("Tool '%s' model-visible result: %s", bg_tool.tool_name, tool_result_for_model)
        else:
            logger.warning("Tool '%s' (id=%s) returned no result and no error", bg_tool.tool_name, bg_tool.id)
            tool_result = {"error": "No result returned from tool execution"}
            tool_result_for_model = tool_result

        # Connection may have closed while tool was running
        if not self.connection:
            logger.warning(
                "Connection closed during tool '%s' (id=%s) execution; cannot send result back",
                bg_tool.tool_name,
                bg_tool.id,
            )
            return

        try:
            self._mark_activity("tool_result_ready")
            if isinstance(bg_tool.id, str):
                await self.connection.conversation.item.create(
                    item={
                        "type": "function_call_output",
                        "call_id": bg_tool.id,
                        "output": json.dumps(tool_result_for_model),
                    },
                )

            await self.output_queue.put(
                AdditionalOutputs(
                    {
                        "role": "assistant",
                        "content": json.dumps(tool_result_for_model),
                        # Gradio UI metadata.status accept only "pending" and "done". Do not accept bg.tool.status values.
                        "metadata": {
                            "title": f"🛠️ Used tool {bg_tool.tool_name}",
                            "status": "done",
                        },
                    },
                ),
            )

            if bg_tool.tool_name == "camera" and "b64_im" in tool_result:
                # use raw base64, don't json.dumps (which adds quotes)
                b64_im = tool_result["b64_im"]
                if not isinstance(b64_im, str):
                    logger.warning("Unexpected type for b64_im: %s", type(b64_im))
                    b64_im = str(b64_im)
                image_width = tool_result.get("image_width")
                image_height = tool_result.get("image_height")
                jpeg_bytes_value = tool_result.get("jpeg_bytes")
                jpeg_bytes = jpeg_bytes_value if isinstance(jpeg_bytes_value, int) else (len(b64_im) * 3) // 4
                await self.connection.conversation.item.create(
                    item={
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{b64_im}",
                            },
                        ],
                    },
                )
                if isinstance(image_width, int) and isinstance(image_height, int):
                    logger.info(
                        "Added camera image to conversation frame=%sx%s jpeg_bytes=%s",
                        image_width,
                        image_height,
                        jpeg_bytes,
                    )
                else:
                    logger.info(
                        "Added camera image to conversation jpeg_bytes=%s",
                        jpeg_bytes,
                    )

                if self.deps.camera_worker is not None:
                    np_img = self.deps.camera_worker.get_latest_frame()
                    if np_img is not None:
                        # Camera frames are BGR; reverse channels without requiring OpenCV in core installs.
                        rgb_frame = np_img[:, :, ::-1].copy() if np_img.ndim == 3 and np_img.shape[-1] == 3 else np_img
                    else:
                        rgb_frame = None
                    img = gr.Image(value=rgb_frame)

                    await self.output_queue.put(
                        AdditionalOutputs(
                            {
                                "role": "assistant",
                                "content": img,
                            },
                        ),
                    )

            # If this tool call was triggered by an idle signal, don't make the robot speak.
            # For other tool calls, let the robot reply out loud.
            if not bg_tool.is_idle_tool_call:
                await self._safe_response_create(
                    response=RealtimeResponseCreateParamsParam(
                        instructions="Use the tool result just returned and answer concisely in speech.",
                    ),
                )

        except self._connection_closed_errors():
            logger.warning("Connection closed while sending tool result")
            self.connection = None
            self._response_done_event.set()

    async def _run_realtime_session(self) -> None:
        """Establish and manage a single realtime session."""
        tool_specs = self._get_active_tool_specs()
        logger.info(
            "Tools to be used in conversation: %s",
            [tool["name"] for tool in tool_specs],
        )
        connect_kwargs: dict[str, Any] = {}
        if config.MODEL_NAME:
            connect_kwargs["model"] = config.MODEL_NAME
        if self._realtime_connect_query:
            connect_kwargs["extra_query"] = self._realtime_connect_query
        async with self.client.realtime.connect(**connect_kwargs) as conn:
            try:
                session_config = self._get_session_config(tool_specs)
                # Append dynamic state block (mood + user voice) to the
                # backend's resolved instructions before sending.
                state_block = await self._build_state_block()
                if state_block:
                    base = getattr(session_config, "instructions", None) or ""
                    try:
                        session_config.instructions = f"{base}\n\n{state_block}"
                    except Exception:
                        logger.warning(
                            "Could not append state block to session_config; "
                            "instructions field is non-writable on this backend"
                        )
                await conn.session.update(session=session_config)
                logger.info(
                    "Realtime session initialized with profile=%r voice=%r",
                    getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None),
                    self.get_current_voice(),
                )
                self._persist_credentials_if_needed()
                # Branch #6 Phase 1: open a capture episode. No-op when
                # _capture_store isn't loaded (workspace tools/ dir not
                # set or import failed). Stored as instance attribute so
                # transcript event handlers below can append turns to it.
                self._capture_episode_id = None
                # Face-ID (Phase 2): reset per-session continuity state. The
                # handler is reused across reconnects, so clear the recognized
                # set IN PLACE (preserving the deps-shared object identity) and
                # reset the enrollment-cue flag + user-turn counter.
                self._session_recognized_ids.clear()
                self._face_unrecognized_present = False
                self._user_turn_count = 0
                if _CAPTURE_STORE is not None:
                    try:
                        self._capture_episode_id = await _CAPTURE_STORE.open_episode([])
                    except Exception:
                        logger.exception("open_episode failed; capture disabled this session")
                # Re-assert the live episode id onto deps so the enroll/correct
                # tools log sightings + merge participants against this episode.
                self.deps.capture_episode_id = self._capture_episode_id
            except Exception:
                logger.exception("Realtime session.update failed; aborting startup")
                raise

            logger.info("Realtime session updated successfully")

            # Fire-and-forget a scene observation shortly after the
            # session opens so Bemo's first response can reference what
            # she sees. Delay keeps SmolVLM2 inference off the critical
            # startup path; the eventual refresh_session_instructions()
            # injects the result when ready.
            async def _startup_scene_scan() -> None:
                try:
                    await asyncio.sleep(2.0)
                    await self._run_scene_observation()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Startup scene scan failed")

            asyncio.create_task(_startup_scene_scan())

            # Reset the partial-transcript accumulator for each new session
            self.input_transcript_chunks_by_item = InputTranscriptChunksByItem()

            # Track function-call names by call_id from response.output_item.added
            # events, so .function_call_arguments.done dispatch can fall back to
            # this dict when the .done event omits the `name` field. (Inworld
            # observed dropping name in the .done event; OpenAI/Gemini set both.)
            self._function_call_names_by_call_id: dict[str, str] = {}

            # Manage events received from the realtime server.
            self.connection = conn
            try:
                self._connected_event.set()
            except Exception:
                pass

            response_sender_task: asyncio.Task[None] | None = None
            try:
                # Start the background tool manager
                self.tool_manager.start_up(tool_callbacks=[self._handle_tool_result])

                # Start the response sender worker
                response_sender_task = asyncio.create_task(self._response_sender_loop(), name="response-sender")

                async for event in self.connection:
                    logger.debug("Realtime event: %s", event.type)

                    if event.type == "input_audio_buffer.speech_started":
                        self._mark_activity("user_speech_started")
                        self._turn_user_done_at = None
                        self._turn_response_created_at = None
                        self._turn_first_audio_at = None
                        if hasattr(self, "_clear_queue") and callable(self._clear_queue):
                            self._clear_queue()
                        if self.deps.head_wobbler is not None:
                            self.deps.head_wobbler.reset()
                        self.deps.movement_manager.set_listening(True)
                        logger.debug("User speech started")

                    if event.type == "input_audio_buffer.speech_stopped":
                        self._mark_activity("user_speech_stopped")
                        self.deps.movement_manager.set_listening(False)
                        logger.debug("User speech stopped - server will auto-commit with VAD")

                    if event.type == "response.output_audio.done":
                        if self.deps.head_wobbler is not None:
                            self.deps.head_wobbler.request_reset_after_current_audio()
                        logger.debug("response completed")

                    if event.type == "response.created":
                        self._mark_activity("response_created")
                        self._response_done_event.clear()
                        self._response_started_or_rejected_event.set()
                        if self._turn_user_done_at is not None and self._turn_response_created_at is None:
                            self._turn_response_created_at = time.perf_counter()
                            delta_ms = (self._turn_response_created_at - self._turn_user_done_at) * 1000
                            logger.info("Turn latency: response.created %.0f ms after user transcript", delta_ms)
                        logger.debug("Response created (active)")

                    if event.type == "response.done":
                        # Doesn't mean the audio is done playing
                        self._response_done_event.set()
                        self._response_started_or_rejected_event.set()
                        self.is_idle_tool_call = False
                        logger.debug("Response done")

                        response = getattr(event, "response", None)
                        usage = getattr(response, "usage", None) if response else None
                        if usage:
                            cost = self._compute_response_cost(usage)
                            self.cumulative_cost += cost
                            logger.debug("Cost: $%.4f | Cumulative: $%.4f", cost, self.cumulative_cost)
                        else:
                            logger.warning("No usage data available for cost tracking")

                    if event.type == "conversation.item.input_audio_transcription.delta":
                        self._mark_activity("user_transcription_delta")
                        logger.debug(f"User partial transcript: {event.delta}")

                        item_id = event.item_id
                        delta = event.delta or ""

                        input_transcript = self.input_transcript_chunks_by_item
                        self._record_partial_transcript_delta(input_transcript, item_id, delta)

                        current_partial = "".join(input_transcript.deltas)
                        sequence_counter = len(input_transcript.deltas) - 1

                        # Cancel previous debounce task if it exists
                        if self.partial_transcript_task and not self.partial_transcript_task.done():
                            self.partial_transcript_task.cancel()
                            try:
                                await self.partial_transcript_task
                            except asyncio.CancelledError:
                                pass

                        # Start new debounce timer with the last delta
                        self.partial_transcript_task = asyncio.create_task(
                            self._emit_debounced_partial(current_partial, item_id, sequence_counter)
                        )

                    # Handle completed transcription (user finished speaking)
                    if event.type == "conversation.item.input_audio_transcription.completed":
                        self._mark_activity("user_transcription_completed")
                        raw_transcript = event.transcript or ""
                        transcript = raw_transcript.strip()
                        logger.debug("User transcript: %s", raw_transcript)
                        self.deps.movement_manager.set_listening(False)

                        # Inworld voice-profile extension: classification ride-alongs on
                        # the transcription completion event. Other backends don't emit
                        # this, so the lookup harmlessly returns None.
                        self._maybe_update_voice_profile(event)

                        # Cancel any pending partial emission
                        if self.partial_transcript_task and not self.partial_transcript_task.done():
                            self.partial_transcript_task.cancel()
                            try:
                                await self.partial_transcript_task
                            except asyncio.CancelledError:
                                pass

                        if not transcript:
                            logger.debug("Ignoring empty user transcript")
                            continue

                        self._turn_user_done_at = time.perf_counter()
                        self._turn_response_created_at = None
                        self._turn_first_audio_at = None

                        await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))

                        # Branch #6 Phase 1: capture user turn synchronously
                        # to disk. SIGKILL-safe (no buffering).
                        if _CAPTURE_STORE is not None and self._capture_episode_id is not None:
                            try:
                                await _CAPTURE_STORE.append_turn(
                                    self._capture_episode_id,
                                    "user",
                                    transcript,
                                    getattr(event, "item_id", None),
                                )
                            except Exception:
                                logger.exception("append_turn (user) failed")

                    # Handle assistant transcription
                    if event.type == "response.output_audio_transcript.done":
                        self._mark_activity("assistant_transcript_done")
                        logger.info("Assistant transcript: %s", event.transcript)
                        await self.output_queue.put(
                            AdditionalOutputs({"role": "assistant", "content": event.transcript})
                        )

                        # Branch #6 Phase 1: capture assistant turn.
                        if _CAPTURE_STORE is not None and self._capture_episode_id is not None:
                            try:
                                await _CAPTURE_STORE.append_turn(
                                    self._capture_episode_id,
                                    "assistant",
                                    event.transcript or "",
                                    getattr(event, "item_id", None),
                                )
                            except Exception:
                                logger.exception("append_turn (assistant) failed")

                    # Handle audio delta
                    if event.type == "response.output_audio.delta":
                        decoded_pcm_bytes = base64.b64decode(event.delta)
                        decoded_pcm = np.frombuffer(decoded_pcm_bytes, dtype=np.int16).reshape(1, -1)
                        if self.gradio_mode and self.deps.head_wobbler is not None:
                            self.deps.head_wobbler.feed_pcm(decoded_pcm, self.output_sample_rate)
                        self._mark_activity("assistant_audio_delta")
                        if self._turn_user_done_at is not None and self._turn_first_audio_at is None:
                            self._turn_first_audio_at = time.perf_counter()
                            delta_ms = (self._turn_first_audio_at - self._turn_user_done_at) * 1000
                            logger.info("Turn latency: first audio delta %.0f ms after user transcript", delta_ms)
                        await self.output_queue.put(
                            (
                                self.output_sample_rate,
                                decoded_pcm,
                            ),
                        )
                    # ---- tool-calling plumbing ----
                    # Capture function names as they're announced (Inworld emits
                    # the name here but later omits it from .arguments.done; OpenAI
                    # populates both, so this is a no-op there).
                    if event.type == "response.output_item.added":
                        item = getattr(event, "item", None)
                        if item is not None and getattr(item, "type", None) == "function_call":
                            item_call_id = getattr(item, "call_id", None)
                            item_name = getattr(item, "name", None)
                            if isinstance(item_call_id, str) and isinstance(item_name, str):
                                self._function_call_names_by_call_id[item_call_id] = item_name

                    if event.type == "response.function_call_arguments.done":
                        self._mark_activity("tool_call_received")
                        tool_name = getattr(event, "name", None)
                        args_json_str = getattr(event, "arguments", None)
                        call_id: str = str(getattr(event, "call_id", uuid.uuid4()))

                        # Fall back to the name captured from output_item.added
                        # if this event didn't carry it (Inworld behavior).
                        if not isinstance(tool_name, str) or not tool_name:
                            tool_name = self._function_call_names_by_call_id.get(call_id)

                        logger.info(
                            "Tool call received — tool_name=%r, call_id=%s, is_idle=%s, args=%s",
                            tool_name,
                            call_id,
                            self.is_idle_tool_call,
                            args_json_str,
                        )

                        if not isinstance(tool_name, str) or not isinstance(args_json_str, str):
                            logger.error(
                                "Invalid tool call: tool_name=%s (type=%s), args=%s (type=%s), call_id=%s",
                                tool_name,
                                type(tool_name).__name__,
                                args_json_str,
                                type(args_json_str).__name__,
                                call_id,
                            )
                            continue

                        bg_tool = await self.tool_manager.start_tool(
                            call_id=call_id,
                            tool_call_routine=ToolCallRoutine(
                                tool_name=tool_name,
                                args_json_str=args_json_str,
                                deps=self.deps,
                            ),
                            is_idle_tool_call=self.is_idle_tool_call,
                        )

                        await self.output_queue.put(
                            AdditionalOutputs(
                                {
                                    "role": "assistant",
                                    "content": f"🛠️ Used tool {tool_name} with args {args_json_str}. The tool is now running. Tool ID: {bg_tool.tool_id}",
                                },
                            ),
                        )
                        logger.info(
                            "Started background tool: %s (id=%s, call_id=%s)", tool_name, bg_tool.tool_id, call_id
                        )

                    # server error
                    if event.type == "error":
                        err = getattr(event, "error", None)
                        msg = getattr(err, "message", str(err) if err else "unknown error")
                        code = getattr(err, "code", "") or getattr(err, "type", "")

                        if code == "conversation_already_has_active_response":
                            # response.create was rejected.  The sender worker
                            # is waiting on _response_done_event; when the active
                            # response finishes it will wake up and see this flag.
                            self._last_response_rejected = True
                            self._response_started_or_rejected_event.set()
                            logger.debug("response.create rejected; worker will retry after active response finishes")
                        else:
                            self._response_started_or_rejected_event.set()
                            logger.error("Realtime error [%s]: %s (raw=%s)", code, msg, err)

                        if code == "input_audio_buffer_commit_empty":
                            self.deps.movement_manager.set_listening(False)

                        # Only show user-facing errors, not internal state errors.
                        if code not in ("input_audio_buffer_commit_empty", "conversation_already_has_active_response"):
                            await self.output_queue.put(
                                AdditionalOutputs({"role": "assistant", "content": f"[error] {msg}"})
                            )
            finally:
                # Stop the response sender worker.
                if response_sender_task is not None:
                    response_sender_task.cancel()
                    try:
                        await response_sender_task
                    except asyncio.CancelledError:
                        pass

                # Stop background tool manager tasks (listener + cleanup) in all paths.
                await self.tool_manager.shutdown()

                # Branch #6 Phase 1: close the capture episode and
                # spawn the enricher detached so SIGKILL/teardown of
                # the conv-app doesn't kill it. Uses close_episode_sync
                # rather than `await close_episode` because by the time
                # the asyncio finally runs, the loop is being canceled
                # and any awaitable racing with teardown loses (verified
                # 2026-05-27 — ended_at stayed NULL despite the await).
                # SQLite writes are <5ms, so the sync block is cheap.
                if _CAPTURE_STORE is not None and self._capture_episode_id is not None:
                    eid = self._capture_episode_id
                    self._capture_episode_id = None
                    try:
                        _CAPTURE_STORE.close_episode_sync(eid)
                    except Exception:
                        logger.exception("close_episode_sync failed for episode %d", eid)
                    if _BEMO_ENRICH_BIN is not None:
                        # Redirect stdio to a per-episode log instead of
                        # DEVNULL. Prior silent failures (the spawned
                        # child ran but wrote nothing to the DB) were
                        # un-diagnosable with DEVNULL. -v gives DEBUG
                        # output so any future spawn-side errors are
                        # captured. The conv-app closes its dup of the
                        # log fd; the child keeps fd 1/2 pointed at it.
                        try:
                            log_dir = Path("/tmp/bemo-reachy-logs")
                            log_dir.mkdir(parents=True, exist_ok=True)
                            log_path = log_dir / f"enrich-{eid}.log"
                            with open(log_path, "ab") as log_fp:
                                subprocess.Popen(
                                    [sys.executable, str(_BEMO_ENRICH_BIN),
                                     "--episode", str(eid), "-v"],
                                    stdout=log_fp,
                                    stderr=subprocess.STDOUT,
                                    start_new_session=True,
                                    close_fds=True,
                                )
                            logger.info(
                                "Spawned bemo-enrich for episode %d (log: %s)",
                                eid, log_path,
                            )
                        except Exception:
                            logger.exception(
                                "Failed to spawn bemo-enrich for episode %d "
                                "(cron --unprocessed will catch it later)",
                                eid,
                            )

    # Microphone receive
    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive audio frame from the microphone and send it to the realtime server.

        Handles both mono and stereo audio formats, converting to the expected
        mono format for the realtime API. Resamples if the input sample rate differs
        from the expected rate.

        Args:
            frame: A tuple containing (sample_rate, audio_data).

        """
        if not self.connection:
            return

        input_sample_rate, audio_frame = frame

        # Reshape if needed
        if audio_frame.ndim == 2:
            # Scipy channels last convention
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            # Multiple channels -> Mono channel
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]

        # Resample if needed
        if self.input_sample_rate != input_sample_rate:
            audio_frame = resample(audio_frame, int(len(audio_frame) * self.input_sample_rate / input_sample_rate))

        # Cast if needed
        audio_frame = audio_to_int16(audio_frame)

        # Send to the realtime input buffer (guard against races during reconnect).
        try:
            audio_message = base64.b64encode(audio_frame.tobytes()).decode("utf-8")
            await self.connection.input_audio_buffer.append(audio=audio_message)
        except Exception as e:
            logger.debug("Dropping audio frame: connection not ready (%s)", e)
            return

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit audio frame to be played by the speaker."""
        # Sends output queued by the realtime event handler to the stream.
        # This is called periodically by the fastrtc Stream

        # Handle idle
        idle_duration = asyncio.get_event_loop().time() - self.last_activity_time
        if (
            idle_duration > _IDLE_THRESHOLD_SECONDS
            and self._response_done_event.is_set()
            and self.deps.movement_manager.is_idle()
        ):
            try:
                await self.send_idle_signal(idle_duration)
            except Exception as e:
                logger.warning("Idle signal skipped (connection closed?): %s", e)
                return None

            self.last_activity_time = asyncio.get_event_loop().time()  # avoid repeated resets

        return await self._wait_for_output_item()

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        # Unblock the response sender worker so it can exit
        self._response_done_event.set()

        # Stop background tool manager tasks (listener + cleanup)
        await self.tool_manager.shutdown()

        # Cancel any pending debounce task
        if self.partial_transcript_task and not self.partial_transcript_task.done():
            self.partial_transcript_task.cancel()
            try:
                await self.partial_transcript_task
            except asyncio.CancelledError:
                pass

        if self.connection:
            try:
                await self.connection.close()
            except self._connection_closed_errors() as e:
                logger.debug(f"Connection already closed during shutdown: {e}")
            except Exception as e:
                logger.debug(f"connection.close() ignored: {e}")
            finally:
                self.connection = None

        # Clear any remaining items in the output queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def format_timestamp(self) -> str:
        """Format current timestamp with date, time, and elapsed seconds."""
        loop_time = asyncio.get_event_loop().time()  # monotonic
        elapsed_seconds = loop_time - self.start_time
        dt = datetime.now()  # wall-clock
        return f"[{dt.strftime('%Y-%m-%d %H:%M:%S')} | +{elapsed_seconds:.1f}s]"

    async def get_available_voices(self) -> list[str]:
        """Return available voices for this backend."""
        return get_available_voices_for_backend(self.BACKEND_PROVIDER)

    @abstractmethod
    async def _build_realtime_client(self) -> AsyncOpenAI:
        """Build the realtime SDK client for this backend."""

    async def send_idle_signal(self, idle_duration: float) -> None:
        """Send an idle signal to the realtime server.

        Injects a synthetic user message marking the silence and lets the
        LLM pick freely (speech, tool, both, or nothing). The session-level
        instructions in the active profile own the actual behavior policy
        — see Bemo's "IDLE TIME" paragraph.

        Sets ``is_idle_tool_call`` so any tool calls made in response to
        the idle signal don't each trigger a follow-up ``response.create``
        in ``_handle_tool_result``. Without that guard, every tool result
        prompts a new "use the tool result and answer concisely in speech"
        response — which cascades into runaway chatter when the LLM has
        nothing meaningful to add but is told to speak anyway. With the
        guard, all idle output (speech + tools) is contained in the single
        response triggered by this signal.
        """
        logger.debug("Sending idle signal")
        # Embodied vision: try a scene scan before sending the idle
        # signal so Bemo's idle reply can include what she sees. The
        # cooldown gate inside the helper keeps this from running on
        # every idle cycle.
        await self._run_scene_observation()
        self.is_idle_tool_call = True
        timestamp_msg = (
            f"[Idle time update: {self.format_timestamp()} - "
            f"{idle_duration:.1f}s since last activity] "
            "The room's gone quiet. Take a moment — volunteer a thought, "
            "recall something you'd want to bring up, do a small action, "
            "or stay still."
        )
        if not self.connection:
            logger.debug("No connection, cannot send idle signal")
            return
        await self.connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": timestamp_msg}],
            },
        )
        await self._safe_response_create()
