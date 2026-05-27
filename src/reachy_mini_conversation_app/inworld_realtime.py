import os
import time
import logging
from typing import Any
from functools import cached_property

import httpx
from openai import AsyncOpenAI
from openai._types import omit
from openai.resources.realtime.realtime import (
    AsyncRealtime,
    AsyncRealtimeConnectionManager,
)

from reachy_mini_conversation_app.config import INWORLD_BACKEND, INWORLD_DEFAULT_LLM, config
from reachy_mini_conversation_app.prompts import get_session_voice, get_session_instructions
from reachy_mini_conversation_app.base_realtime import BaseRealtimeHandler, to_realtime_tools_config
from reachy_mini_conversation_app.tools.core_tools import get_active_tool_specs


logger = logging.getLogger(__name__)

__all__ = ["InworldRealtimeHandler"]


INWORLD_WS_BASE = "wss://api.inworld.ai/api/v1"
INWORLD_HTTP_BASE = "https://api.inworld.ai/api/v1"
# INWORLD_DEFAULT_LLM is imported from config — single source of truth.
INWORLD_STT_MODEL = "assemblyai/u3-rt-pro"
INWORLD_TTS_MODEL = "inworld-tts-2"
# 16 kHz PCM for parity with the Reachy Mini audio pipeline. Inworld's server
# defaults to 24 kHz but accepts explicit rate via `audio.{in,out}.format.rate`
# in session.update.
INWORLD_SAMPLE_RATE = 16000


class _InworldConnectionManager(AsyncRealtimeConnectionManager):
    """Inworld realtime endpoint is `/realtime/session` not `/realtime`."""

    def _prepare_url(self) -> httpx.URL:
        url = super()._prepare_url()
        return url.copy_with(raw_path=url.raw_path + b"/session")


class _InworldAsyncRealtime(AsyncRealtime):
    """`AsyncRealtime` that returns our patched connection manager."""

    def connect(
        self,
        *,
        call_id: Any = omit,
        model: Any = omit,
        extra_query: Any = None,
        extra_headers: Any = None,
        websocket_connection_options: Any = None,
    ) -> AsyncRealtimeConnectionManager:
        """Return an `_InworldConnectionManager` with Basic auth injected.

        The SDK builds the WS auth header from `client.auth_headers` (which is
        always `Bearer <api_key>`); `default_headers` passed to AsyncOpenAI()
        only flows to HTTP REST, not WebSocket. To send `Authorization: Basic
        <key>`, we put it in `extra_headers`, which the SDK merges *after*
        auth_headers and therefore overrides it.
        """
        merged_headers = dict(extra_headers) if extra_headers else {}
        basic_auth = getattr(self._client, "_inworld_basic_auth", None)
        if basic_auth:
            merged_headers["Authorization"] = basic_auth
        ws_opts = websocket_connection_options if websocket_connection_options is not None else {}
        return _InworldConnectionManager(
            client=self._client,
            extra_query=extra_query if extra_query is not None else {},
            extra_headers=merged_headers,
            websocket_connection_options=ws_opts,  # type: ignore[arg-type]
            call_id=call_id,
            model=model,
        )


class _InworldAsyncOpenAI(AsyncOpenAI):
    """`AsyncOpenAI` subclass that exposes Inworld's URL/auth via `.realtime`."""

    def __init__(self, *, inworld_basic_auth: str, **kwargs: Any) -> None:
        """Store the `Basic <base64-key>` header so `.realtime.connect()` can inject it."""
        super().__init__(**kwargs)
        self._inworld_basic_auth = inworld_basic_auth

    @cached_property
    def realtime(self) -> _InworldAsyncRealtime:
        """Return an `_InworldAsyncRealtime`."""
        return _InworldAsyncRealtime(self)


class InworldRealtimeHandler(BaseRealtimeHandler):
    """Realtime handler for Inworld AI's combined STT + LLM + TTS endpoint.

    Inworld speaks the OpenAI Realtime wire protocol verbatim, with these
    deltas: (1) URL path is `/realtime/session` not `/realtime`, (2) auth is
    `Basic <base64-key>` not `Bearer`, (3) the session.update payload carries
    Inworld-specific `audio.output.model` (TTS model) and a `model` field at
    session level for LLM routing.
    """

    BACKEND_PROVIDER = INWORLD_BACKEND
    SAMPLE_RATE = INWORLD_SAMPLE_RATE
    REFRESH_CLIENT_ON_RECONNECT = False
    # Inworld's published pricing varies by routed model; leave as 0 until we
    # wire per-model accounting. The conv app's cost display will just show $0.
    AUDIO_INPUT_COST_PER_1M = 0.0
    AUDIO_OUTPUT_COST_PER_1M = 0.0
    TEXT_INPUT_COST_PER_1M = 0.0
    TEXT_OUTPUT_COST_PER_1M = 0.0
    IMAGE_INPUT_COST_PER_1M = 0.0

    def _get_session_instructions(self) -> str:
        return get_session_instructions()

    def _get_session_voice(self, default: str | None = None) -> str:
        return get_session_voice(default)

    def _get_active_tool_specs(self) -> list[dict[str, Any]]:
        return get_active_tool_specs(self.deps)

    def _get_session_config(self, tool_specs: list[dict[str, Any]]) -> dict[str, Any]:  # type: ignore[override]
        """Return the Inworld realtime session config.

        Returned as a plain dict (rather than the OpenAI TypedDicts the other
        handlers use) because Inworld's schema diverges from OpenAI's:
        `audio.output.model` (TTS routing), a session-level `model` for LLM
        routing, `providerData.stt.voice_profile`, and a different default
        transcription model (no `gpt-4o-transcribe` — Inworld uses providers
        like `assemblyai/u3-rt-pro`).

        Audio format is set explicitly to 16 kHz PCM (Inworld's server default
        is 24 kHz, but the API accepts an explicit rate). This matches
        `SAMPLE_RATE = INWORLD_SAMPLE_RATE` and the Reachy Mini audio path.
        """
        rate = INWORLD_SAMPLE_RATE
        return {
            "type": "realtime",
            # config.MODEL_NAME is resolved by _resolve_model_name to a
            # non-empty default (INWORLD_DEFAULT_LLM) when MODEL_NAME is
            # unset, so no fallback `or` clause is needed here.
            "model": config.MODEL_NAME,
            "instructions": self._get_session_instructions(),
            "output_modalities": ["audio", "text"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": rate},
                    "transcription": {
                        "model": INWORLD_STT_MODEL,
                        # https://docs.inworld.ai/stt/voice-profiles — per-utterance
                        # age/emotion/pitch/vocalStyle/accent classification piggybacks
                        # on transcription. Returned payload populates VoiceProfileStore.
                        "voiceProfileConfig": {
                            "enableVoiceProfile": True,
                            "topN": 5,
                        },
                    },
                    "turn_detection": {
                        "type": "semantic_vad",
                        "eagerness": "medium",
                        "create_response": True,
                        "interrupt_response": True,
                    },
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": rate},
                    "model": INWORLD_TTS_MODEL,
                    "voice": self.get_current_voice(),
                },
            },
            "providerData": {
                # Older realtime-API path for the same toggle; harmless to also
                # set in case the new voiceProfileConfig location is ignored.
                "stt": {"voice_profile": True},
            },
            "tools": to_realtime_tools_config(tool_specs),
            "tool_choice": "auto",
        }

    async def get_available_voices(self) -> list[str]:
        """Return the curated Inworld voice catalog from `config.INWORLD_AVAILABLE_VOICES`.

        Inworld doesn't expose a public model-introspection endpoint analogous
        to OpenAI's `/v1/models/<id>`, so we fall back to the hardcoded list.
        """
        return await super().get_available_voices()

    async def _build_realtime_client(self) -> AsyncOpenAI:
        api_key = (os.environ.get("INWORLD_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError(
                "INWORLD_API_KEY must be set (base64-encoded API key from "
                "https://studio.inworld.ai > API Keys) to use BACKEND_PROVIDER=inworld."
            )
        client = _InworldAsyncOpenAI(
            api_key="UNUSED",  # SDK's Bearer auth is overridden via inworld_basic_auth
            base_url=INWORLD_HTTP_BASE,
            websocket_base_url=INWORLD_WS_BASE,
            inworld_basic_auth=f"Basic {api_key}",
        )
        # Inworld's URL needs ?key=<session_id>&protocol=realtime; the base
        # class forwards self._realtime_connect_query as extra_query on connect.
        self._realtime_connect_query = {
            "key": f"reachy-mini-{int(time.time() * 1000)}",
            "protocol": "realtime",
        }
        return client
