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

from reachy_mini_conversation_app.config import INWORLD_BACKEND, config
from reachy_mini_conversation_app.prompts import get_session_voice, get_session_instructions
from reachy_mini_conversation_app.base_realtime import BaseRealtimeHandler, to_realtime_tools_config
from reachy_mini_conversation_app.tools.core_tools import get_active_tool_specs


logger = logging.getLogger(__name__)

__all__ = ["InworldRealtimeHandler"]


INWORLD_WS_BASE = "wss://api.inworld.ai/api/v1"
INWORLD_HTTP_BASE = "https://api.inworld.ai/api/v1"
INWORLD_DEFAULT_LLM = "openai/gpt-4o-mini"
INWORLD_TTS_MODEL = "inworld-tts-2"
INWORLD_OUTPUT_SAMPLE_RATE = 16000


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
        """Return an `_InworldConnectionManager` instead of the SDK default."""
        ws_opts = websocket_connection_options if websocket_connection_options is not None else {}
        return _InworldConnectionManager(
            client=self._client,
            extra_query=extra_query if extra_query is not None else {},
            extra_headers=extra_headers if extra_headers is not None else {},
            websocket_connection_options=ws_opts,  # type: ignore[arg-type]
            call_id=call_id,
            model=model,
        )


class _InworldAsyncOpenAI(AsyncOpenAI):
    """`AsyncOpenAI` subclass that exposes Inworld's URL/auth via `.realtime`."""

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
    SAMPLE_RATE = INWORLD_OUTPUT_SAMPLE_RATE
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
        handlers use) because Inworld's schema adds `audio.output.model` and
        a session-level `model` field that aren't in the OpenAI TypedDict
        definitions.
        """
        rate = INWORLD_OUTPUT_SAMPLE_RATE
        return {
            "type": "realtime",
            "model": config.MODEL_NAME or INWORLD_DEFAULT_LLM,
            "instructions": self._get_session_instructions(),
            "output_modalities": ["audio", "text"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": rate},
                    "transcription": {"model": "gpt-4o-transcribe", "language": "en"},
                    "turn_detection": {"type": "server_vad", "interrupt_response": True},
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": rate},
                    "model": INWORLD_TTS_MODEL,
                    "voice": self.get_current_voice(),
                },
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
            api_key="UNUSED",  # overridden by default_headers
            base_url=INWORLD_HTTP_BASE,
            websocket_base_url=INWORLD_WS_BASE,
            default_headers={"Authorization": f"Basic {api_key}"},
        )
        # Inworld's URL needs ?key=<session_id>&protocol=realtime; the base
        # class forwards self._realtime_connect_query as extra_query on connect.
        self._realtime_connect_query = {
            "key": f"reachy-mini-{int(time.time() * 1000)}",
            "protocol": "realtime",
        }
        return client
