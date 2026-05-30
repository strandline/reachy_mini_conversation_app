import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from reachy_mini_conversation_app import base_realtime
from reachy_mini_conversation_app.openai_realtime import OpenaiRealtimeHandler


def _bare_handler():
    # __new__ skips __init__ (no backend/credentials needed); OpenaiRealtimeHandler
    # is concrete so it is instantiable, unlike the abstract BaseRealtimeHandler.
    return OpenaiRealtimeHandler.__new__(OpenaiRealtimeHandler)


@pytest.mark.asyncio
async def test_inject_passive_item_is_passive():
    """Creates a user/input_text item and does NOT trigger a response."""
    handler = _bare_handler()
    conn = MagicMock()
    conn.conversation.item.create = AsyncMock()
    handler.connection = conn
    handler._safe_response_create = AsyncMock()

    await handler.inject_passive_item("[recall: the Miata]")

    conn.conversation.item.create.assert_awaited_once()
    item = conn.conversation.item.create.await_args.kwargs["item"]
    assert item["type"] == "message"
    assert item["role"] == "user"
    assert item["content"][0]["type"] == "input_text"
    assert item["content"][0]["text"] == "[recall: the Miata]"
    handler._safe_response_create.assert_not_called()


@pytest.mark.asyncio
async def test_inject_passive_item_noop_without_connection():
    handler = _bare_handler()
    handler.connection = None
    await handler.inject_passive_item("[recall: x]")  # must not raise


@pytest.mark.asyncio
async def test_maybe_run_subconscious_injects_when_due(monkeypatch):
    handler = _bare_handler()
    handler._user_turn_count = 3            # divisible by N=3 → due
    handler._latest_face_recognition = None
    handler._last_user_transcript = "tell me about the car"
    handler.inject_passive_item = AsyncMock()
    handler._response_done_event = asyncio.Event()
    handler._response_done_event.set()      # idle: safe to inject
    fake = MagicMock()
    fake.render_delta = MagicMock(return_value="[recall: the Miata]")
    monkeypatch.setattr(base_realtime, "_SUBCONSCIOUS", fake)
    monkeypatch.setattr(base_realtime, "_SUBCONSCIOUS_EVERY_N_TURNS", 3)

    await handler._maybe_run_subconscious()

    fake.render_delta.assert_called_once_with("tell me about the car", None)
    handler.inject_passive_item.assert_awaited_once_with("[recall: the Miata]")


@pytest.mark.asyncio
async def test_maybe_run_subconscious_skips_off_cadence(monkeypatch):
    handler = _bare_handler()
    handler._user_turn_count = 2            # 2 % 3 != 0 → skip
    handler.inject_passive_item = AsyncMock()
    monkeypatch.setattr(base_realtime, "_SUBCONSCIOUS", MagicMock())
    monkeypatch.setattr(base_realtime, "_SUBCONSCIOUS_EVERY_N_TURNS", 3)

    await handler._maybe_run_subconscious()

    handler.inject_passive_item.assert_not_called()


@pytest.mark.asyncio
async def test_maybe_run_subconscious_noop_when_module_absent(monkeypatch):
    handler = _bare_handler()
    handler._user_turn_count = 3
    handler.inject_passive_item = AsyncMock()
    monkeypatch.setattr(base_realtime, "_SUBCONSCIOUS", None)

    await handler._maybe_run_subconscious()

    handler.inject_passive_item.assert_not_called()


@pytest.mark.asyncio
async def test_maybe_run_subconscious_skips_none_delta(monkeypatch):
    handler = _bare_handler()
    handler._user_turn_count = 3
    handler._latest_face_recognition = None
    handler._last_user_transcript = "x"
    handler.inject_passive_item = AsyncMock()
    handler._response_done_event = asyncio.Event()
    handler._response_done_event.set()
    fake = MagicMock()
    fake.render_delta = MagicMock(return_value=None)   # nothing to surface
    monkeypatch.setattr(base_realtime, "_SUBCONSCIOUS", fake)
    monkeypatch.setattr(base_realtime, "_SUBCONSCIOUS_EVERY_N_TURNS", 3)

    await handler._maybe_run_subconscious()

    handler.inject_passive_item.assert_not_called()
