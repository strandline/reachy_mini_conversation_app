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
