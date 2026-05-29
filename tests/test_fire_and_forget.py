"""Tool.fire_and_forget classification + is_fire_and_forget lookup.

The realtime ack-on-dispatch wiring that consumes this flag (answer a
fire-and-forget tool's function_call immediately so its call_id isn't left
dangling for the Inworld→chat-completions proxy) is verified live — the race
can't be reproduced in pytest. These tests guard the deterministic
classification that wiring depends on.
"""

from reachy_mini_conversation_app.tools.core_tools import Tool


def test_tool_base_default_is_result_bearing():
    """A new tool is result-bearing unless it explicitly opts into fire-and-forget."""
    assert Tool.fire_and_forget is False


def test_is_fire_and_forget_reads_registry(monkeypatch):
    """The lookup reads the flag off the registry; unknown names are safe (False)."""
    from reachy_mini_conversation_app.tools import core_tools

    class _FF:
        fire_and_forget = True

    class _RB:
        fire_and_forget = False

    monkeypatch.setattr(core_tools, "ALL_TOOLS", {"ff": _FF(), "rb": _RB()})
    assert core_tools.is_fire_and_forget("ff") is True
    assert core_tools.is_fire_and_forget("rb") is False
    assert core_tools.is_fire_and_forget("unknown") is False  # unknown → safe default


def test_face_id_write_tools_are_fire_and_forget():
    """The face-ID write tools (submodule-resident) carry the flag."""
    from reachy_mini_conversation_app.tools.enroll_face import EnrollFace
    from reachy_mini_conversation_app.tools.correct_identity import CorrectIdentity

    assert CorrectIdentity.fire_and_forget is True
    assert EnrollFace.fire_and_forget is True
