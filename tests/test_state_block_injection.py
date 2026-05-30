"""State-block injection into the realtime session config (both backend shapes).

The dynamic state block is appended to the session's `instructions` at session
start. OpenAI/HF/Gemini return a RealtimeSessionCreateRequestParam (attribute
access); Inworld returns a plain dict (item access — see
inworld_realtime._get_session_config). The original session-start code only did
attribute assignment (`session_config.instructions = ...`), which raised
AttributeError on Inworld's dict → the state block was silently dropped at
session open (backfilled only seconds later by the first refresh). This pins the
shape-agnostic injection that fixes it.
"""

from reachy_mini_conversation_app.base_realtime import _inject_state_block


class _ObjConfig:
    """Mimics RealtimeSessionCreateRequestParam: a settable .instructions attr."""

    def __init__(self, instructions: str = "") -> None:
        self.instructions = instructions


def test_inject_into_dict_config_appends_block():
    # Inworld shape: a plain dict carrying an "instructions" key.
    cfg = {"type": "realtime", "instructions": "BASE PERSONA"}
    _inject_state_block(cfg, "STATE BLOCK")
    assert cfg["instructions"] == "BASE PERSONA\n\nSTATE BLOCK"


def test_inject_into_object_config_appends_block():
    # OpenAI/HF/Gemini shape: an object with a settable .instructions attribute.
    cfg = _ObjConfig(instructions="BASE PERSONA")
    _inject_state_block(cfg, "STATE BLOCK")
    assert cfg.instructions == "BASE PERSONA\n\nSTATE BLOCK"


def test_inject_empty_block_is_noop():
    # No state block → instructions untouched (no dangling blank lines).
    cfg = {"instructions": "BASE PERSONA"}
    _inject_state_block(cfg, "")
    assert cfg["instructions"] == "BASE PERSONA"


def test_inject_into_dict_without_instructions_key():
    # Defensive: a dict missing "instructions" still receives the block (no KeyError).
    cfg = {"type": "realtime"}
    _inject_state_block(cfg, "STATE BLOCK")
    assert cfg["instructions"] == "\n\nSTATE BLOCK"
