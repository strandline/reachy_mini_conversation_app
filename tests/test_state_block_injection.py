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

from reachy_mini_conversation_app.base_realtime import (
    _inject_state_block,
    _format_entity_roster,
)


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


def test_roster_excludes_pets_so_household_line_owns_them():
    # The "Known entities" roster excludes pets: the authoritative Household line
    # renders the speaker's pets with ownership, and a flat all-households pet
    # list with no ownership gets conflated into "our household" (e.g. another
    # household's dog). People/places/topics still appear.
    entities = [
        {"name": "Jason", "kind": "person"},
        {"name": "Sammy", "kind": "pet", "extra": {"species": "dog"}},
        {"name": "Seattle", "kind": "place"},
    ]
    roster = _format_entity_roster(entities, exclude_kinds=frozenset({"pet"}))
    assert "Jason" in roster
    assert "Seattle" in roster
    assert "Sammy" not in roster
    assert "pets:" not in roster


def test_roster_without_exclusion_still_lists_pets():
    # Formatter stays general: with no exclusion, pets render with species (only
    # the state-block call site opts into excluding them).
    entities = [{"name": "Gingy", "kind": "pet", "extra": {"species": "cat"}}]
    roster = _format_entity_roster(entities)
    assert "Gingy (cat)" in roster
