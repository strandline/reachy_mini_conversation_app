"""Phase 2 face-ID tests — DI-mocked; no real InsightFace model, no camera.

Part B covers ``vision/face_id.py`` (``FaceRecognizer`` + ``embed_largest`` +
``initialize_face_recognizer``). Parts C and D extend this same module. The
fake ``app``/face objects stand in for insightface so these run without the
package installed and without a webcam.
"""

import os
import sys
import json
import asyncio
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import pytest_asyncio

from reachy_mini_conversation_app.vision.face_id import (
    FaceRecognizer,
    initialize_face_recognizer,
)
from reachy_mini_conversation_app.openai_realtime import OpenaiRealtimeHandler
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


class _FakeFace:
    """Stand-in for an insightface Face: .bbox, .embedding, .det_score."""

    def __init__(self, bbox, embedding, det_score=0.9):
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.embedding = np.asarray(embedding, dtype=np.float32)
        self.det_score = det_score


class _FakeApp:
    """Stand-in for insightface FaceAnalysis; .get() returns canned faces."""

    def __init__(self, faces):
        self._faces = faces

    def get(self, frame_bgr):
        return self._faces


def _frame():
    return np.zeros((120, 160, 3), dtype=np.uint8)


def _onehot(i, n=512):
    v = np.zeros(n, dtype=np.float32)
    v[i] = 1.0
    return v


def test_embed_largest_picks_largest_bbox():
    """The largest-AREA detection wins, not the first or the highest score."""
    # Orthogonal one-hot embeddings so the two faces stay distinguishable after
    # L2 normalization (scalar-multiple vectors would collapse to the same unit
    # vector and make the assertion vacuous).
    small = _FakeFace(bbox=[0, 0, 10, 10], embedding=_onehot(0))     # area 100
    large = _FakeFace(bbox=[0, 0, 100, 100], embedding=_onehot(1))   # area 10000
    rec = FaceRecognizer(app=_FakeApp([small, large]))
    out = rec.embed_largest(_frame())
    assert out is not None
    # Largest-AREA face wins → its one-hot at index 1.
    assert int(np.argmax(out["embedding"])) == 1
    assert out["bbox"].tolist() == [0, 0, 100, 100]


def test_embed_largest_normalizes_unnormalized_embedding():
    """embed_largest L2-normalizes the raw embedding itself (float32)."""
    # Raw insightface embeddings are not unit-norm; embed_largest owns the
    # normalization (reads .embedding, not .normed_embedding).
    face = _FakeFace(
        bbox=[0, 0, 50, 50], embedding=np.full(512, 25.8, dtype=np.float32)
    )
    rec = FaceRecognizer(app=_FakeApp([face]))
    out = rec.embed_largest(_frame())
    assert out is not None
    assert np.isclose(np.linalg.norm(out["embedding"]), 1.0, atol=1e-6)
    assert out["embedding"].dtype == np.float32


@pytest.mark.parametrize("faces", [[], None])
def test_embed_largest_returns_none_when_no_faces(faces):
    """No detections (empty list or None) yields None, not a crash."""
    rec = FaceRecognizer(app=_FakeApp(faces))
    assert rec.embed_largest(_frame()) is None


def test_embed_largest_bbox_and_det_score_passthrough():
    """The returned dict carries bbox (float32) and det_score for C/D."""
    face = _FakeFace(bbox=[5, 5, 55, 75], embedding=_onehot(3), det_score=0.92)
    rec = FaceRecognizer(app=_FakeApp([face]))
    out = rec.embed_largest(_frame())
    assert out is not None
    assert out["det_score"] == pytest.approx(0.92)
    assert out["bbox"].tolist() == [5, 5, 55, 75]


def test_initialize_face_recognizer_returns_none_without_insightface(monkeypatch):
    """Missing insightface → factory returns None (silent degrade), no raise."""
    # The factory diverges from initialize_vision_processor: it returns None
    # (silent degrade) instead of re-raising, so the camera-present auto-enable
    # path never blocks startup.
    def _boom(self):
        raise ImportError("no insightface")

    monkeypatch.setattr(FaceRecognizer, "load", _boom)
    assert initialize_face_recognizer() is None


# ---------------------------------------------------------------------------
# Step S — shared ToolDependencies fields + deps-attach wiring
# ---------------------------------------------------------------------------


def test_tool_dependencies_face_fields_default_none():
    """ToolDependencies gains face_recognizer/capture_episode_id/session_recognized_ids, all None."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    assert deps.face_recognizer is None
    assert deps.capture_episode_id is None
    assert deps.session_recognized_ids is None


@pytest.mark.asyncio
async def test_handler_shares_session_recognized_ids_with_deps():
    """The handler attaches its live continuity set to deps BY IDENTITY.

    Pins the shared-object invariant: enroll/correct (Part D) read
    deps.session_recognized_ids while the recognizer (Part C) mutates
    self._session_recognized_ids — they must be the same object, and stay so
    across reconnects (the connect-path reset must clear in place, not rebind).
    """
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = OpenaiRealtimeHandler(deps)
    assert handler.deps.session_recognized_ids is handler._session_recognized_ids


# ---------------------------------------------------------------------------
# Part C — _run_face_recognition + triggers + state block + _tag_identity
#
# Integration-style: the REAL outer-repo store modules (_memory_store /
# _face_match / _capture_store / _speaker_state) run against a temp DB and are
# wired onto the base_realtime globals (the conv-app conftest pops the tools-dir
# env var, so those globals import as None). Camera + recognizer are DI fakes.
# The round-trip catches what mocks can't: log_face_sighting_sync RAISES on a
# bad `source`, kwarg drift is a real TypeError, and a written 'recognize'
# sighting must become a future exemplar (the whole "recognize-next-time" point).
# Skips cleanly when the outer-repo tools/ dir isn't importable (a standalone
# submodule/fork checkout).
# ---------------------------------------------------------------------------

_OUTER_TOOLS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "tools")
)


def _import_outer_stores():
    """Import the outer-repo store modules, or None if the tools/ dir is absent."""
    if _OUTER_TOOLS_DIR not in sys.path:
        sys.path.insert(0, _OUTER_TOOLS_DIR)
    try:
        import _face_match
        import _memory_store
        import _capture_store
        import _speaker_state

        return _memory_store, _face_match, _capture_store, _speaker_state
    except ImportError:
        return None


_OUTER_STORES = _import_outer_stores()


@pytest.mark.skipif(
    _OUTER_STORES is None, reason="outer-repo tools/ not importable (no _memory_store)"
)
def test_real_memory_db_is_isolated_during_tests():
    """The session isolation fixture keeps tests off the real ~/.bemo/memory.db.

    _read_latest_mood (and peers) import _memory_store DIRECTLY, bypassing the
    base_realtime _MEMORY_STORE global the conftest neutralizes. Once this
    integration harness puts tools/ on sys.path that import resolves, so the
    conftest's _isolate_real_memory_db fixture must redirect DB_PATH away from
    the developer's real DB. Without that fixture this asserts the real path.
    """
    import _memory_store

    real = Path.home() / ".bemo" / "memory.db"
    assert Path(_memory_store.DB_PATH).resolve() != real.resolve()


def _gray_probe() -> np.ndarray:
    """Build a unit vector whose cosine with one-hot e0 is 0.37 — squarely gray.

    0.37 ∈ [_FACE_LOW (0.30), _FACE_HIGH (0.45)); 0.929 ≈ sqrt(1 - 0.37²) keeps
    it unit so score_profiles' unit-vector assumption holds. e1 carries the
    orthogonal remainder so max-cosine against an {e0}-only gallery is exactly
    0.37.
    """
    v = 0.37 * _onehot(0) + 0.929 * _onehot(1)
    return v.astype(np.float32)


def _all_sightings(ms) -> list[dict]:
    """Return all face_sightings rows (id-ordered) from the temp DB."""
    conn = ms._connect()
    try:
        rows = conn.execute(
            "SELECT id, entity_id, episode_id, confidence, source "
            "FROM face_sightings ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _participants(ms, episode_id: int) -> list[str]:
    """Return decoded episodes.participants for an episode (empty when unset)."""
    conn = ms._connect()
    try:
        row = conn.execute(
            "SELECT participants FROM episodes WHERE id = ?", (episode_id,)
        ).fetchone()
        return json.loads(row["participants"]) if row and row["participants"] else []
    finally:
        conn.close()


@pytest_asyncio.fixture
async def face_ctx(tmp_path, monkeypatch):
    """Real outer-repo stores on a temp DB, wired onto base_realtime globals.

    Yields a namespace with a built handler (fake camera + recognizer), the four
    store modules, and an open episode id. Async so the handler is constructed
    inside the running loop (its __init__ reads asyncio.get_event_loop().time()).
    refresh_session_instructions is spied so the per-tier "did it refresh?"
    assertions work and the real session.update path never runs.
    """
    if _OUTER_STORES is None:
        pytest.skip("outer-repo tools/ not importable (standalone submodule checkout)")
    ms, fm, cs, ss = _OUTER_STORES

    # Temp DB — mirror the outer-repo temp_db fixture (reset the schema flag).
    monkeypatch.setattr(ms, "DB_PATH", tmp_path / "memory.db")
    monkeypatch.setattr(ms, "_initialized", False)

    # Wire the real modules onto the base_realtime globals (conftest popped the
    # tools-dir env var, so they imported as None at module load).
    import reachy_mini_conversation_app.base_realtime as br

    monkeypatch.setattr(br, "_MEMORY_STORE", ms)
    monkeypatch.setattr(br, "_FACE_MATCH", fm)
    monkeypatch.setattr(br, "_CAPTURE_STORE", cs)
    monkeypatch.setattr(br, "_SPEAKER_STATE", ss)

    # _speaker_state is in-memory module-global; reset so a pin never leaks
    # between tests (gotcha: would silently corroborate the next gray scan).
    ss.clear_current_speaker()

    camera = MagicMock()
    camera.get_latest_frame.return_value = np.zeros((120, 160, 3), dtype=np.uint8)
    recognizer = MagicMock()
    recognizer.embed_largest.return_value = None  # default: no face; tests override

    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        camera_worker=camera,
        face_recognizer=recognizer,
    )
    handler = OpenaiRealtimeHandler(deps)

    episode_id = await cs.open_episode([])
    handler._capture_episode_id = episode_id
    handler.deps.capture_episode_id = episode_id

    refresh = AsyncMock()
    monkeypatch.setattr(handler, "refresh_session_instructions", refresh)
    handler.deps.refresh_session_instructions = refresh

    try:
        yield SimpleNamespace(
            handler=handler, ms=ms, fm=fm, cs=cs, ss=ss,
            camera=camera, recognizer=recognizer,
            episode_id=episode_id, refresh=refresh,
        )
    finally:
        ss.clear_current_speaker()


def _set_probe(ctx, vec: np.ndarray) -> None:
    """Make the fake recognizer return a probe dict for `vec`."""
    ctx.recognizer.embed_largest.return_value = {
        "embedding": vec.astype(np.float32),
        "bbox": np.array([0, 0, 50, 50], dtype=np.float32),
        "det_score": 0.9,
    }


@pytest.mark.asyncio
async def test_run_face_recognition_high_tier_logs_sighting_and_pins(face_ctx):
    """A confident match logs a recognize sighting, pins, tags, and refreshes."""
    ctx = face_ctx
    eid = ctx.ms.upsert_entity_sync("Jeff", kind="person")
    ctx.ms.seed_face_centroid_sync(eid, _onehot(0))
    _set_probe(ctx, _onehot(0))  # identical to centroid → cosine 1.0 → match

    await ctx.handler._run_face_recognition()

    sightings = _all_sightings(ctx.ms)
    assert len(sightings) == 1
    assert sightings[0]["entity_id"] == eid
    assert sightings[0]["source"] == "recognize"
    assert sightings[0]["confidence"] >= 0.45
    assert ctx.ss.get_current_speaker()["id"] == eid
    assert "Jeff" in _participants(ctx.ms, ctx.episode_id)
    assert ctx.handler._latest_face_recognition is not None
    cached_eid, cached_name, _ts = ctx.handler._latest_face_recognition
    assert (cached_eid, cached_name) == (eid, "Jeff")
    assert eid in ctx.handler._session_recognized_ids
    ctx.refresh.assert_awaited()
    # Round-trip: the just-written recognize sighting becomes a future exemplar.
    match_set = ctx.ms.get_face_match_set_sync()
    jeff = next(p for p in match_set if p["entity_id"] == eid)
    assert len(jeff["exemplars"]) >= 2  # centroid + the new sighting


@pytest.mark.asyncio
async def test_run_face_recognition_none_tier_logs_unmatched_and_sets_flag(face_ctx):
    """A detected-but-unmatched face logs entity_id=NULL and raises the cue flag."""
    ctx = face_ctx
    # No enrolled entity → empty match set → tier 'none'.
    _set_probe(ctx, _onehot(7))

    await ctx.handler._run_face_recognition()

    sightings = _all_sightings(ctx.ms)
    assert len(sightings) == 1
    assert sightings[0]["entity_id"] is None
    assert sightings[0]["source"] == "recognize"
    assert ctx.handler._face_unrecognized_present is True
    assert ctx.handler._latest_face_recognition is None
    assert ctx.ss.get_current_speaker() is None
    ctx.refresh.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_face_recognition_none_tier_clears_stale_match(face_ctx):
    """An unknown face (NONE tier) clears a prior fresh recognition (A→B swap).

    Without clearing, the state block would keep naming the person who left
    (cache fresh for up to _FACE_STALE_SECONDS) AND suppress the newcomer's
    enrollment cue. NONE means a face IS present but unmatched — a positive
    signal the prior speaker is gone — so the cache is dropped, like NO-FACE.
    """
    ctx = face_ctx
    eid = ctx.ms.upsert_entity_sync("Alice", kind="person")
    ctx.ms.seed_face_centroid_sync(eid, _onehot(0))
    _set_probe(ctx, _onehot(0))  # Alice HIGH-matched → cached
    await ctx.handler._run_face_recognition()
    assert ctx.handler._latest_face_recognition is not None

    _set_probe(ctx, _onehot(7))  # unknown face, orthogonal → NONE tier
    await ctx.handler._run_face_recognition(force=True)  # force past the cooldown

    assert ctx.handler._latest_face_recognition is None  # Alice's stale name dropped
    assert ctx.handler._face_unrecognized_present is True


@pytest.mark.asyncio
async def test_run_face_recognition_no_face_clears_flag(face_ctx):
    """No face in frame → no writes, cache cleared, cue flag off, no pin."""
    ctx = face_ctx
    ctx.recognizer.embed_largest.return_value = None
    ctx.handler._face_unrecognized_present = True  # pretend a prior scan saw one

    await ctx.handler._run_face_recognition()

    assert _all_sightings(ctx.ms) == []
    assert ctx.handler._latest_face_recognition is None
    assert ctx.handler._face_unrecognized_present is False
    assert ctx.ss.get_current_speaker() is None


@pytest.mark.asyncio
async def test_run_face_recognition_gray_uncorroborated_is_noop(face_ctx):
    """A gray match with no corroboration is a pure no-op (no pin, no write, no refresh)."""
    ctx = face_ctx
    eid = ctx.ms.upsert_entity_sync("Ambiguous", kind="person")
    ctx.ms.seed_face_centroid_sync(eid, _onehot(0))
    _set_probe(ctx, _gray_probe())  # cosine 0.37 → gray, nothing corroborates

    await ctx.handler._run_face_recognition()

    assert _all_sightings(ctx.ms) == []
    assert ctx.ss.get_current_speaker() is None
    assert eid not in ctx.handler._session_recognized_ids
    assert ctx.handler._latest_face_recognition is None
    assert _participants(ctx.ms, ctx.episode_id) == []  # nobody attributed
    ctx.refresh.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_face_recognition_gray_corroborated_by_pin_accepts(face_ctx):
    """A gray match corroborated by the current pin is accepted — but not promoted.

    Accepting logs a reinforcement sighting (source='recognize'), keeps the pin,
    attributes the participant, and caches — yet must NOT add to
    _session_recognized_ids (that is HIGH-tier only; otherwise a pin-corroborated
    gray-accept self-corroborates the next scan via "session continuity").
    """
    ctx = face_ctx
    eid = ctx.ms.upsert_entity_sync("Maya", kind="person")
    ctx.ms.seed_face_centroid_sync(eid, _onehot(0))
    ctx.ss.set_current_speaker(eid, "Maya")  # pin corroborates the gray candidate
    _set_probe(ctx, _gray_probe())

    await ctx.handler._run_face_recognition()

    sightings = _all_sightings(ctx.ms)
    assert len(sightings) == 1
    assert sightings[0]["entity_id"] == eid
    assert sightings[0]["source"] == "recognize"
    assert ctx.ss.get_current_speaker()["id"] == eid
    assert "Maya" in _participants(ctx.ms, ctx.episode_id)
    assert ctx.handler._latest_face_recognition is not None
    assert eid not in ctx.handler._session_recognized_ids  # gray must not promote
    ctx.refresh.assert_awaited()


@pytest.mark.asyncio
async def test_run_face_recognition_bails_no_recognizer(face_ctx):
    """No recognizer on deps → returns immediately, touches nothing."""
    ctx = face_ctx
    ctx.handler.deps.face_recognizer = None

    await ctx.handler._run_face_recognition()

    assert _all_sightings(ctx.ms) == []
    ctx.recognizer.embed_largest.assert_not_called()
    ctx.refresh.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_face_recognition_cooldown_respected(face_ctx):
    """Within cooldown the scan is skipped; force=True overrides it."""
    ctx = face_ctx
    ctx.handler._last_face_scan_at = asyncio.get_event_loop().time()

    await ctx.handler._run_face_recognition()  # no force → cooldown gate
    ctx.recognizer.embed_largest.assert_not_called()

    await ctx.handler._run_face_recognition(force=True)
    ctx.recognizer.embed_largest.assert_called()


@pytest.mark.asyncio
async def test_run_face_recognition_no_frame_does_not_burn_cooldown(face_ctx):
    """A None frame returns without setting _last_face_scan_at (no burned cooldown)."""
    ctx = face_ctx
    ctx.camera.get_latest_frame.return_value = None
    ctx.handler._last_face_scan_at = None

    await ctx.handler._run_face_recognition(force=True)

    assert ctx.handler._last_face_scan_at is None
    assert _all_sightings(ctx.ms) == []


@pytest.mark.asyncio
async def test_state_block_emits_who_you_can_see_when_fresh_match(face_ctx):
    """A fresh cached recognition surfaces the 'Who you can see' line, no cue."""
    ctx = face_ctx
    now = asyncio.get_event_loop().time()
    ctx.handler._latest_face_recognition = (1, "Jeff", now)

    block = await ctx.handler._build_state_block()

    assert "Who you can see right now: Jeff" in block
    assert "you don't yet recognize" not in block


@pytest.mark.asyncio
async def test_state_block_suppresses_stale_match(face_ctx):
    """A recognition older than the stale window is dropped from the block."""
    ctx = face_ctx
    stale = asyncio.get_event_loop().time() - 700  # > _FACE_STALE_SECONDS (600)
    ctx.handler._latest_face_recognition = (1, "Jeff", stale)

    block = await ctx.handler._build_state_block()

    assert "Who you can see right now" not in block


@pytest.mark.asyncio
async def test_state_block_emits_enrollment_cue_only_for_active_conversant(face_ctx):
    """The enrollment cue appears only once the user has actually spoken."""
    ctx = face_ctx
    ctx.handler._face_unrecognized_present = True

    ctx.handler._user_turn_count = 0
    block_silent = await ctx.handler._build_state_block()
    assert "you don't yet recognize" not in block_silent

    ctx.handler._user_turn_count = 1
    block_active = await ctx.handler._build_state_block()
    assert "you don't yet recognize" in block_active
    assert "Who you can see right now" not in block_active


@pytest.mark.asyncio
async def test_tag_identity_uses_merge_participant_not_replace(face_ctx):
    """High-tier tagging is additive: a later set_participants keeps the face name."""
    ctx = face_ctx
    eid = ctx.ms.upsert_entity_sync("Jeff", kind="person")
    ctx.ms.seed_face_centroid_sync(eid, _onehot(0))
    _set_probe(ctx, _onehot(0))

    await ctx.handler._run_face_recognition()
    assert "Jeff" in _participants(ctx.ms, ctx.episode_id)

    # A later session-level participant write must not drop the face-merged name.
    await ctx.cs.set_participants(ctx.episode_id, ["Other"])
    parts = _participants(ctx.ms, ctx.episode_id)
    assert "Jeff" in parts
    assert "Other" in parts


@pytest.mark.asyncio
async def test_send_idle_signal_triggers_face_recognition(face_ctx, monkeypatch):
    """An idle tick fires a (cooldown-gated) face scan alongside the scene scan."""
    ctx = face_ctx
    ctx.handler.connection = None  # early-return before any WebSocket work
    monkeypatch.setattr(ctx.handler, "_run_scene_observation", AsyncMock())
    face_scan = AsyncMock()
    monkeypatch.setattr(ctx.handler, "_run_face_recognition", face_scan)

    await ctx.handler.send_idle_signal(60.0)

    face_scan.assert_awaited()


# ---------------------------------------------------------------------------
# Part D — enroll_face + correct_identity tools
#
# Same integration harness as Part C (real outer-repo stores on a temp DB via
# face_ctx). The tools are Tool subclasses called directly with face_ctx.handler.deps
# (carries the fake camera + recognizer, the open episode, the refresh spy, and the
# shared session_recognized_ids set).
# ---------------------------------------------------------------------------

import reachy_mini_conversation_app.tools.core_tools as ct  # noqa: E402
from reachy_mini_conversation_app.tools.enroll_face import EnrollFace  # noqa: E402
from reachy_mini_conversation_app.tools.correct_identity import CorrectIdentity  # noqa: E402


def _entity_count(ms) -> int:
    return len(ms.list_all_entities_sync())


def _entity_by_name(ms, name: str) -> dict | None:
    for e in ms.list_all_entities_sync():
        if e["name"].lower() == name.lower():
            return e
    return None


# --- enroll_face ---------------------------------------------------------


@pytest.mark.asyncio
async def test_enroll_face_new_creates_entity_and_seeds(face_ctx):
    """A novel face+name creates an entity, seeds a centroid, logs an enroll sighting, pins."""
    ctx = face_ctx
    _set_probe(ctx, _onehot(0))  # nobody enrolled yet → match set empty → 'new'

    res = await EnrollFace()(ctx.handler.deps, name="Jason")

    assert res["status"] == "enrolled"
    assert res["action"] == "new"
    assert res["name"] == "Jason"
    eid = res["entity_id"]
    assert _entity_count(ctx.ms) == 1
    # Seeded → now matchable.
    match_set = ctx.ms.get_face_match_set_sync()
    assert any(p["entity_id"] == eid and p["name"] == "Jason" for p in match_set)
    sightings = _all_sightings(ctx.ms)
    assert len(sightings) == 1
    assert sightings[0]["source"] == "enroll"
    assert sightings[0]["entity_id"] == eid
    assert ctx.ss.get_current_speaker() == {"id": eid, "name": "Jason"}
    assert "Jason" in _participants(ctx.ms, ctx.episode_id)
    assert eid in ctx.handler.deps.session_recognized_ids  # consent → seeds continuity
    ctx.refresh.assert_awaited()


@pytest.mark.asyncio
async def test_enroll_face_reuse_known_face_adds_alias_no_duplicate(face_ctx):
    """A face that already matches reuses the entity, adds the new name as an alias."""
    ctx = face_ctx
    eid = ctx.ms.upsert_entity_sync("Jeff", kind="person")
    ctx.ms.seed_face_centroid_sync(eid, _onehot(0))
    _set_probe(ctx, _onehot(0))  # cosine 1.0 ≥ high → face known → reuse

    res = await EnrollFace()(ctx.handler.deps, name="Jeff Gordon")

    assert res["action"] == "reuse"
    assert res["entity_id"] == eid
    assert res["name"] == "Jeff"  # canonical, not the spoken fuller name
    assert _entity_count(ctx.ms) == 1  # NO duplicate entity
    jeff = _entity_by_name(ctx.ms, "Jeff")
    assert any(a.lower() == "jeff gordon" for a in jeff["aliases"])
    # Reinforcing enroll sighting under the existing entity.
    sightings = _all_sightings(ctx.ms)
    assert len(sightings) == 1
    assert sightings[0]["source"] == "enroll"
    assert sightings[0]["entity_id"] == eid
    assert ctx.ss.get_current_speaker()["name"] == "Jeff"


@pytest.mark.asyncio
async def test_enroll_face_name_collision_does_not_enroll(face_ctx):
    """A new face whose spoken name is taken → collision, no entity/sighting/pin."""
    ctx = face_ctx
    # 'Jeff' exists but has NO face centroid → this face won't match (collision).
    ctx.ms.upsert_entity_sync("Jeff", kind="person")
    _set_probe(ctx, _onehot(5))

    res = await EnrollFace()(ctx.handler.deps, name="Jeff")

    assert res["status"] == "collision"
    assert _entity_count(ctx.ms) == 1  # no new entity
    assert _all_sightings(ctx.ms) == []  # nothing logged
    assert ctx.ss.get_current_speaker() is None  # not pinned
    ctx.refresh.assert_not_awaited()


@pytest.mark.asyncio
async def test_enroll_face_no_frame_returns_error(face_ctx):
    """No camera frame → error, no writes."""
    ctx = face_ctx
    ctx.camera.get_latest_frame.return_value = None

    res = await EnrollFace()(ctx.handler.deps, name="Jason")

    assert "error" in res
    assert _entity_count(ctx.ms) == 0
    assert _all_sightings(ctx.ms) == []


@pytest.mark.asyncio
async def test_enroll_face_no_recognizer_returns_error(face_ctx):
    """No recognizer on deps → error, no writes, recognizer never invoked."""
    ctx = face_ctx
    ctx.handler.deps.face_recognizer = None

    res = await EnrollFace()(ctx.handler.deps, name="Jason")

    assert "error" in res
    assert _entity_count(ctx.ms) == 0
    ctx.recognizer.embed_largest.assert_not_called()


@pytest.mark.asyncio
async def test_enroll_face_empty_name_returns_error(face_ctx):
    """Empty name → error before any embedding happens."""
    ctx = face_ctx

    res = await EnrollFace()(ctx.handler.deps, name="   ")

    assert "error" in res
    ctx.recognizer.embed_largest.assert_not_called()
    assert _entity_count(ctx.ms) == 0


@pytest.mark.asyncio
async def test_enroll_face_no_face_detected(face_ctx):
    """A frame with no detectable face → no_face status, no writes."""
    ctx = face_ctx
    ctx.recognizer.embed_largest.return_value = None

    res = await EnrollFace()(ctx.handler.deps, name="Jason")

    assert res["status"] == "no_face"
    assert _entity_count(ctx.ms) == 0
    assert _all_sightings(ctx.ms) == []


# --- correct_identity ----------------------------------------------------


@pytest.mark.asyncio
async def test_correct_identity_relabels_session_sightings_and_repins(face_ctx):
    """A misgreet correction relabels this episode's recognize sightings to the new name."""
    ctx = face_ctx
    wrong = ctx.ms.upsert_entity_sync("Wrongname", kind="person")
    # Two auto-recognize sightings logged under the wrong entity this episode.
    emb = _onehot(0)
    s1 = ctx.ms.log_face_sighting_sync(
        entity_id=wrong, episode_id=ctx.episode_id, embedding=emb,
        confidence=0.5, source="recognize")
    s2 = ctx.ms.log_face_sighting_sync(
        entity_id=wrong, episode_id=ctx.episode_id, embedding=emb,
        confidence=0.5, source="recognize")
    # No recognizer-bind for this test → isolate the relabel path.
    ctx.handler.deps.face_recognizer = None

    res = await CorrectIdentity()(ctx.handler.deps, name="Maya")

    assert res["status"] == "corrected"
    assert res["name"] == "Maya"
    assert res["relabeled"] == 2
    maya = _entity_by_name(ctx.ms, "Maya")
    conn = ctx.ms._connect()
    try:
        rows = conn.execute(
            "SELECT entity_id, source FROM face_sightings WHERE id IN (?, ?)", (s1, s2)
        ).fetchall()
    finally:
        conn.close()
    assert all(r["entity_id"] == maya["id"] for r in rows)
    assert all(r["source"] == "correct" for r in rows)
    assert ctx.ss.get_current_speaker() == {"id": maya["id"], "name": "Maya"}
    assert "Maya" in _participants(ctx.ms, ctx.episode_id)
    ctx.refresh.assert_awaited()


@pytest.mark.asyncio
async def test_correct_identity_clear_unsets_identity(face_ctx):
    """clear=true relabels this session's sightings to NULL and clears the pin."""
    ctx = face_ctx
    wrong = ctx.ms.upsert_entity_sync("Wrongname", kind="person")
    ctx.ss.set_current_speaker(wrong, "Wrongname")
    s1 = ctx.ms.log_face_sighting_sync(
        entity_id=wrong, episode_id=ctx.episode_id, embedding=_onehot(0),
        confidence=0.5, source="recognize")
    ctx.handler.deps.face_recognizer = None

    res = await CorrectIdentity()(ctx.handler.deps, clear=True)

    assert res["cleared"] is True
    conn = ctx.ms._connect()
    try:
        row = conn.execute(
            "SELECT entity_id, source FROM face_sightings WHERE id = ?", (s1,)
        ).fetchone()
    finally:
        conn.close()
    assert row["entity_id"] is None
    assert row["source"] == "correct"
    assert ctx.ss.get_current_speaker() is None


@pytest.mark.asyncio
async def test_correct_identity_binds_current_frame_when_recognizer_present(face_ctx):
    """With a recognizer, correct binds the current frame: a correct-sighting + centroid seed."""
    ctx = face_ctx
    wrong = ctx.ms.upsert_entity_sync("Wrongname", kind="person")
    ctx.ms.log_face_sighting_sync(
        entity_id=wrong, episode_id=ctx.episode_id, embedding=_onehot(0),
        confidence=0.5, source="recognize")
    _set_probe(ctx, _onehot(1))  # current frame embedding to bind to the corrected entity

    res = await CorrectIdentity()(ctx.handler.deps, name="Maya")

    assert res["status"] == "corrected"
    maya = _entity_by_name(ctx.ms, "Maya")
    # The relabeled sighting plus a fresh frame-bound correct sighting under Maya.
    maya_sightings = [
        s for s in _all_sightings(ctx.ms)
        if s["entity_id"] == maya["id"] and s["source"] == "correct"
    ]
    assert len(maya_sightings) >= 2
    # Centroid seeded so same-session recognition flips immediately.
    assert any(p["entity_id"] == maya["id"] for p in ctx.ms.get_face_match_set_sync())


@pytest.mark.asyncio
async def test_correct_identity_relabels_whole_episode_single_conversant_assumption(face_ctx):
    """PINS v1 contract: correct relabels EVERY 'recognize' sighting this episode.

    Decision (2026-05-28): correct_identity assumes one conversant per
    correction — it relabels all of the episode's recognize sightings, not just
    the current speaker's. In a rare two-people-share-an-episode case this
    over-relabels the earlier (correctly-recognized) person onto the corrected
    name. Accepted for v1 (Bemo's use is predominantly 1:1); a precise
    turn-timestamp-scoped relabel is a Phase-3 refinement. This test guards the
    current behavior — the Phase-3 fix will deliberately change it.
    """
    ctx = face_ctx
    alice = ctx.ms.upsert_entity_sync("Alice", kind="person")
    # Two distinct people's recognize sightings in the same episode.
    s_alice = ctx.ms.log_face_sighting_sync(
        entity_id=alice, episode_id=ctx.episode_id, embedding=_onehot(0),
        confidence=0.9, source="recognize")
    s_unknown = ctx.ms.log_face_sighting_sync(
        entity_id=None, episode_id=ctx.episode_id, embedding=_onehot(1),
        confidence=0.4, source="recognize")
    ctx.handler.deps.face_recognizer = None  # isolate the relabel path

    res = await CorrectIdentity()(ctx.handler.deps, name="Bob")

    assert res["relabeled"] == 2  # BOTH relabeled — the documented over-reach
    bob = _entity_by_name(ctx.ms, "Bob")
    conn = ctx.ms._connect()
    try:
        rows = conn.execute(
            "SELECT entity_id FROM face_sightings WHERE id IN (?, ?)",
            (s_alice, s_unknown),
        ).fetchall()
    finally:
        conn.close()
    assert all(r["entity_id"] == bob["id"] for r in rows)


@pytest.mark.asyncio
async def test_correct_identity_no_args_errors(face_ctx):
    """Neither a name nor clear → error."""
    ctx = face_ctx

    res = await CorrectIdentity()(ctx.handler.deps, name="  ")

    assert "error" in res


@pytest.mark.asyncio
async def test_correct_identity_no_episode_degrades(face_ctx):
    """No open episode → skip relabel/merge, still re-pin, return corrected with relabeled=0."""
    ctx = face_ctx
    ctx.handler.deps.capture_episode_id = None
    ctx.handler.deps.face_recognizer = None

    res = await CorrectIdentity()(ctx.handler.deps, name="Maya")

    assert res["status"] == "corrected"
    assert res["relabeled"] == 0
    maya = _entity_by_name(ctx.ms, "Maya")
    assert ctx.ss.get_current_speaker() == {"id": maya["id"], "name": "Maya"}


# --- gate ----------------------------------------------------------------


def test_face_tools_gated_on_recognizer():
    """get_active_tool_specs lists the face tools only when a recognizer is present."""
    # Rebuild the registry so the freshly-imported tool classes are included
    # (ALL_TOOL_SPECS is built once at core_tools import, before these load).
    all_tools = {c.name: c() for c in ct.get_concrete_subclasses(ct.Tool)}
    specs = [t.spec() for t in all_tools.values()]
    orig_tools, orig_specs = ct.ALL_TOOLS, ct.ALL_TOOL_SPECS
    ct.ALL_TOOLS, ct.ALL_TOOL_SPECS = all_tools, specs
    try:
        with_rec = ToolDependencies(
            reachy_mini=MagicMock(), movement_manager=MagicMock(),
            face_recognizer=MagicMock(),
        )
        names_with = {s["name"] for s in ct.get_active_tool_specs(with_rec)}
        assert "enroll_face" in names_with
        assert "correct_identity" in names_with

        without_rec = ToolDependencies(
            reachy_mini=MagicMock(), movement_manager=MagicMock(),
        )
        names_without = {s["name"] for s in ct.get_active_tool_specs(without_rec)}
        assert "enroll_face" not in names_without
        assert "correct_identity" not in names_without
    finally:
        ct.ALL_TOOLS, ct.ALL_TOOL_SPECS = orig_tools, orig_specs
