"""Phase 2 face-ID tests — DI-mocked; no real InsightFace model, no camera.

Part B covers ``vision/face_id.py`` (``FaceRecognizer`` + ``embed_largest`` +
``initialize_face_recognizer``). Parts C and D extend this same module. The
fake ``app``/face objects stand in for insightface so these run without the
package installed and without a webcam.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

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
