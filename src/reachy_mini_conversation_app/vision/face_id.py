"""InsightFace face-recognition wrapper for the realtime face-ID slice.

Mirrors ``vision/local_vision.py``'s "load a model once, run on a trigger"
pattern, but for InsightFace ``buffalo_l`` (RetinaFace detect + ArcFace embed)
via onnxruntime. This file is **pure inference**: it has no database access and
does NOT import ``_face_match`` — every identity *decision* stays in the
outer-repo pure layer (``tools/_face_match.py``). The recognizer
(``base_realtime._run_face_recognition``) and the enroll/correct tools call
``embed_largest`` for the embedding and then drive the decision logic.

The ``insightface`` import is deferred into ``FaceRecognizer.load()`` so this
module (and the DI tests) import cleanly without the package present. Install
the inference deps with ``pip install '.[face_id]'``.

Execution-provider portability: ``DEFAULT_PROVIDERS`` is the macOS dev order
(CoreML ANE → CPU fallback, verified active on all ``buffalo_l`` models). The
eventual Jetson Orin deployment passes ``providers=`` explicitly
(TensorRT/CUDA) — the ONNX model is identical, only the provider list changes.
"""

from __future__ import annotations
import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)

DEFAULT_PROVIDERS: list[str] = ["CoreMLExecutionProvider", "CPUExecutionProvider"]


def _bbox_area(bbox: Any) -> float:
    """Area of an ``[x1, y1, x2, y2]`` box (clamped non-negative)."""
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


class FaceRecognizer:
    """Load-once InsightFace wrapper: detect the largest face → unit embedding.

    The ``app`` argument is an injectable seam: when provided (tests), ``load()``
    is a no-op and the real ``insightface`` import is never touched.
    """

    def __init__(
        self, providers: list[str] | None = None, app: Any | None = None
    ) -> None:
        """Initialize the recognizer (does not load the model — call ``load``).

        Args:
            providers: onnxruntime execution providers, highest priority first.
                Defaults to ``DEFAULT_PROVIDERS`` (CoreML → CPU).
            app: A pre-built / fake ``FaceAnalysis``-like object. When given,
                ``load()`` returns early and no model is loaded.

        """
        self.providers = providers or DEFAULT_PROVIDERS
        self.app = app
        self._initialized = app is not None

    def load(self) -> None:
        """Build and prepare the InsightFace ``buffalo_l`` pipeline.

        No-op when an ``app`` was injected. Otherwise defers the ``insightface``
        import to here and lets ``ImportError`` propagate (the factory catches
        it). Mirrors ``VisionProcessor.initialize``.
        """
        if self.app is not None:
            return
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(name="buffalo_l", providers=self.providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self._initialized = True
        logger.info("FaceRecognizer loaded (buffalo_l) providers=%s", self.providers)

    def embed_largest(
        self, frame_bgr: NDArray[np.uint8]
    ) -> dict[str, Any] | None:
        """Detect the largest face in a BGR frame; return its unit embedding.

        Args:
            frame_bgr: A BGR frame, as produced by
                ``camera_worker.get_latest_frame()``. Fed straight to
                ``app.get()`` — InsightFace expects BGR, so do NOT flip to RGB.

        Returns:
            ``{'embedding': (512,) L2-normalized float32, 'bbox': float32[4],
            'det_score': float}`` for the largest-area detection, or ``None``
            when no face is found.

        """
        faces = self.app.get(frame_bgr)
        if not faces:
            return None
        face = max(faces, key=lambda f: _bbox_area(f.bbox))
        emb = np.asarray(face.embedding, dtype=np.float32)
        emb = emb / float(np.linalg.norm(emb))
        return {
            "embedding": emb,
            "bbox": np.asarray(face.bbox, dtype=np.float32),
            "det_score": float(face.det_score),
        }


def initialize_face_recognizer(
    providers: list[str] | None = None,
) -> FaceRecognizer | None:
    """Build a loaded ``FaceRecognizer``, or ``None`` if insightface is absent.

    Deliberately diverges from ``initialize_vision_processor`` (which re-raises):
    the guard lives *inside* this factory and returns ``None`` so the
    camera-present auto-enable path degrades silently (Decision 3) and face-ID
    never blocks startup (Decision 4). Do not "fix" this to re-raise.
    """
    try:
        recognizer = FaceRecognizer(providers=providers)
        recognizer.load()
        logger.info("Face recognition enabled (providers=%s)", recognizer.providers)
        return recognizer
    except ImportError:
        logger.warning(
            "insightface not installed; face recognition disabled. "
            "Install with: pip install '.[face_id]'"
        )
        return None
