from __future__ import annotations
import sys
import logging
import argparse
import warnings
import subprocess
from typing import TYPE_CHECKING, Any, Optional

from reachy_mini import ReachyMini
from reachy_mini_conversation_app.camera_worker import CameraWorker
from reachy_mini_conversation_app.vision.head_tracking import HeadTracker


if TYPE_CHECKING:
    from reachy_mini_conversation_app.vision.local_vision import VisionProcessor


class CameraVisionInitializationError(Exception):
    """Raised when camera or vision setup fails in an expected way."""


def parse_args() -> tuple[argparse.Namespace, list]:  # type: ignore
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("Reachy Mini Conversation App")
    parser.add_argument(
        "--head-tracker",
        choices=["yolo", "mediapipe"],
        default=None,
        help=(
            "Optional head-tracking backend: yolo uses a local face detector in a subprocess, "
            "mediapipe uses reachy_mini_toolbox in process. Disabled by default."
        ),
    )
    parser.add_argument("--no-camera", default=False, action="store_true", help="Disable camera usage")
    parser.add_argument(
        "--local-vision",
        default=False,
        action="store_true",
        help="Use local vision model instead of the selected realtime backend vision",
    )
    parser.add_argument("--gradio", default=False, action="store_true", help="Open gradio interface")
    parser.add_argument("--debug", default=False, action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--robot-name",
        type=str,
        default=None,
        help="[Optional] Robot name to target. Must match the daemon's --robot-name when connecting to a specific robot, mainly useful for development with multiple robots.",
    )
    return parser.parse_known_args()


def initialize_camera_and_vision(
    args: argparse.Namespace,
    current_robot: ReachyMini,
) -> tuple[CameraWorker | None, VisionProcessor | None]:
    """Initialize camera capture, optional head tracking, and optional local vision."""
    camera_worker: Optional[CameraWorker] = None
    head_tracker: HeadTracker | None = None
    vision_processor: Optional[VisionProcessor] = None

    if not args.no_camera:
        if args.head_tracker is not None:
            try:
                if args.head_tracker == "yolo":
                    from reachy_mini_conversation_app.vision.head_tracking.yolo_process import (
                        YoloHeadTrackerProcess,
                    )

                    head_tracker = YoloHeadTrackerProcess()
                    logging.getLogger(__name__).info("Using yolo head tracker subprocess")
                else:
                    from reachy_mini_conversation_app.vision.head_tracking.mediapipe import (
                        MediapipeHeadTracker,
                    )

                    head_tracker = MediapipeHeadTracker()
                    logging.getLogger(__name__).info("Using mediapipe head tracker in process")
            except Exception as e:
                raise CameraVisionInitializationError(
                    f"Failed to initialize {args.head_tracker} head tracker: {e}",
                ) from e

        camera_worker = CameraWorker(current_robot, head_tracker)

        if args.local_vision:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from reachy_mini_conversation_app.vision.local_vision import VisionProcessor",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode < 0:
                raise CameraVisionInitializationError(
                    "Local vision import crashed on this machine. "
                    "Run without --local-vision or install compatible dependencies.",
                )
            try:
                from reachy_mini_conversation_app.vision.local_vision import initialize_vision_processor

            except ImportError as e:
                raise CameraVisionInitializationError(
                    "To use --local-vision, please install the extra dependencies: pip install '.[local_vision]'",
                ) from e

            vision_processor = initialize_vision_processor()
        else:
            logging.getLogger(__name__).info(
                "Using the selected realtime backend for vision (default). Use --local-vision for local processing.",
            )

    return camera_worker, vision_processor


def initialize_face_recognizer(
    args: argparse.Namespace, camera_worker: CameraWorker | None
) -> Any | None:
    """Build a FaceRecognizer when a camera is present, else None.

    Realtime face-ID (Phase 2) is auto-enabled with NO flag (Decision 3) and
    must NEVER block startup (Decision 4). So, unlike the ``--local-vision``
    path (which raises CameraVisionInitializationError → sys.exit), this gates
    on the already-built ``camera_worker`` and degrades to ``None`` on ANY
    failure — insightface absent (logged with an install hint) or a
    model/provider init error (logged as a warning). It never raises.

    Args:
        args: Parsed CLI args (reserved for future provider overrides).
        camera_worker: The already-built camera worker, or None when the
            camera is disabled. Face-ID requires a camera, so None → None.

    Returns:
        A loaded ``vision.face_id.FaceRecognizer``, or None when face-ID is
        unavailable for any reason.

    """
    if camera_worker is None:
        return None
    log = logging.getLogger(__name__)
    try:
        # Alias on import to disambiguate this gate-wrapper from the vision
        # module's same-named loader. Deferred so a missing insightface only
        # matters when a camera is actually present.
        from reachy_mini_conversation_app.vision.face_id import (
            initialize_face_recognizer as _load_recognizer,
        )

        return _load_recognizer()
    except ImportError:
        log.info(
            "Face recognition unavailable; to enable it install the extras: "
            "pip install '.[face_id]'",
        )
        return None
    except Exception:
        log.warning(
            "Face recognizer initialization failed; continuing without face-ID",
            exc_info=True,
        )
        return None


def setup_logger(debug: bool) -> logging.Logger:
    """Setups the logger."""
    log_level = "DEBUG" if debug else "INFO"
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s:%(lineno)d | %(message)s",
        force=True,
    )
    logger = logging.getLogger(__name__)

    # Suppress WebRTC warnings
    warnings.filterwarnings("ignore", message=".*AVCaptureDeviceTypeExternal.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="aiortc")

    # Tame third-party noise (looser in DEBUG)
    if log_level == "DEBUG":
        logging.getLogger("aiortc").setLevel(logging.INFO)
        logging.getLogger("fastrtc").setLevel(logging.INFO)
        logging.getLogger("aioice").setLevel(logging.INFO)
        logging.getLogger("openai").setLevel(logging.INFO)
        logging.getLogger("websockets").setLevel(logging.INFO)
    else:
        logging.getLogger("aiortc").setLevel(logging.ERROR)
        logging.getLogger("fastrtc").setLevel(logging.ERROR)
        logging.getLogger("aioice").setLevel(logging.WARNING)
    return logger


def log_connection_troubleshooting(logger: logging.Logger, robot_name: Optional[str]) -> None:
    """Log troubleshooting steps for connection issues."""
    logger.error("Troubleshooting steps:")
    logger.error("  1. Verify reachy-mini-daemon is running")

    if robot_name is not None:
        logger.error(f"  2. Daemon must be started with: --robot-name '{robot_name}'")
    else:
        logger.error("  2. If daemon uses --robot-name, add the same flag here: --robot-name <name>")

    logger.error("  3. For wireless: check network connectivity")
    logger.error("  4. Review daemon logs")
    logger.error("  5. Restart the daemon")
