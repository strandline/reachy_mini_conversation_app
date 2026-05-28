"""External tool: fix a wrong identity after a misgreet.

Intentional asymmetry vs enroll_face: correct_identity resolves by NAME, not by
face. After Bemo greets someone by the wrong name and they correct her, the
SPOKEN name is ground truth — so this relabels this session's mis-attributed
face sightings to the corrected entity (or clears them), optionally binds the
current frame so same-session recognition flips immediately, and re-pins the
speaker. Do NOT "fix" this to score_profiles.

Like enroll_face, decisions/persistence live in the outer-repo helpers reached
via REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY, imported at CALL time.
"""

from __future__ import annotations
import os
import sys
import asyncio
import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


def _load_stores() -> tuple[Any, Any, Any, Any] | None:
    """Import the outer-repo store modules at call time, or None if unavailable.

    Same contract as enroll_face._load_stores (never caches a load-time None).
    """
    tools_dir = os.environ.get("REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY")
    if tools_dir:
        tools_dir = os.path.abspath(os.path.expanduser(tools_dir))
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
    try:
        import _face_match as fm  # type: ignore[import-not-found]
        import _memory_store as ms  # type: ignore[import-not-found]
        import _capture_store as cs  # type: ignore[import-not-found]
        import _speaker_state as ss  # type: ignore[import-not-found]

        return ms, fm, ss, cs
    except ImportError:
        return None


def _episode_id(deps: ToolDependencies) -> int | None:
    """Return the current open capture episode id, or None."""
    return getattr(deps, "capture_episode_id", None)


class CorrectIdentity(Tool):
    """Fix a wrong name after a misgreet; resolves by the spoken name."""

    name = "correct_identity"
    description = (
        "Fix a wrong name after you greeted someone incorrectly and they "
        "corrected you. Pass the correct name; pass clear=true if it turns out "
        "you do not actually know who they are."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The correct name for the current speaker.",
            },
            "clear": {
                "type": "boolean",
                "description": "Set true if you do not actually know who they are.",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Relabel this session's sightings to the corrected identity (or clear)."""
        name = (kwargs.get("name") or "").strip()
        clear = bool(kwargs.get("clear"))
        if not name and not clear:
            return {"error": "provide a name or set clear=true"}
        stores = _load_stores()
        if stores is None:
            return {"error": "memory store not available"}
        ms, fm, ss, cs = stores

        # Resolve target by NAME — the spoken correction IS ground truth here
        # (intentional asymmetry vs enroll's face-as-anchor; do NOT score_profiles).
        if clear:
            new_entity_id: int | None = None
            final_name: str | None = None
        else:
            new_entity_id = await asyncio.to_thread(ms.upsert_entity_sync, name, kind="person")
            final_name = name

        # Relabel this episode's mis-attributed recognize sightings. Degrade
        # (don't error) when there's no open episode.
        eid = _episode_id(deps)
        relabeled = 0
        if eid is not None:
            sighting_ids = await asyncio.to_thread(
                ms.get_sighting_ids_for_episode_sync, eid
            )
            relabeled = await asyncio.to_thread(
                ms.relabel_sightings_sync, sighting_ids, new_entity_id
            )

        # Bind the current frame to the corrected entity so same-session
        # recognition flips immediately.
        if (
            deps.face_recognizer is not None
            and deps.camera_worker is not None
            and new_entity_id is not None
        ):
            frame = deps.camera_worker.get_latest_frame()
            if frame is not None:
                probe = await asyncio.to_thread(deps.face_recognizer.embed_largest, frame)
                if probe is not None:
                    await asyncio.to_thread(
                        ms.log_face_sighting_sync,
                        entity_id=new_entity_id,
                        episode_id=eid,
                        embedding=probe["embedding"],
                        confidence=None,
                        source="correct",
                    )
                    await asyncio.to_thread(
                        ms.seed_face_centroid_sync, new_entity_id, probe["embedding"]
                    )

        # Re-pin (or clear) the current speaker.
        if new_entity_id is not None:
            await asyncio.to_thread(ss.set_current_speaker, new_entity_id, final_name)
            if deps.session_recognized_ids is not None:
                deps.session_recognized_ids.add(new_entity_id)
            if eid is not None:
                await cs.merge_participant(eid, final_name)
        else:
            await asyncio.to_thread(ss.clear_current_speaker)

        if deps.refresh_session_instructions:
            await deps.refresh_session_instructions()

        return {
            "status": "corrected",
            "name": final_name,
            "relabeled": relabeled,
            "cleared": clear,
        }
