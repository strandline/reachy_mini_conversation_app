"""External tool: enroll the face of the person Bemo is talking with.

Face-as-anchor (Phase 2): identity is keyed by the FACE, not the spoken name.
A confident match to an existing face → reuse that entity (the name just adds
an alias); a new face whose name collides with someone else → ask for a
distinguisher rather than collapsing two people; otherwise create a new entity.

This tool lives in the conv-app submodule but delegates every decision to the
outer-repo pure helpers (_face_match) and persistence (_memory_store /
_capture_store / _speaker_state), reached at runtime via
REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY. The store import is done at CALL time
(``_load_stores``) so this module imports cleanly — and stays discoverable by
the tool loader — even when the outer tools/ dir is absent, and a transient
missing-dir is never cached as a permanent None.
"""

from __future__ import annotations
import os
import sys
import asyncio
import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

_FACE_HIGH = float(os.environ.get("REACHY_MINI_FACE_HIGH_THRESHOLD", "0.45"))


def _load_stores() -> tuple[Any, Any, Any, Any] | None:
    """Import the outer-repo store modules at call time, or None if unavailable.

    Mirrors base_realtime's loaders: honor REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY
    for the sys.path insert, then import. Never caches a load-time None — callers
    re-invoke this each call so a dir that appears later still works.
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


class EnrollFace(Tool):
    """Remember the face of the person Bemo is talking with (consent-gated)."""

    name = "enroll_face"
    fire_and_forget = True  # side-effect write; model doesn't need the result
    description = (
        "Remember the face of the person you are talking with, after they have "
        "agreed, so you recognize them next time. Pass the name they gave you."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The name the person gave you.",
            },
        },
        "required": ["name"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Enroll the current face under `name` (face-as-anchor)."""
        name = (kwargs.get("name") or "").strip()
        if not name:
            return {"error": "name must be a non-empty string"}
        if deps.face_recognizer is None:
            return {"error": "face recognition not available"}
        if deps.camera_worker is None:
            return {"error": "Camera worker not available"}
        stores = _load_stores()
        if stores is None:
            return {"error": "memory store not available"}
        ms, fm, ss, cs = stores

        frame = deps.camera_worker.get_latest_frame()
        if frame is None:
            return {"error": "No frame available"}
        probe = await asyncio.to_thread(deps.face_recognizer.embed_largest, frame)
        if probe is None:
            return {"status": "no_face", "message": "I cannot see a face right now"}
        emb = probe["embedding"]

        match_set = await asyncio.to_thread(ms.get_face_match_set_sync)
        scored = fm.score_profiles(emb, match_set)
        all_entities = await asyncio.to_thread(ms.list_all_entities_sync)
        # existing_names must include ALIASES (lowercased): upsert_entity_sync is
        # alias-aware, so a names-only set would let a spoken alias slip through as
        # 'new' and silently collapse two faces onto one aliased entity.
        existing_names = {
            n.lower()
            for e in all_entities
            for n in ([e["name"]] + (e.get("aliases") or []))
        }
        # faced_names = names/aliases of entities that ALREADY own a face profile.
        # Only a clash with one of these is a real collision (two distinct faces);
        # a clash with a FACELESS name (e.g. someone known only from text-only
        # memory) is reuse_by_name — attach the first face to that existing entity.
        # WITHOUT this, name_has_face is always False and collision never fires.
        faced_ids = {m["entity_id"] for m in match_set}
        faced_names = {
            n.lower()
            for e in all_entities
            if e["id"] in faced_ids
            for n in ([e["name"]] + (e.get("aliases") or []))
        }
        res = fm.resolve_enrollment(
            scored,
            spoken_name=name,
            existing_names=existing_names,
            faced_names=faced_names,
            high=_FACE_HIGH,
        )
        action = res["action"]

        if action == "collision":
            # Face-as-anchor: never collapse two faces. Bemo asks for a
            # distinguisher per the REMEMBERING instructions, then re-calls.
            return {
                "status": "collision",
                "message": f"I think I already know a {name}.",
                "spoken_name": name,
            }

        if action == "reuse":
            entity_id = res["entity_id"]
            final_name = res["name"]  # canonical
            if name.lower() != (final_name or "").lower():
                await asyncio.to_thread(ms.add_alias_sync, final_name, name, kind="person")
            await asyncio.to_thread(ms.seed_face_centroid_sync, entity_id, emb)
        else:  # 'new' or 'reuse_by_name'
            # upsert_entity_sync is get-or-create: for 'new' it creates a fresh
            # entity; for 'reuse_by_name' it returns the existing FACELESS entity
            # (e.g. one known only from text-only memory), unifying the face anchor
            # with that dossier-bearing record. Either way, seed the first centroid.
            entity_id = await asyncio.to_thread(ms.upsert_entity_sync, name, kind="person")
            await asyncio.to_thread(ms.seed_face_centroid_sync, entity_id, emb)
            final_name = name

        await asyncio.to_thread(
            ms.log_face_sighting_sync,
            entity_id=entity_id,
            episode_id=_episode_id(deps),
            embedding=emb,
            confidence=res.get("score"),
            source="enroll",
        )
        await asyncio.to_thread(ss.set_current_speaker, entity_id, final_name)
        eid = _episode_id(deps)
        if eid is not None:
            await cs.merge_participant(eid, final_name)
        # Explicit consent → seed session continuity so a later gray-zone match
        # of this face self-corroborates (unlike the recognizer's gray branch).
        if deps.session_recognized_ids is not None:
            deps.session_recognized_ids.add(entity_id)
        if deps.refresh_session_instructions:
            await deps.refresh_session_instructions()

        return {
            "status": "enrolled",
            "action": action,
            "name": final_name,
            "entity_id": entity_id,
        }
