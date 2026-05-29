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

_FACE_HIGH = float(os.environ.get("REACHY_MINI_FACE_HIGH_THRESHOLD", "0.45"))


def _load_stores() -> tuple[Any, Any, Any, Any, Any] | None:
    """Import the outer-repo store modules at call time, or None if unavailable.

    Same contract as enroll_face._load_stores (never caches a load-time None).
    Returns (ms, fm, ss, cs, ir) — ir is _identity_route, for the merge gate's
    name-consistency check.
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
        import _identity_route as ir  # type: ignore[import-not-found]

        return ms, fm, ss, cs, ir
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
        "you do not actually know who they are. If this returns "
        "status='merge_candidate', you already have this person saved twice — "
        "ask them whether to merge the two records, and only if they agree, call "
        "this again with the same name and confirm_merge=true."
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
            "confirm_merge": {
                "type": "boolean",
                "description": (
                    "Set true ONLY after a prior call returned "
                    "status='merge_candidate' and the person agreed to merge "
                    "their two records into one."
                ),
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Relabel this session's sightings to the corrected identity (or clear)."""
        name = (kwargs.get("name") or "").strip()
        clear = bool(kwargs.get("clear"))
        confirm_merge = bool(kwargs.get("confirm_merge"))
        if not name and not clear:
            return {"error": "provide a name or set clear=true"}
        stores = _load_stores()
        if stores is None:
            return {"error": "memory store not available"}
        ms, fm, ss, cs, ir = stores

        # Capture the currently-pinned (mis-greeted) entity BEFORE re-pinning.
        # The recognizer's high-tier match added it to session_recognized_ids;
        # without dropping it the recognizer could re-corroborate the wrong
        # identity from "session continuity" on the next gray-zone scan and undo
        # this correction (true for clear=true too).
        prev = await asyncio.to_thread(ss.get_current_speaker)
        prev_id = prev["id"] if prev else None

        # --- merge gate ---------------------------------------------------
        # When the correction points at a DIFFERENT existing entity that is
        # plausibly the same person (name-consistent) and the live face confirms
        # it (>= HIGH on a faced target, or a faceless text-only dossier), the
        # two records are a fragment of one identity. Propose a merge first;
        # only fold on explicit confirm_merge. Anything else falls through to the
        # plain relabel below — a cross-person misgreet (name-inconsistent) is
        # never merged, so no cross-person link is asserted.
        if name and not clear and prev_id is not None:
            decision = await self._merge_gate(deps, ms, fm, ss, cs, ir,
                                              prev_id, name, confirm_merge)
            if decision is not None:
                return decision

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
        #
        # v1 assumption (single conversant per correction): relabels ALL of the
        # episode's recognize sightings (entity_id=None below), not just the
        # current speaker's. Correct for Bemo's predominantly 1:1 use; in the
        # rare two-people-share-an-episode case it over-relabels the earlier
        # (correctly-recognized) person onto the corrected name. A precise
        # turn-timestamp-scoped relabel is a Phase-3 refinement. The helper's
        # entity_id= filter exists for that future scoping.
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

        # Drop the stale (wrong) identity from session continuity first, then
        # re-pin (or clear). discard() is a no-op when prev_id is None or the
        # corrected name resolves to the same entity (re-added below).
        if deps.session_recognized_ids is not None and prev_id is not None:
            deps.session_recognized_ids.discard(prev_id)
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

    async def _merge_gate(
        self, deps: ToolDependencies, ms: Any, fm: Any, ss: Any, cs: Any, ir: Any,
        prev_id: int, name: str, confirm_merge: bool,
    ) -> Dict[str, Any] | None:
        """Propose or execute a same-person merge, or return None to fall through.

        None → not a merge candidate; the caller does the plain relabel. A merge
        is only ever proposed for a DIFFERENT existing entity whose name is
        consistent with the source (so a cross-person misgreet is never merged),
        confirmed by the live face (>= HIGH on a faced target) or, for a faceless
        text-only dossier, a light confirm.
        """
        target_row = await asyncio.to_thread(ms.resolve_entity_sync, name)
        if target_row is None or target_row["id"] == prev_id:
            return None  # a new name, or the same entity → not a merge
        source = await asyncio.to_thread(ms.get_entity_sync, prev_id)
        target = await asyncio.to_thread(ms.get_entity_sync, target_row["id"])
        if source is None or target is None:
            return None
        # #5: only merge plausibly-same-person names; otherwise fall through to a
        # plain relabel so no cross-person identity link is ever asserted.
        if not ir.names_consistent(
            source["name"], target["name"],
            aliases_a=source["aliases"], aliases_b=target["aliases"],
        ):
            return None
        if target["has_face"]:
            score = await self._live_score_against(deps, fm, ms, target["id"])
            if score is None or score < _FACE_HIGH:
                return None
            faceless = False
        else:
            faceless = True  # no second face to compare → light confirm

        if not confirm_merge:
            summary = await asyncio.to_thread(
                ms.merge_entities_sync, source["id"], target["id"], dry_run=True)
            return {
                "status": "merge_candidate",
                "source": source["name"],
                "target": target["name"],
                "faceless": faceless,
                "summary": summary,
            }

        # Confirmed → fold source into target, drop the stale id from continuity,
        # re-pin the survivor.
        summary = await asyncio.to_thread(
            ms.merge_entities_sync, source["id"], target["id"])
        if deps.session_recognized_ids is not None:
            deps.session_recognized_ids.discard(source["id"])
            deps.session_recognized_ids.add(target["id"])
        await asyncio.to_thread(ss.set_current_speaker, target["id"], target["name"])
        eid = _episode_id(deps)
        if eid is not None:
            await cs.merge_participant(eid, target["name"])
        if deps.refresh_session_instructions:
            await deps.refresh_session_instructions()
        return {"status": "merged", "name": target["name"], "summary": summary}

    async def _live_score_against(
        self, deps: ToolDependencies, fm: Any, ms: Any, entity_id: int
    ) -> float | None:
        """Top cosine of the current frame against `entity_id`'s gallery, or None.

        None when there's no recognizer/camera/frame/face or the entity has no
        gallery — the caller treats that as "can't confirm" → no merge.
        """
        if deps.face_recognizer is None or deps.camera_worker is None:
            return None
        frame = deps.camera_worker.get_latest_frame()
        if frame is None:
            return None
        probe = await asyncio.to_thread(deps.face_recognizer.embed_largest, frame)
        if probe is None:
            return None
        match_set = await asyncio.to_thread(ms.get_face_match_set_sync)
        target_set = [m for m in match_set if m["entity_id"] == entity_id]
        if not target_set:
            return None
        scored = fm.score_profiles(probe["embedding"], target_set)
        return scored[0]["score"] if scored else None
