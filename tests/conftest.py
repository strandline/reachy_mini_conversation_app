"""Pytest configuration for path setup."""

import os
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parents[1].resolve()
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# Make tests reproducible by ignoring machine-specific profile/tool env config.
# Without this, importing config during test collection can pick up a developer's
# local .env and fail before tests run.
os.environ["REACHY_MINI_SKIP_DOTENV"] = "1"
os.environ.pop("REACHY_MINI_CUSTOM_PROFILE", None)
os.environ.pop("REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY", None)
os.environ.pop("REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY", None)

# Backend-readiness checks read provider credentials straight from the env
# (config reads GEMINI_API_KEY *or* GOOGLE_API_KEY, plus OPENAI/INWORLD/HF), so
# a developer's exported keys make "can_proceed_with_<backend>" non-deterministic
# — a test that deletes GEMINI_API_KEY still sees Gemini "ready" via the
# GOOGLE_API_KEY fallback. Pop them all so the suite is reproducible regardless
# of the shell; tests that need a credential set it explicitly via monkeypatch.
for _cred in (
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "INWORLD_API_KEY",
    "HF_TOKEN",
):
    os.environ.pop(_cred, None)


@pytest.fixture(scope="session", autouse=True)
def _isolate_real_memory_db(tmp_path_factory):
    """Never let any test read or write the developer's real ~/.bemo/memory.db.

    Popping REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY (above) makes the base_realtime
    store *globals* load as None, but several code paths import _memory_store
    DIRECTLY (e.g. BaseRealtimeHandler._read_latest_mood). If any test puts the
    outer-repo tools/ dir on sys.path — the face-ID integration harness in
    test_face_id.py does, for its temp-DB round-trips — that direct import then
    succeeds and would otherwise hit the real memory DB, leaking live data into
    assertions (and risking writes). Redirect DB_PATH to a session-scoped
    throwaway whenever _memory_store is importable; no-op (and the real DB stays
    unreachable) when it isn't. Function-scoped fixtures that want their own DB
    (face_ctx) still monkeypatch DB_PATH on top of this and revert to it.
    """
    try:
        import _memory_store
    except ImportError:
        yield
        return
    original_path = _memory_store.DB_PATH
    original_initialized = _memory_store._initialized
    _memory_store.DB_PATH = tmp_path_factory.mktemp("bemo_isolated") / "memory.db"
    _memory_store._initialized = False
    try:
        yield
    finally:
        _memory_store.DB_PATH = original_path
        _memory_store._initialized = original_initialized
