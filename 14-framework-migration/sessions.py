"""
Session storage for multi-turn conversations.

Maps a session_id to a list of OpenAI-format messages (role + content).
In-memory only — restarts wipe history. Swap for Redis/Postgres in prod.
"""

# session_id -> list of {"role": …, "content": …} dicts
_SESSIONS: dict[str, list[dict]] = {}

def get_history(session_id: str) -> list[dict]:
    """Return the message history for a session, or an empty list if new."""
    return _SESSIONS.get(session_id, [])

def append_turn(session_id: str, role: str, content: str) -> None:
    """Append a single message to the session history."""
    _SESSIONS.setdefault(session_id, []).append({"role": role, "content": content})