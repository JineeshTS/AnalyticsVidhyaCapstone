"""
Multi-user conversational memory (stretch goal).

A small, transparent SQLite layer that stores chat turns keyed by session_id,
so each user/session keeps an isolated history. Kept dependency-light on
purpose -- no ORM, just sqlite3 -- so it is easy to audit and runs anywhere.
"""

import sqlite3
from contextlib import contextmanager
from typing import List, Tuple

import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role       TEXT NOT NULL,          -- 'user' or 'assistant'
    content    TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_session ON messages(session_id);
"""


@contextmanager
def _conn():
    conn = sqlite3.connect(str(config.SQLITE_PATH))
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as conn:
        conn.executescript(_SCHEMA)


def add_message(session_id: str, role: str, content: str) -> None:
    init_db()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )


def get_history(session_id: str, limit: int = 20) -> List[Tuple[str, str]]:
    """Return the last `limit` (role, content) turns for a session, oldest first."""
    init_db()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return list(reversed(rows))


def history_as_text(session_id: str, limit: int = 6) -> str:
    """Render recent history as a plain transcript for prompt conditioning."""
    turns = get_history(session_id, limit)
    if not turns:
        return ""
    lines = [f"{role.capitalize()}: {content}" for role, content in turns]
    return "\n".join(lines)


def clear_session(session_id: str) -> None:
    init_db()
    with _conn() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
