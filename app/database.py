"""SQLite persistence: conversations, messages, tickets, analytics."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone

from app import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sentiment TEXT,
    intent TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tickets (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    category TEXT NOT NULL,
    priority TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: str | None = None) -> None:
        self.path = path or config.DATABASE_PATH
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    # -- conversations -------------------------------------------------

    def ensure_conversation(self, conversation_id: str | None) -> str:
        cid = conversation_id or uuid.uuid4().hex[:12]
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO conversations (id, started_at) VALUES (?, ?)",
                (cid, _now()),
            )
        return cid

    def add_message(self, conversation_id: str, role: str, content: str,
                    sentiment: str | None = None, intent: str | None = None) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, sentiment, intent, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (conversation_id, role, content, sentiment, intent, _now()),
            )

    def history(self, conversation_id: str, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT role, content, sentiment FROM messages "
                "WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def negative_streak(self, conversation_id: str) -> int:
        """Consecutive most-recent user messages with negative sentiment."""
        streak = 0
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT sentiment FROM messages WHERE conversation_id = ? AND role = 'user' "
                "ORDER BY id DESC LIMIT 10",
                (conversation_id,),
            ).fetchall()
        for row in rows:
            if row["sentiment"] in ("frustrated", "angry"):
                streak += 1
            else:
                break
        return streak

    # -- tickets ---------------------------------------------------------

    def create_ticket(self, conversation_id: str, reason: str, category: str, priority: str) -> str:
        ticket_id = "TCK-" + uuid.uuid4().hex[:8].upper()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO tickets (id, conversation_id, reason, category, priority, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ticket_id, conversation_id, reason, category, priority, _now()),
            )
        return ticket_id

    def list_tickets(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM tickets ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    # -- analytics -------------------------------------------------------

    def analytics(self) -> dict:
        with self._conn() as conn:
            conversations = conn.execute("SELECT COUNT(*) c FROM conversations").fetchone()["c"]
            messages = conn.execute(
                "SELECT COUNT(*) c FROM messages WHERE role = 'user'"
            ).fetchone()["c"]
            tickets = conn.execute("SELECT COUNT(*) c FROM tickets").fetchone()["c"]
            sentiment_rows = conn.execute(
                "SELECT sentiment, COUNT(*) c FROM messages "
                "WHERE role = 'user' AND sentiment IS NOT NULL GROUP BY sentiment"
            ).fetchall()
        sentiments = {r["sentiment"]: r["c"] for r in sentiment_rows}
        escalation_rate = round(tickets / messages * 100, 1) if messages else 0.0
        return {
            "conversations": conversations,
            "user_messages": messages,
            "tickets": tickets,
            "escalation_rate_pct": escalation_rate,
            "sentiment_breakdown": sentiments,
        }
