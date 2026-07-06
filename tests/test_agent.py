"""End-to-end agent pipeline test in keyless demo mode (no API calls)."""

import os
import tempfile

from app.agent import SupportAgent
from app.database import Database
from app.llm import LLMClient
from app.rag import Retriever


def make_agent(tmp_path):
    db = Database(path=os.path.join(tmp_path, "test.db"))
    retriever = Retriever()
    retriever.index_texts([
        ("Returns Policy", "# Returns\n\n## Return window\nItems can be returned within 30 days for a full refund."),
    ])
    return SupportAgent(LLMClient(), retriever, db), db


def test_happy_path_answers_with_source():
    with tempfile.TemporaryDirectory() as tmp:
        agent, _ = make_agent(tmp)
        resp = agent.handle("What is your return window?", None)
        assert resp.conversation_id
        assert resp.reply
        assert "Returns Policy" in resp.sources


def test_angry_refund_request_creates_ticket():
    with tempfile.TemporaryDirectory() as tmp:
        agent, db = make_agent(tmp)
        resp = agent.handle("This is unacceptable! I demand a refund right now!", None)
        assert resp.sentiment in ("angry", "frustrated")
        assert resp.escalated is True
        assert resp.ticket_id is not None
        assert db.list_tickets()[0]["id"] == resp.ticket_id


def test_conversation_persists_across_turns():
    with tempfile.TemporaryDirectory() as tmp:
        agent, db = make_agent(tmp)
        first = agent.handle("What is your return window?", None)
        second = agent.handle("Thanks, that helps!", first.conversation_id)
        assert second.conversation_id == first.conversation_id
        history = db.history(first.conversation_id)
        assert len(history) == 4  # 2 user + 2 assistant


def test_analytics_counts():
    with tempfile.TemporaryDirectory() as tmp:
        agent, db = make_agent(tmp)
        agent.handle("What is your return window?", None)
        stats = db.analytics()
        assert stats["conversations"] == 1
        assert stats["user_messages"] == 1
