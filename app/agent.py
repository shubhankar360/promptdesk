"""Agent orchestrator: sentiment -> intent -> RAG -> answer -> escalation.

This is the pipeline a single /api/chat call flows through. Each stage is a
separate prompt so behavior is observable and testable stage by stage.
"""

from __future__ import annotations

from app import prompts
from app.database import Database
from app.llm import LLMClient
from app.models import ChatResponse, parse_escalation
from app.rag import Retriever


class SupportAgent:
    def __init__(self, llm: LLMClient, retriever: Retriever, db: Database) -> None:
        self.llm = llm
        self.retriever = retriever
        self.db = db

    def handle(self, message: str, conversation_id: str | None) -> ChatResponse:
        cid = self.db.ensure_conversation(conversation_id)

        # 1. Chain-of-thought sentiment
        raw_sentiment = self.llm.complete(
            system="You are a precise sentiment analysis engine.",
            prompt=prompts.build_sentiment_prompt(message),
            max_tokens=250,
        )
        sentiment = prompts.parse_sentiment(raw_sentiment)

        # 2. Few-shot intent classification
        raw_intent = self.llm.complete(
            system="You are a precise intent classifier. Reply with one word.",
            prompt=prompts.build_intent_prompt(message),
            max_tokens=10,
        )
        intent = next((l for l in prompts.INTENT_LABELS if l in raw_intent.lower()), "other")

        self.db.add_message(cid, "user", message, sentiment=sentiment, intent=intent)

        # 3. Retrieve grounding passages
        passages = self.retriever.search(message, top_k=3)
        sources = list(dict.fromkeys(p["title"] for p in passages))

        # 4. Generate the answer with sentiment-adapted tone
        history = self.db.history(cid, limit=10)
        history_text = "\n".join(f'{m["role"]}: {m["content"]}' for m in history[:-1])
        prompt = prompts.build_answer_prompt(message, passages)
        if history_text:
            prompt = f"Conversation so far:\n{history_text}\n\n{prompt}"
        reply = self.llm.complete(
            system=prompts.build_system_prompt(sentiment),
            prompt=prompt,
            max_tokens=600,
        )

        # 5. Structured escalation decision (validated before side effects)
        streak = self.db.negative_streak(cid)
        raw_decision = self.llm.complete(
            system="You are an escalation controller. Output only JSON.",
            prompt=prompts.build_escalation_prompt(message, sentiment, streak, reply),
            max_tokens=200,
        )
        decision = parse_escalation(raw_decision)

        ticket_id = None
        if decision.escalate:
            ticket_id = self.db.create_ticket(cid, decision.reason, decision.category, decision.priority)
            reply += (
                f"\n\nI've raised this with our support team - your ticket is "
                f"{ticket_id} ({decision.priority} priority). A human agent will follow up shortly."
            )

        self.db.add_message(cid, "assistant", reply)

        return ChatResponse(
            conversation_id=cid,
            reply=reply,
            sentiment=sentiment,
            intent=intent,
            sources=sources,
            escalated=decision.escalate,
            ticket_id=ticket_id,
        )
