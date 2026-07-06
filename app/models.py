"""Pydantic schemas - LLM structured outputs are validated before side effects."""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ValidationError


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


class EscalationDecision(BaseModel):
    escalate: bool
    reason: str = ""
    category: Literal["billing", "shipping", "returns", "account", "technical", "other"] = "other"
    priority: Literal["low", "medium", "high", "urgent"] = "medium"


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    sentiment: str
    intent: str
    sources: list[str]
    escalated: bool
    ticket_id: str | None = None


def parse_escalation(raw: str) -> EscalationDecision:
    """Parse the LLM's escalation JSON defensively (fences, prose, bad enums)."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return EscalationDecision(escalate=False, reason="unparseable response")
    try:
        return EscalationDecision(**json.loads(match.group(0)))
    except (json.JSONDecodeError, ValidationError):
        return EscalationDecision(escalate=False, reason="invalid escalation JSON")
