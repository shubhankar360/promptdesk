"""PromptDesk prompt library.

Every prompt used by the agent lives here, composed from small, testable
blocks. Techniques demonstrated:

1. Layered system prompts   (role + guardrails + tone + output rules)
2. Few-shot classification  (intent examples)
3. Chain-of-thought         (sentiment reasoning before labeling)
4. Structured JSON output   (escalation decision schema)
5. Context injection + citations (RAG grounding)
6. Dynamic tone adaptation  (sentiment rewrites the tone block)
"""

from __future__ import annotations

COMPANY_NAME = "CloudCart"

# ---------------------------------------------------------------------------
# 1. Layered system prompt blocks
# ---------------------------------------------------------------------------

ROLE_BLOCK = f"""\
You are Aria, the customer support assistant for {COMPANY_NAME}, an
e-commerce platform for small businesses. You are knowledgeable, concise,
and warm. You resolve issues on first contact whenever possible."""

GUARDRAILS_BLOCK = """\
Rules you must always follow:
- Only answer using the knowledge base context provided. If the context
  does not contain the answer, say so honestly and offer to escalate.
- Never invent order numbers, prices, dates, or policy details.
- Never ask for passwords, full card numbers, or other sensitive data.
- Never promise refunds or exceptions to policy; only a human agent can.
- Keep answers under 150 words unless the customer asks for detail."""

CITATION_BLOCK = """\
When you use information from the knowledge base context, mention the
source document naturally, e.g. "According to our Returns Policy, ...".
If multiple documents apply, cite each one once."""

TONE_BLOCKS = {
    "positive": "The customer is in a good mood. Match their energy: be friendly and efficient.",
    "neutral": "Keep a professional, helpful, warm tone.",
    "frustrated": (
        "The customer is frustrated. Open by briefly acknowledging the "
        "inconvenience (one sentence, no groveling), then get straight to "
        "the most concrete next step you can offer. Avoid corporate filler "
        'phrases like "we apologize for any inconvenience".'
    ),
    "angry": (
        "The customer is angry. Do NOT be defensive. Acknowledge the problem "
        "plainly, take ownership on behalf of the company, give the single "
        "most useful action available, and proactively offer escalation to a "
        "human agent. Be brief - long replies read as stalling."
    ),
}


def build_system_prompt(sentiment: str = "neutral") -> str:
    """Compose the layered system prompt for the answer-generation call."""
    tone = TONE_BLOCKS.get(sentiment, TONE_BLOCKS["neutral"])
    return "\n\n".join([ROLE_BLOCK, GUARDRAILS_BLOCK, CITATION_BLOCK, f"Tone guidance:\n{tone}"])


# ---------------------------------------------------------------------------
# RAG context injection
# ---------------------------------------------------------------------------

def build_context_block(passages: list[dict]) -> str:
    """Wrap retrieved passages in tagged blocks the model can cite."""
    if not passages:
        return (
            "<knowledge_base>\n(No relevant documents were found for this "
            "question.)\n</knowledge_base>"
        )
    chunks = []
    for p in passages:
        chunks.append(
            f'<document title="{p["title"]}" relevance="{p["score"]:.2f}">\n'
            f'{p["text"]}\n</document>'
        )
    return "<knowledge_base>\n" + "\n\n".join(chunks) + "\n</knowledge_base>"


def build_answer_prompt(question: str, passages: list[dict]) -> str:
    return (
        f"{build_context_block(passages)}\n\n"
        f"Customer message:\n{question}\n\n"
        "Answer the customer using only the knowledge base above."
    )


# ---------------------------------------------------------------------------
# 2. Few-shot intent classification
# ---------------------------------------------------------------------------

INTENT_LABELS = ["question", "complaint", "request", "chitchat", "other"]

INTENT_PROMPT = """\
Classify the customer's message into exactly one intent label:
question | complaint | request | chitchat | other

Examples:
Message: "How long does standard shipping take?"
Intent: question

Message: "This is the third time my order arrived damaged. Unacceptable."
Intent: complaint

Message: "Please cancel order #4482 and refund my card."
Intent: request

Message: "Thanks so much, you've been great!"
Intent: chitchat

Message: "{message}"
Intent:"""


def build_intent_prompt(message: str) -> str:
    return INTENT_PROMPT.format(message=message.replace('"', "'"))


# ---------------------------------------------------------------------------
# 3. Chain-of-thought sentiment analysis
# ---------------------------------------------------------------------------

SENTIMENT_PROMPT = """\
Analyze the emotional state of this customer support message.

Think step by step inside <reasoning> tags:
1. What literal emotion words or punctuation cues are present?
2. Is there sarcasm or implied frustration despite polite wording?
3. Is this about repeated failures (which raises intensity)?

Then output exactly one label inside <label> tags:
positive | neutral | frustrated | angry

Message: "{message}"

<reasoning>"""


def build_sentiment_prompt(message: str) -> str:
    return SENTIMENT_PROMPT.format(message=message.replace('"', "'"))


def parse_sentiment(raw: str) -> str:
    """Extract the label from a chain-of-thought sentiment response."""
    lowered = raw.lower()
    if "<label>" in lowered:
        segment = lowered.split("<label>")[-1].split("</label>")[0]
    else:  # fall back to scanning the tail of the response
        segment = lowered[-60:]
    for label in ("angry", "frustrated", "positive", "neutral"):
        if label in segment:
            return label
    return "neutral"


# ---------------------------------------------------------------------------
# 4. Structured JSON escalation decision
# ---------------------------------------------------------------------------

ESCALATION_PROMPT = """\
You are the escalation controller for a support bot. Given the customer's
message, the bot's draft answer, and the detected sentiment, decide whether
a human agent must take over.

Escalate when ANY of these hold:
- The knowledge base could not answer the question
- The customer explicitly asks for a human, refund, or cancellation
- Sentiment is "angry", or "frustrated" for the 2nd+ consecutive message
- Legal threats, chargebacks, or account security issues are mentioned

Respond with ONLY a JSON object, no markdown fences, matching:
{{"escalate": true/false, "reason": "<short reason>", "category": "billing|shipping|returns|account|technical|other", "priority": "low|medium|high|urgent"}}

Customer message: "{message}"
Detected sentiment: {sentiment}
Consecutive negative messages: {negative_streak}
Bot draft answer: "{draft}"

JSON:"""


def build_escalation_prompt(message: str, sentiment: str, negative_streak: int, draft: str) -> str:
    return ESCALATION_PROMPT.format(
        message=message.replace('"', "'"),
        sentiment=sentiment,
        negative_streak=negative_streak,
        draft=draft[:400].replace('"', "'"),
    )
