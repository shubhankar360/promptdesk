# Prompt Engineering in PromptDesk

This document explains each prompting technique used in [`app/prompts.py`](../app/prompts.py), why it exists, and what breaks without it. This is the heart of the project.

## 1. Layered system prompts

The system prompt is composed from four independent blocks — role, guardrails, citation rules, and tone — joined at call time by `build_system_prompt()`.

**Why layers instead of one big string?** Each concern can be edited, versioned, and tested in isolation. `test_system_prompt_layers_present` asserts every layer survives composition, so a refactor can't silently drop the guardrails.

**What the guardrails prevent:** invented order numbers, promised refunds, requests for sensitive data, and answers not grounded in the knowledge base. These are the four most common failure modes of naive support bots.

## 2. Few-shot intent classification

`INTENT_PROMPT` shows the model four labeled examples before asking it to classify the real message. With zero-shot prompting, models drift between label vocabularies ("query" vs "question", "complaint" vs "issue"). Anchoring with examples keeps outputs inside the fixed label set, which downstream analytics depend on.

## 3. Chain-of-thought sentiment analysis

`SENTIMENT_PROMPT` forces the model to reason inside `<reasoning>` tags **before** committing to a label inside `<label>` tags:

1. literal emotion cues,
2. sarcasm / implied frustration,
3. repeated-failure intensity.

The pre-seeded `<reasoning>` opening tag ("prefilling") means the model starts mid-thought instead of skipping to a snap judgment. `parse_sentiment()` reads only the `<label>` block, so verbose reasoning never leaks into the pipeline. Edge cases this catches that keyword matching misses: *"Great, another broken order. Fantastic."* (sarcasm → frustrated, not positive).

## 4. Structured JSON output with validation

The escalation controller must produce machine-actionable output — a ticket either gets created or it doesn't. `ESCALATION_PROMPT`:

- states the decision criteria explicitly (no vibes),
- pins the exact JSON schema in the prompt,
- says "ONLY a JSON object, no markdown fences".

Then the response is **validated with Pydantic** (`parse_escalation`) before any side effect runs. Malformed output fails safe: no ticket, `escalate=false`, reason logged. LLM output is never trusted directly — `test_parse_escalation_survives_garbage` proves the failure path.

## 5. RAG context injection with citations

Retrieved passages are wrapped in XML-style tags with metadata:

```xml
<document title="Returns Policy" relevance="0.82">
...passage text...
</document>
```

Tagged blocks give the model unambiguous boundaries between *context* and *instruction* — a common source of prompt injection and hallucination. The citation instruction ("mention the source document naturally") makes answers auditable: the UI shows which documents were used for every reply.

When retrieval finds nothing, the prompt says so explicitly rather than leaving the block empty — silence invites the model to improvise.

## 6. Dynamic tone adaptation

The detected sentiment selects a tone block that rewrites part of the system prompt for the answer call. The angry-customer block encodes real support playbook rules: don't be defensive, take ownership, one concrete action, offer a human. This is prompting as *behavior design*, not just formatting.

## Pipeline order matters

```
sentiment → intent → retrieve → answer (tone-adapted) → escalation check
```

Sentiment runs first because it changes the answer prompt. Escalation runs last because it needs the draft answer to judge whether the bot actually helped.

## Things intentionally not done

- **No fine-tuning** — every behavior here is achievable with prompting, which iterates in seconds instead of hours.
- **No agentic tool-calling loop** — the pipeline is deterministic and observable; each stage can be logged and unit-tested. For support workflows, predictability beats autonomy.
