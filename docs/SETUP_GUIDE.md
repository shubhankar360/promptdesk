# PromptDesk — Full Setup & Showcase Guide

Everything you need: running locally, publishing to GitHub, demoing it, and putting it on your CV.

---

## Part 1 — Run it locally (5 minutes)

**Prerequisites:** Python 3.10+ and git.

```bash
cd promptdesk
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open **http://localhost:8000** — the bot works immediately in *demo mode* (retrieval-only answers, heuristic sentiment) with no API key.

### Enable a real LLM

```bash
cp .env.example .env
```

Edit `.env` and set **one** of:

- `ANTHROPIC_API_KEY` — get one at https://console.anthropic.com (recommended)
- `OPENAI_API_KEY` — get one at https://platform.openai.com

Restart the server. The dashboard's "LLM provider" card confirms which mode is active.

### Try these demo messages

| Message | What it demonstrates |
|---|---|
| "How long does standard shipping take?" | RAG answer with citation |
| "Great, another broken order. Fantastic." | Sarcasm → frustrated sentiment |
| "This is unacceptable, I want a refund NOW" | Angry sentiment + auto ticket + escalation |
| "Someone hacked my account!" | Security → urgent human escalation |
| "Thanks, you've been really helpful!" | Positive sentiment, no escalation |

Then open **/dashboard** to show tickets and sentiment analytics.

### Run the tests

```bash
pytest        # 19 tests, no API key required
```

---

## Part 2 — Publish to GitHub (2 minutes)

The project folder is already a git repository with an initial commit.

1. Go to **https://github.com/new**
2. Repository name: `promptdesk` · leave "Add a README" **unchecked** (we have one)
3. Click **Create repository**, then run:

```bash
cd promptdesk
git remote add origin https://github.com/YOUR_USERNAME/promptdesk.git
git branch -M main
git push -u origin main
```

(If git asks who you are first: `git config user.name "Your Name"` and `git config user.email "you@example.com"`, then `git commit --amend --reset-author --no-edit`.)

### Make the repo look professional

- **About section** (right sidebar → gear icon): description *"AI customer support agent — prompt engineering, RAG, sentiment-aware escalation (FastAPI)"*, topics: `llm`, `prompt-engineering`, `rag`, `chatbot`, `fastapi`, `python`, `customer-support`.
- **Add a screenshot**: run the app, screenshot the chat UI, save as `docs/screenshot.png`, reference it near the top of README.md.
- **Pin the repo** on your GitHub profile (Profile → Customize your pins).

---

## Part 3 — Put it on your CV

### Project entry (pick 2-3 bullets)

> **PromptDesk — AI Customer Support Agent** · Python, FastAPI, Claude/OpenAI API · github.com/YOU/promptdesk
>
> - Engineered a 5-stage LLM pipeline (sentiment → intent → RAG → generation → escalation) using layered system prompts, few-shot classification, and chain-of-thought reasoning
> - Implemented a TF-IDF retrieval engine from scratch (no ML libraries) to ground answers in a document knowledge base with per-response source citations
> - Enforced reliability with Pydantic-validated structured JSON outputs — malformed LLM responses fail safe with zero side effects; 19-test pytest suite runs without API keys
> - Built automatic ticket creation and human escalation triggered by detected customer sentiment, plus a real-time analytics dashboard (escalation rate, sentiment breakdown)

### Interview talking points

- **"Why a pipeline instead of one prompt?"** Each stage is observable and testable; sentiment must run before generation because it rewrites the tone layer of the system prompt.
- **"How do you prevent hallucination?"** Retrieval-grounded context in tagged blocks, explicit guardrails ("never invent order numbers"), citation requirements, and an honest no-answer path that escalates instead of guessing.
- **"How do you handle unreliable LLM output?"** Structured outputs are parsed defensively and validated against a Pydantic schema before any side effect (ticket creation) runs.
- **"Why TF-IDF instead of embeddings?"** Zero heavy dependencies, fully explainable ranking, fast enough for the corpus size — and it shows I understand the math rather than just calling an API. Swapping in embeddings is a one-class change.

### Suggested next steps (great "future work" answers)

1. Swap TF-IDF for vector embeddings + hybrid search
2. Add streaming responses (SSE) to the chat UI
3. Add an LLM-as-judge eval harness scoring answer faithfulness against the KB
4. Dockerize and deploy (Railway/Render free tier) so recruiters can click a live link

---

## Part 4 — How it works (architecture walkthrough)

1. **`POST /api/chat`** receives the message and session id.
2. **Sentiment** (`prompts.build_sentiment_prompt`): chain-of-thought classification → `positive | neutral | frustrated | angry`.
3. **Intent** (`prompts.build_intent_prompt`): few-shot classification → `question | complaint | request | chitchat | other`.
4. **Retrieval** (`rag.Retriever`): the message is tokenized and scored against knowledge-base chunks by TF-IDF cosine similarity; top-3 passages above threshold are kept.
5. **Generation**: system prompt is composed with the sentiment-matched tone block; retrieved passages are injected as tagged `<document>` blocks; conversation history is prepended.
6. **Escalation** (`prompts.build_escalation_prompt`): a JSON-only prompt judges the message + draft answer + negative-message streak; the JSON is validated by Pydantic; if `escalate=true`, a ticket row is written and the reply is amended with the ticket number.
7. Everything is persisted to SQLite and surfaced at `/dashboard`.

File map:

| File | Responsibility |
|---|---|
| `app/prompts.py` | All prompt templates and parsers (the showcase) |
| `app/agent.py` | Pipeline orchestration |
| `app/llm.py` | Claude / OpenAI / demo-mode abstraction |
| `app/rag.py` | From-scratch TF-IDF retriever |
| `app/database.py` | SQLite conversations, tickets, analytics |
| `app/models.py` | Pydantic schemas + defensive JSON parsing |
| `app/main.py` | FastAPI routes |
| `static/` | Chat UI + analytics dashboard |
| `knowledge_base/` | Markdown docs the bot answers from |
