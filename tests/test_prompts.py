"""Prompt construction and structured-output parsing tests."""

from app import prompts
from app.models import parse_escalation


def test_system_prompt_layers_present():
    sp = prompts.build_system_prompt("neutral")
    assert "Aria" in sp                     # role block
    assert "Never invent" in sp             # guardrails block
    assert "source document" in sp          # citation block


def test_tone_adapts_to_sentiment():
    angry = prompts.build_system_prompt("angry")
    positive = prompts.build_system_prompt("positive")
    assert "escalation" in angry.lower()
    assert angry != positive


def test_context_block_wraps_passages_with_titles():
    block = prompts.build_context_block(
        [{"title": "Returns Policy", "text": "30 days.", "score": 0.9}]
    )
    assert '<document title="Returns Policy"' in block
    assert "30 days." in block


def test_context_block_handles_no_passages():
    assert "No relevant documents" in prompts.build_context_block([])


def test_parse_sentiment_reads_label_tag():
    assert prompts.parse_sentiment("<reasoning>...</reasoning>\n<label>angry</label>") == "angry"


def test_parse_sentiment_defaults_to_neutral():
    assert prompts.parse_sentiment("no label here") == "neutral"


def test_parse_escalation_valid_json():
    d = parse_escalation('{"escalate": true, "reason": "refund", "category": "billing", "priority": "high"}')
    assert d.escalate is True
    assert d.category == "billing"


def test_parse_escalation_strips_markdown_fences():
    raw = '```json\n{"escalate": false, "reason": "ok", "category": "other", "priority": "low"}\n```'
    assert parse_escalation(raw).escalate is False


def test_parse_escalation_survives_garbage():
    d = parse_escalation("the model rambled and returned no json")
    assert d.escalate is False  # fail safe: no side effects on bad output
