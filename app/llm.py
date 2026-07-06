"""Provider-agnostic LLM client: Anthropic Claude, OpenAI, or keyless demo mode.

The rest of the codebase only ever calls `complete(system, prompt)` - swapping
providers is a config change, not a code change.
"""

from __future__ import annotations

from app import config


class LLMClient:
    def __init__(self) -> None:
        self.provider = config.active_provider()
        self._client = None
        if self.provider == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        elif self.provider == "openai":
            import openai

            self._client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

    def complete(self, system: str, prompt: str, max_tokens: int = 600) -> str:
        if self.provider == "anthropic":
            resp = self._client.messages.create(
                model=config.ANTHROPIC_MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        if self.provider == "openai":
            resp = self._client.chat.completions.create(
                model=config.OPENAI_MODEL,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            return resp.choices[0].message.content or ""
        return self._demo_complete(prompt)

    # ------------------------------------------------------------------
    # Keyless demo mode: deterministic heuristics so the app is runnable
    # (and demoable to recruiters) without any API key.
    # ------------------------------------------------------------------

    @staticmethod
    def _demo_complete(prompt: str) -> str:
        if "<reasoning>" in prompt:  # sentiment prompt
            # Scan only the customer message, not the prompt template itself.
            text = prompt.lower()
            if 'message: "' in text:
                text = text.split('message: "')[-1].split('"')[0]
            if any(w in text for w in ("unacceptable", "furious", "worst", "angry", "scam", "lawyer")):
                label = "angry"
            elif any(w in text for w in ("frustrat", "annoy", "third time", "again", "still not", "disappointed")):
                label = "frustrated"
            elif any(w in text for w in ("thank", "great", "love", "awesome", "perfect")):
                label = "positive"
            else:
                label = "neutral"
            return f"Demo-mode heuristic.</reasoning>\n<label>{label}</label>"

        if '"escalate"' in prompt:  # escalation prompt
            # Scan only the customer message and detected sentiment, not the rules text.
            text = prompt.lower()
            message = text.split('customer message: "')[-1].split('"')[0] if 'customer message: "' in text else text
            needs_human = any(
                w in message
                for w in ("refund", "cancel", "human", "agent", "lawyer", "chargeback", "hacked")
            ) or "detected sentiment: angry" in text
            reason = "customer requested human/refund or is angry" if needs_human else "bot can handle"
            return (
                '{"escalate": %s, "reason": "%s", "category": "other", "priority": "%s"}'
                % (str(needs_human).lower(), reason, "high" if needs_human else "low")
            )

        if "Intent:" in prompt:  # intent prompt
            return "question"

        # Answer generation: extract the top retrieved passage verbatim.
        if "<document" in prompt:
            try:
                title = prompt.split('title="')[1].split('"')[0]
                text = prompt.split('">')[1].split("</document>")[0].strip()
                snippet = text[:400]
                return (
                    f"[Demo mode - no API key configured] Based on our "
                    f"“{title}” document: {snippet}"
                )
            except (IndexError, ValueError):
                pass
        return (
            "[Demo mode] I couldn't find that in the knowledge base. "
            "Add an ANTHROPIC_API_KEY or OPENAI_API_KEY to .env for full answers."
        )
