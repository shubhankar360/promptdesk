"""Environment-based configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

DATABASE_PATH = os.getenv("DATABASE_PATH", str(BASE_DIR / "promptdesk.db"))
KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", str(BASE_DIR / "knowledge_base"))


def active_provider() -> str:
    if ANTHROPIC_API_KEY:
        return "anthropic"
    if OPENAI_API_KEY:
        return "openai"
    return "demo"
