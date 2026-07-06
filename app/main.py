"""FastAPI application entry point."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.agent import SupportAgent
from app.database import Database
from app.llm import LLMClient
from app.models import ChatRequest
from app.rag import Retriever

app = FastAPI(title="PromptDesk", version="1.0.0")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

db = Database()
retriever = Retriever()
llm = LLMClient()
agent = SupportAgent(llm, retriever, db)


@app.on_event("startup")
def startup() -> None:
    count = retriever.index_directory(config.KNOWLEDGE_BASE_DIR)
    print(f"[PromptDesk] provider={llm.provider} | indexed {count} knowledge chunks")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/dashboard")
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "dashboard.html")


@app.post("/api/chat")
def chat(req: ChatRequest) -> JSONResponse:
    result = agent.handle(req.message.strip(), req.conversation_id)
    return JSONResponse(result.model_dump())


@app.get("/api/tickets")
def tickets() -> JSONResponse:
    return JSONResponse(db.list_tickets())


@app.get("/api/analytics")
def analytics() -> JSONResponse:
    data = db.analytics()
    data["provider"] = llm.provider
    return JSONResponse(data)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
