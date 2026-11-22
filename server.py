"""Public API for the portfolio chat widget.

POST /chat   {"message": "...", "history": [{"role": "user"|"assistant", "content": "..."}], "lang": "en"|"fr"}
             -> the answer streamed as plain text
GET  /health -> {"status": "ok" | "loading", ...}

Run locally:  uvicorn server:app --port 8000
"""
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import chatbot

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "https://www.sidickabdoulaye.com,https://sidickabdoulaye.com").split(",")
                   if o.strip()]
MAX_MESSAGE_CHARS = int(os.getenv("MAX_MESSAGE_CHARS", "500"))
RATE_PER_MINUTE = int(os.getenv("RATE_PER_MINUTE", "6"))     # per visitor
RATE_PER_DAY = int(os.getenv("RATE_PER_DAY", "60"))          # per visitor
GLOBAL_PER_DAY = int(os.getenv("GLOBAL_PER_DAY", "500"))     # all visitors together

bot = None
load_error = None


def _load_bot():
    """Load in the background so /health answers right away (first start may download models)."""
    global bot, load_error
    while bot is None:
        try:
            bot = chatbot.SidickBot()
            load_error = None
        except Exception as e:  # Ollama not up yet, model downloading, ...
            load_error = f"{type(e).__name__}: {e}"
            print(f"[warn] bot not ready ({load_error[:200]}), retrying in 10s", flush=True)
            time.sleep(10)


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=_load_bot, daemon=True).start()
    yield


app = FastAPI(title="Sidick chatbot", lifespan=lifespan, docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ---------------------------------------------------------------- rate limiting (in memory)
_lock = threading.Lock()
_visitor_hits = defaultdict(deque)  # ip -> timestamps of the last 24h
_global_hits = deque()
DAY = 86400


def check_rate(ip):
    now = time.time()
    with _lock:
        while _global_hits and now - _global_hits[0] > DAY:
            _global_hits.popleft()
        if len(_visitor_hits) > 5000:  # forget visitors with no recent activity
            for k in [k for k, q in _visitor_hits.items() if not q or now - q[-1] > DAY]:
                del _visitor_hits[k]
        hits = _visitor_hits[ip]
        while hits and now - hits[0] > DAY:
            hits.popleft()
        if sum(1 for t in hits if now - t < 60) >= RATE_PER_MINUTE:
            raise HTTPException(429, "You're sending messages too fast. Please wait a minute.")
        if len(hits) >= RATE_PER_DAY:
            raise HTTPException(429, "You've reached today's message limit. Please come back tomorrow.")
        if len(_global_hits) >= GLOBAL_PER_DAY:
            raise HTTPException(503, "The assistant is resting for today. Please use the contact form.")
        hits.append(now)
        _global_hits.append(now)


# ---------------------------------------------------------------- endpoints
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    history: list[Message] = Field(default_factory=list, max_length=40)
    # The site's language. Only breaks ties: the bot answers in the language of the message itself.
    lang: Literal["en", "fr"] | None = None


@app.get("/health")
def health():
    if bot is None:
        return {"status": "loading", "detail": (load_error or "starting")[:200]}
    return {"status": "ok", "chunks": len(bot.chunks), "model": chatbot.CHAT_MODEL}


@app.post("/chat")
def chat(req: ChatRequest, request: Request):
    if bot is None:
        raise HTTPException(503, "The assistant is waking up. Please try again in a moment.")
    check_rate(request.client.host if request.client else "unknown")
    history = [m.model_dump() for m in req.history]
    return StreamingResponse(
        bot.stream_answer(req.message, history, req.lang),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
