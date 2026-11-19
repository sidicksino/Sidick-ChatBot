"""Sidick's personal chatbot: a small RAG (retrieval-augmented generation) pipeline on Ollama.

Pipeline: load knowledge -> chunk -> embed (Ollama) -> retrieve top-k by cosine similarity
-> build a first-person prompt -> stream an answer from a local Ollama model.
No external API: everything runs on your own machine or server.
If Ollama is unreachable, the bot still answers straight from the best-matching knowledge chunks.
"""
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

KNOWLEDGE_DIR = Path(os.getenv("KNOWLEDGE_DIR", ROOT / "knowledge"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", ROOT / ".cache"))
PROFILE_FILE = "profile.md"  # loaded first: its opening chunks are always sent as identity context

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen2.5:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")  # multilingual: works for English and French
KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "24h")  # keep models in memory -> no reload delay
MAX_ANSWER_TOKENS = int(os.getenv("MAX_ANSWER_TOKENS", "350"))
CONTEXT_TOKENS = int(os.getenv("CONTEXT_TOKENS", "4096"))

MAX_CHUNK_CHARS = 1200
HISTORY_TURNS = 6  # only the last N messages are sent, to keep prompts short (= faster on CPU)
TOP_K = 5

SYSTEM_PROMPT = """You are Sidick Abdoulaye Hissein (also known as Sidick Sino). You are his personal AI \
assistant on his portfolio website, and you speak in the first person as Sidick.
Rules:
- Answer ONLY from the CONTEXT below. If the answer is not there, say you are not sure and suggest \
contacting Sidick through the contact form on his portfolio, GitHub or LinkedIn. Never invent facts.
- Be friendly, concise and natural: at most about 100 words.
- Reply in the language the visitor uses (English or French).
- If asked "Who are you?", introduce yourself as Sidick."""

OFFLINE_PREFIX = "My AI brain is resting right now, but here is what I know about that:\n\n"


# ---------------------------------------------------------------- knowledge loading

def _extract_text(content):
    """Gradio may give message content as a str, a list of parts, a dict or an object with .text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
            elif item is not None:
                parts.append(str(item))
        return " ".join(parts).strip()
    if isinstance(content, dict) and "text" in content:
        return str(content["text"])
    if hasattr(content, "text"):
        return str(content.text)
    return str(content) if content is not None else ""


def split_long(section, limit=MAX_CHUNK_CHARS):
    """Split a long section into pieces <= limit chars, repeating its heading line as context."""
    if len(section) <= limit:
        return [section]
    first, _, rest = section.partition("\n")
    title = first if first.startswith("#") else ""
    paragraphs = [p for p in (rest if title else section).split("\n") if p.strip()]
    pieces, current = [], ""
    for p in paragraphs:
        if current and len(current) + len(p) > limit:
            pieces.append(current)
            current = ""
        current += p + "\n"
    if current.strip():
        pieces.append(current)
    return [f"{title}\n{p}".strip() if title else p.strip() for p in pieces]


def read_file(path):
    if path.suffix == ".pdf":
        from pypdf import PdfReader  # e.g. your LinkedIn "Save to PDF" export

        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8")


def load_chunks():
    """Read every .md / .txt / .pdf file in knowledge/ (profile.md first) and split it into chunks."""
    paths = [p for p in KNOWLEDGE_DIR.glob("*") if p.suffix in {".md", ".txt", ".pdf"}]
    paths.sort(key=lambda p: (p.name != PROFILE_FILE, p.name))
    chunks = []
    for path in paths:
        text = read_file(path)
        sections = re.split(r"\n(?=## )", text) if path.suffix == ".md" else text.split("\n\n")
        for section in sections:
            section = section.strip()
            if len(section) > 20:
                chunks.extend(split_long(section))
    return chunks


# ---------------------------------------------------------------- Ollama helpers

def ensure_models(models=(CHAT_MODEL, EMBED_MODEL)):
    """Download any missing model into Ollama (first start on a new server takes a few minutes)."""
    tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10).json().get("models", [])
    have = {m["name"] for m in tags} | {m["name"].removesuffix(":latest") for m in tags}
    for model in models:
        if model not in have:
            print(f"Pulling {model} into Ollama (one-time download)...", flush=True)
            r = requests.post(f"{OLLAMA_URL}/api/pull", json={"model": model, "stream": False}, timeout=3600)
            r.raise_for_status()


def embed(texts, batch=32):
    vectors = []
    for i in range(0, len(texts), batch):
        r = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": texts[i:i + batch], "keep_alive": KEEP_ALIVE},
            timeout=600,
        )
        r.raise_for_status()
        vectors.extend(r.json()["embeddings"])
    v = np.asarray(vectors, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)  # normalized -> dot product == cosine


# ---------------------------------------------------------------- the bot

class SidickBot:
    def __init__(self):
        self.chunks = load_chunks()
        ensure_models()
        self.vectors = self._chunk_vectors()
        self.cache = {}  # repeated first questions (e.g. example buttons) are answered instantly
        print(f"Loaded {len(self.chunks)} knowledge chunks. Chat: {CHAT_MODEL}, embeddings: {EMBED_MODEL}",
              flush=True)

    def _chunk_vectors(self):
        """Embed all chunks once and cache them on disk; re-embed only when knowledge changes."""
        key = hashlib.sha256((EMBED_MODEL + "\0" + "\0".join(self.chunks)).encode()).hexdigest()[:16]
        path = CACHE_DIR / f"embeddings-{key}.npy"
        if path.exists():
            return np.load(path)
        print(f"Embedding {len(self.chunks)} chunks with {EMBED_MODEL}...", flush=True)
        vectors = embed(self.chunks)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for old in CACHE_DIR.glob("embeddings-*.npy"):
            old.unlink()
        np.save(path, vectors)
        return vectors

    def retrieve(self, question, k=TOP_K):
        try:
            scores = self.vectors @ embed([question])[0]
        except requests.RequestException:
            scores = self._keyword_scores(question)  # Ollama down: crude word-overlap retrieval
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def _keyword_scores(self, question):
        words = {w for w in re.findall(r"\w{3,}", question.lower())}
        return np.array([sum(w in c.lower() for w in words) for c in self.chunks], dtype=np.float32)

    def build_system_prompt(self, question):
        # Always include the identity chunks so "Who are you?" works even when similarity is weak.
        retrieved = [c for c, _ in self.retrieve(question)]
        always = [c for c in self.chunks[:2] if c not in retrieved]
        return f"{SYSTEM_PROMPT}\n\nCONTEXT:\n" + "\n\n".join(always + retrieved)

    @staticmethod
    def _history_messages(history):
        messages = []
        for m in list(history or [])[-HISTORY_TURNS:]:
            role = m["role"] if isinstance(m, dict) else getattr(m, "role", "user")
            raw = m["content"] if isinstance(m, dict) else getattr(m, "content", "")
            text = _extract_text(raw)
            if text:
                messages.append({"role": "assistant" if role == "assistant" else "user", "content": text})
        return messages

    def offline_answer(self, question):
        best = [c for c, _ in self.retrieve(question, k=2)]
        return OFFLINE_PREFIX + "\n\n".join(re.sub(r"^#+\s*", "", c, flags=re.M) for c in best)

    def stream_answer(self, question, history=None):
        """Yield the answer piece by piece (so visitors see text appear while a CPU server thinks)."""
        question = _extract_text(question).strip()
        key = question.lower()
        if not history and key in self.cache:
            yield self.cache[key]
            return

        messages = [{"role": "system", "content": self.build_system_prompt(question)}]
        messages += self._history_messages(history)
        messages.append({"role": "user", "content": question})

        parts = []
        try:
            with requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": CHAT_MODEL,
                    "messages": messages,
                    "stream": True,
                    "keep_alive": KEEP_ALIVE,
                    "options": {"num_predict": MAX_ANSWER_TOKENS, "num_ctx": CONTEXT_TOKENS, "temperature": 0.3},
                },
                stream=True,
                timeout=(10, 300),
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        raise RuntimeError(data["error"])
                    piece = data.get("message", {}).get("content", "")
                    if piece:
                        parts.append(piece)
                        yield piece
                    if data.get("done"):
                        break
        except (requests.RequestException, RuntimeError, ValueError) as e:
            print(f"[warn] Ollama failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
            if not parts:  # nothing sent yet -> fall back to a retrieval-only answer
                yield self.offline_answer(question)
            return

        text = "".join(parts)
        if text and not history:
            self.cache[key] = text

    def answer(self, question, history=None):
        return "".join(self.stream_answer(question, history))
