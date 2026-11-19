"""Sidick's personal chatbot: a small RAG (retrieval-augmented generation) pipeline.

Pipeline: load knowledge -> chunk -> embed -> retrieve top-k by cosine similarity
-> build a first-person prompt -> ask an LLM (with model fallback and an offline fallback).
"""
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()  # reads GEMINI_API_KEY etc. from a .env file next to this script

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
PROFILE_FILE = "profile.md"  # loaded first: its opening chunks are always sent as identity context
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CHUNK_CHARS = 1200
HISTORY_TURNS = 6  # only the last N messages are sent, to save tokens/quota
# Tried in order when one hits its quota; *-lite models have a larger free quota.
DEFAULT_MODELS = "gemini-flash-latest,gemini-flash-lite-latest,gemini-2.5-flash"

SYSTEM_PROMPT = """You are Sidick Abdoulaye Hissein (also known as Sidick Sino). You are his personal AI twin, \
speaking in the first person as Sidick, on his portfolio website.
Rules:
- Answer ONLY from the CONTEXT below. If the answer is not there, say you are not sure and suggest \
contacting Sidick through his portfolio, GitHub or LinkedIn. Never invent facts.
- Be friendly, concise and natural. Reply in the language the visitor uses (English or French).
- If asked "Who are you?", introduce yourself as Sidick."""


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


class SidickBot:
    def __init__(self):
        self.chunks = load_chunks()
        self.model = SentenceTransformer(EMBED_MODEL)
        # normalized vectors -> dot product == cosine similarity
        self.vectors = self.model.encode(self.chunks, normalize_embeddings=True)
        self.cache = {}  # repeated first questions (e.g. the example buttons) never hit the API again
        self.models = [m.strip() for m in os.getenv("CHAT_MODELS", DEFAULT_MODELS).split(",") if m.strip()]
        print(f"Loaded {len(self.chunks)} knowledge chunks. Models: {', '.join(self.models)}")

    def retrieve(self, question, k=5):
        q = self.model.encode([question], normalize_embeddings=True)[0]
        scores = self.vectors @ q
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def build_prompt(self, question, k=5):
        # Always include the identity chunks so "Who are you?" works even when similarity is weak.
        retrieved = [c for c, _ in self.retrieve(question, k)]
        always = [c for c in self.chunks[:2] if c not in retrieved]
        context = "\n\n".join(always + retrieved)
        return f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}"

    @staticmethod
    def _to_contents(history, question):
        from google.genai import types

        # Gradio history uses roles "user"/"assistant"; Gemini expects "user"/"model".
        contents = []
        for m in history:
            role = m["role"] if isinstance(m, dict) else getattr(m, "role", "user")
            raw = m["content"] if isinstance(m, dict) else getattr(m, "content", "")
            text = _extract_text(raw)
            if text:
                contents.append(
                    types.Content(
                        role="model" if role == "assistant" else "user",
                        parts=[types.Part(text=text)],
                    )
                )
        contents.append(types.Content(role="user", parts=[types.Part(text=question)]))
        return contents

    def _offline_answer(self, question):
        """No LLM available (quota/network): answer straight from the best matching chunks."""
        best = [c for c, _ in self.retrieve(question, k=2)]
        body = "\n\n".join(re.sub(r"^#+\s*", "", c, flags=re.M) for c in best)
        return "My AI brain is resting right now (API limit reached), but here is what I know about that:\n\n" + body

    def answer(self, question, history=None):
        import httpx  # installed with google-genai
        from google import genai  # pip install google-genai
        from google.genai import errors as genai_errors
        from google.genai import types

        question = _extract_text(question)
        key = question.strip().lower()
        if not history and key in self.cache:
            return self.cache[key]

        client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options={"api_version": "v1beta"},  # system_instruction needs v1beta
        )
        contents = self._to_contents(list(history or [])[-HISTORY_TURNS:], question)
        config = types.GenerateContentConfig(
            system_instruction=self.build_prompt(question),
            max_output_tokens=600,
        )

        for model in self.models:
            try:
                reply = client.models.generate_content(model=model, contents=contents, config=config)
            except genai_errors.APIError as e:
                # 429 = quota exhausted, 5xx = overloaded, 404 = model not available: try the next model.
                # Anything else (e.g. 400/403 bad API key) is a real bug, so let it surface.
                if e.code == 429 or e.code == 404 or e.code >= 500:
                    print(f"[warn] {model} failed ({e.code}): {str(e)[:150]}")
                    continue
                raise
            except (httpx.TransportError, OSError) as e:  # network down / timeout
                print(f"[warn] {model} network error: {e}")
                continue
            if reply.text:
                if not history:
                    self.cache[key] = reply.text
                return reply.text
        return self._offline_answer(question)
