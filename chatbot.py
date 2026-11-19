"""Sidick's personal chatbot: a small RAG (retrieval-augmented generation) pipeline.

Pipeline: load knowledge -> chunk -> embed -> retrieve top-k by cosine similarity
-> build a first-person prompt -> ask an LLM.
"""
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()  # reads GEMINI_API_KEY from a .env file next to this script

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SYSTEM_PROMPT = """You are Sidick Abdoulaye Hissein (also known as Sidick Sino). You are his personal AI twin, \
speaking in the first person as Sidick, on his portfolio website.
Rules:
- Answer ONLY from the CONTEXT below. If the answer is not there, say you are not sure and suggest \
contacting Sidick through his portfolio, GitHub or LinkedIn. Never invent facts.
- Be friendly, concise and natural. Reply in the language the visitor uses (English or French).
- If asked "Who are you?", introduce yourself as Sidick."""


def _extract_text(content):
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


def load_chunks():
    """Split every markdown file in knowledge/ into one chunk per '## ' section."""
    chunks = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for section in re.split(r"\n(?=## )", text):
            section = section.strip()
            if len(section) > 20:
                chunks.append(section)
    return chunks


class SidickBot:
    def __init__(self):
        self.chunks = load_chunks()
        self.model = SentenceTransformer(EMBED_MODEL)
        # normalized vectors -> dot product == cosine similarity
        self.vectors = self.model.encode(self.chunks, normalize_embeddings=True)

    def retrieve(self, question, k=4):
        q = self.model.encode([question], normalize_embeddings=True)[0]
        scores = self.vectors @ q
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def build_prompt(self, question, k=4):
        # Always include the identity chunks so "Who are you?" works even when similarity is weak.
        retrieved = [c for c, _ in self.retrieve(question, k)]
        always = [c for c in self.chunks[:2] if c not in retrieved]
        context = "\n\n".join(always + retrieved)
        return f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}"

    def answer(self, question, history=None):
        from google import genai  # pip install google-genai
        from google.genai import types

        question_text = _extract_text(question)

        client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options={"api_version": "v1beta"},
        )
        # Gradio history uses roles "user"/"assistant"; Gemini expects "user"/"model".
        contents = []
        for m in (history or []):
            role = m["role"] if isinstance(m, dict) else getattr(m, "role", "user")
            raw_content = m["content"] if isinstance(m, dict) else getattr(m, "content", "")
            text = _extract_text(raw_content)
            if text:
                contents.append(
                    types.Content(
                        role="model" if role == "assistant" else "user",
                        parts=[types.Part(text=text)],
                    )
                )

        contents.append(types.Content(role="user", parts=[types.Part(text=question_text)]))
        reply = client.models.generate_content(
            model=os.getenv("CHAT_MODEL", "gemini-2.5-flash"),
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=self.build_prompt(question_text),
                max_output_tokens=500,
            ),
        )
        return reply.text
