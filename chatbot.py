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

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        # Gradio history uses roles "user"/"assistant"; Gemini expects "user"/"model".
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part(text=m["content"])],
            )
            for m in (history or [])
        ]
        contents.append(types.Content(role="user", parts=[types.Part(text=question)]))
        reply = client.models.generate_content(
            model=os.getenv("CHAT_MODEL", "gemini-2.5-flash"),
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=self.build_prompt(question),
                max_output_tokens=500,
            ),
        )
        return reply.text
