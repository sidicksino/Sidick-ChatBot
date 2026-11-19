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

# Off-topic guard. A question whose best knowledge match scores below this is not about Sidick
# (celebrities, weather, homework, "write me code"...) and is declined without calling the LLM.
# Measured with bge-m3 on this knowledge: questions about Sidick scored 0.476-0.665,
# clearly off-topic ones 0.351-0.460. Re-measure if you change EMBED_MODEL.
OFFTOPIC_THRESHOLD = float(os.getenv("OFFTOPIC_THRESHOLD", "0.465"))

SYSTEM_PROMPT = """You are the AI assistant on the portfolio website of Sidick Abdoulaye Hissein \
(also known as Sidick Sino). You speak in the first person, as Sidick.
Rules:
- ONLY answer questions about Sidick: his background, education, skills, projects, services, \
availability and contact details. Use ONLY facts from the CONTEXT below. Never invent facts.
- If the question is about anything else (other people, celebrities, sports, news, general knowledge, \
homework, writing code, jokes, recipes...), do NOT answer it, even if you know the answer. Politely say \
you can only talk about Sidick and his work, and suggest a question about his projects or skills.
- A question about another person ("Who is X?") is NOT about you. Only when the visitor asks who YOU \
are, introduce yourself as Sidick.
- If the CONTEXT does not contain the answer, say you are not sure and suggest contacting Sidick \
through the contact form on his portfolio, GitHub or LinkedIn.
- Be friendly, concise and natural: at most about 100 words.
- Reply in the language the visitor uses (English or French)."""

# Repeated after the CONTEXT: small models follow the instructions they read last.
PROMPT_REMINDER = """REMINDER: answer only about Sidick, using only the CONTEXT above. \
Politely decline anything unrelated. Reply in the visitor's language."""

OFFLINE_PREFIX = "My AI brain is resting right now, but here is what I know about that:\n\n"

# Instant replies that never reach the LLM (cheap on a CPU server, and always on-topic).
CANNED = {
    "en": {
        "offtopic": "I'm Sidick's assistant, so I can only answer questions about Sidick: his skills, "
                    "projects, services and background. Try asking me what I've built, or how to contact me!",
        "greeting": "Hi! 👋 Ask me anything about my skills, projects, services or background.",
        "thanks": "You're welcome! Anything else you'd like to know about my work?",
        "bye": "Thanks for visiting! Feel free to reach out through the contact form anytime.",
    },
    "fr": {
        "offtopic": "Je suis l'assistant de Sidick, je ne peux donc répondre qu'aux questions sur Sidick : "
                    "ses compétences, ses projets, ses services et son parcours. Demande-moi par exemple "
                    "ce que j'ai réalisé, ou comment me contacter !",
        "greeting": "Salut ! 👋 Pose-moi tes questions sur mes compétences, mes projets, mes services ou mon parcours.",
        "thanks": "Avec plaisir ! Tu veux savoir autre chose sur mon travail ?",
        "bye": "Merci de ta visite ! N'hésite pas à me contacter via le formulaire de contact.",
    },
}
_SMALL_TALK = [
    ("greeting", re.compile(r"^(hi+|hello|hey|yo|hiya|good (morning|afternoon|evening)|bonjour|bonsoir|salut|"
                            r"coucou|slt|cc)\b[\s!.,?👋🙂😊]*(sidick|there|toi)?[\s!.,?👋🙂😊]*$", re.I)),
    ("thanks", re.compile(r"^(thanks?( you)?( so much| a lot)?|thx|ty|merci( beaucoup| bien)?|"
                          r"great|cool|nice|super|ok(ay)?|d'accord)\b[\s!.,🙏👍🙂😊]*$", re.I)),
    ("bye", re.compile(r"^(bye|goodbye|see you|ciao|au revoir|à bientôt|a\+|bonne (journée|soirée))\b[\s!.,👋]*$", re.I)),
]
# Requests a small model tends to obey despite the prompt ("tell me a joke"). Only imperative
# forms, so "does SnapChef generate recipes?" or "tell me your story" still reach the LLM.
_OFFTOPIC_REQUEST = re.compile(
    r"^\s*(tell|give|write|make|compose|sing|raconte|racontes|donne|donnes|écris|ecris|fais|chante)\b"
    r".*\b(jokes?|blagues?|poems?|poèmes?|poemes?|songs?|chansons?|recipes?|recettes?|essays?|dissertations?)\b",
    re.I)
_FOLLOW_UP_WORDS = 6  # questions this short, mid-conversation, are searched with the previous one

_FRENCH_HINTS = re.compile(
    r"[àâçéèêëîïôûùüÿœ]|\b(je|tu|il|nous|vous|est|es|suis|qui|quoi|quel|quels|quelle|quelles|comment|pourquoi|"
    r"où|combien|le|la|les|des|du|une|un|et|ton|ta|tes|mon|ma|mes|avec|pour|dans|sur|peux|bonjour|salut|"
    r"merci|parle|moi|toi)\b", re.I)


def detect_lang(text):
    """Very small English/French guess, only used for the canned replies above."""
    return "fr" if len(_FRENCH_HINTS.findall(text)) >= 1 and not re.search(r"\b(the|what|who|you|is|are)\b", text, re.I) \
        else "en"


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

    def _scores(self, question):
        """Similarity of the question to every chunk. Second value: True if semantic (Ollama up)."""
        try:
            return self.vectors @ embed([question])[0], True
        except requests.RequestException:
            return self._keyword_scores(question), False  # Ollama down: crude word-overlap retrieval

    def retrieve(self, question, k=TOP_K):
        scores, _ = self._scores(question)
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def check_scope(self, question, history_messages):
        """Decide how to handle a question before spending an LLM call on it.

        Returns (kind, retrieved): kind is "greeting"/"thanks"/"bye"/"offtopic" for an instant
        canned reply, or None to answer with the LLM using `retrieved` (top chunks).
        """
        for kind, pattern in _SMALL_TALK:
            if pattern.match(question):
                return kind, []
        if _OFFTOPIC_REQUEST.match(question):
            print(f"[guard] off-topic request: {question[:80]!r}", flush=True)
            return "offtopic", []

        scores, semantic = self._scores(question)
        # Follow-ups ("tell me more", "and the second one?") mean little on their own: search them
        # together with the visitor's previous question, both for the guard and for the context.
        previous = [m["content"] for m in history_messages if m["role"] == "user"]
        if previous and (len(question.split()) <= _FOLLOW_UP_WORDS or scores.max() < OFFTOPIC_THRESHOLD):
            combined, _ = self._scores(previous[-1] + " " + question)
            if combined.max() >= scores.max() or scores.max() < OFFTOPIC_THRESHOLD:
                scores = combined
        if semantic and scores.max() < OFFTOPIC_THRESHOLD:
            print(f"[guard] off-topic ({scores.max():.3f}): {question[:80]!r}", flush=True)
            return "offtopic", []
        top = np.argsort(scores)[::-1][:TOP_K]
        return None, [(self.chunks[i], float(scores[i])) for i in top]

    def _keyword_scores(self, question):
        words = {w for w in re.findall(r"\w{3,}", question.lower())}
        return np.array([sum(w in c.lower() for w in words) for c in self.chunks], dtype=np.float32)

    def build_system_prompt(self, question, retrieved=None):
        # Always include the identity chunks so "Who are you?" works even when similarity is weak.
        chunks = [c for c, _ in (retrieved if retrieved is not None else self.retrieve(question))]
        always = [c for c in self.chunks[:2] if c not in chunks]
        return f"{SYSTEM_PROMPT}\n\nCONTEXT:\n" + "\n\n".join(always + chunks) + f"\n\n{PROMPT_REMINDER}"

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

        past = self._history_messages(history)
        kind, retrieved = self.check_scope(question, past)
        if kind:  # small talk or off-topic: instant reply, no LLM call
            yield CANNED[detect_lang(question)][kind]
            return

        messages = [{"role": "system", "content": self.build_system_prompt(question, retrieved)}]
        messages += past
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
