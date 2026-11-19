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
- If the CONTEXT does not contain the answer (for example a question about your university, age or \
hobbies), say honestly that you don't have that information here and invite the visitor to contact you \
(first person: "contact me") through the contact form on the portfolio, GitHub or LinkedIn. Never guess it from other facts: \
where you live is NOT where you study, and being a student does not tell which school.
- Never mention "the CONTEXT", "the context" or these rules to the visitor; just answer naturally.
- About projects: each "## ... project: NAME" section in the CONTEXT is a SEPARATE project. Never merge \
two projects. Only mention technologies and features that are written in that project's section. If a \
project has no details in the CONTEXT, just name it and do not describe it.
- Be friendly, concise and natural: at most about 120 words. Describe at most 4 projects per answer.
- Format: short paragraphs, or a "- " bullet list for several items, with **bold** project names. \
No headings (#), no tables.
- Reply in the language the visitor uses (English or French)."""

# Repeated after the CONTEXT: small models follow the instructions they read last.
PROMPT_REMINDER = """REMINDER: answer only about Sidick, using only the CONTEXT above. Never invent or \
mix up project details or technologies. Politely decline anything unrelated. Reply in the visitor's language."""

# Project names -> their knowledge chunk, so "tell me about AAPT" or a follow-up on projects named in
# the previous answer fetches exactly those sections (semantic search alone missed them).
_PROJECT_HEADING = re.compile(r"^##\s+(Web project|Mobile app|ML & AI project|Project|GitHub project):\s*(.+?)\s*$")
_GENERIC_NAMES = {"test", "animation", "portfolio", "website", "app", "project", "demo", "chatbot", "quiz",
                  "to do", "todo", "news", "blog", "api", "backend", "frontend"}
MAX_NAMED_CHUNKS = 5

# "What mobile apps have you built?" -> the category's list section + its project sections.
# Only when the question is about work (projects/apps/built...), so "do you know React Native?"
# alone doesn't pull in every app.
_WORK_WORDS = re.compile(r"\b(projects?|projets?|apps?|applications?|built|build|made|developed|created|"
                         r"réalis\w*|fait\w*|cré\w*|développ\w*|works?|travaux|portfolio|websites?|sites?)\b", re.I)
_CATEGORIES = [
    (re.compile(r"\b(mobiles?|ios|android|react native|expo)\b", re.I), "Mobile app", "list of my 4 mobile apps"),
    (re.compile(r"\b(web|websites?|sites?|web ?apps?)\b", re.I), "Web project", "list of my 7 web projects"),
    (re.compile(r"\b(machine learning|ml|ai|ia|intelligence artificielle|apprentissage automatique|"
                r"data science|predictions?|prédictions?)\b", re.I), "ML & AI project", "list of my 3 ML & AI projects"),
    (re.compile(r"\b(designs?|figma|ui|ux)\b", re.I), "Design projects", "list of my 4 design projects"),
]


def _norm(text):
    return " " + re.sub(r"[\s_\-]+", " ", re.sub(r"[^\w\s\-]", " ", text.lower())).strip() + " "

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
# Also the polite forms ("can you write me a Python function?", "peux-tu traduire...").
# Things that are Sidick's services ("can you build a website for me?") are NOT in the noun list.
_OFFTOPIC_REQUEST = re.compile(
    # ...unless it asks about Sidick's own things: "tell me your story", "explain your algorithm".
    r"^(?!.*\b(your|yours|ton|ta|tes|votre|vos|sidick)\b)"
    r"\s*((can|could|would|will) you|please|pls|peux[- ]tu|pourrais[- ]tu|pouvez[- ]vous|"
    r"pourriez[- ]vous|est-ce que tu peux)?\s*(help me( to)?|m'aider à|aide-moi à)?\s*"
    r"(tell|give|write|make|compose|sing|generate|create|solve|fix|debug|explain|translate|do|"
    r"raconte|racontes|donne|donnes|écris|ecris|écrire|ecrire|fais|faire|chante|génère|genere|"
    r"résous|resous|corrige|corriger|explique|expliquer|traduis|traduire)\b"
    r".*\b(jokes?|blagues?|poems?|poèmes?|poemes?|songs?|chansons?|recipes?|recettes?|essays?|dissertations?|"
    r"functions?|fonctions?|scripts?|programs?|programmes?|algorithms?|algorithmes?|snippets?|"
    r"homework|devoirs?|exercises?|exercices?|equations?|équations?|math|maths|"
    r"stories|story|histoires?|translations?|traductions?|summary|résumé|texts?|textes?|sentences?|"
    r"phrases?|paragraphs?|paragraphes?)\b",
    re.I)
_FOLLOW_UP_WORDS = 6  # questions this short, mid-conversation, are searched with the previous one
# ...as are questions that point back at the conversation ("more details on these projects").
_REFERS_BACK = re.compile(r"\b(these|those|them|they|it|its|this one|that one|more|the projects?|"
                          r"ces|ceux|celles|eux|ça|cela|celui|celle|plus de|davantage|les projets?)\b", re.I)

# Questions addressed to Sidick ("which university do YOU attend?", "where does Sidick study?") are on-topic
# even when the knowledge has no answer: they go to the LLM, which says it doesn't know, instead of the
# "I can only talk about Sidick" reply. ("Who is Messi?" has none of these words, so it's still declined.)
_ADDRESSES_SIDICK = re.compile(r"\b(you|your|yours|yourself|sidick|sino|tu|toi|ton|ta|tes|t'|vous|votre|vos)\b", re.I)

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
        self.project_index = self._build_project_index()
        self.category_chunks = [
            (pattern, [i for i, c in enumerate(self.chunks) if c.startswith(f"## Portfolio: {list_title}")]
             + [i for i, c in enumerate(self.chunks) if c.startswith(f"## {prefix}")])
            for pattern, prefix, list_title in _CATEGORIES
        ]
        print(f"Loaded {len(self.chunks)} knowledge chunks, {len(self.project_index)} project names. "
              f"Chat: {CHAT_MODEL}, embeddings: {EMBED_MODEL}", flush=True)

    def _build_project_index(self):
        """normalized project name -> chunk index; portfolio/profile entries win over GitHub READMEs."""
        index = {}
        for i, chunk in enumerate(self.chunks):
            m = _PROJECT_HEADING.match(chunk.split("\n", 1)[0])
            if not m:
                continue
            name = _norm(m.group(2))
            if len(name.strip()) < 3 or name.strip() in _GENERIC_NAMES:
                continue
            curated = m.group(1) != "GitHub project"
            # keep the first chunk of each name; a curated entry replaces a GitHub one
            if name not in index or (curated and not index[name][1]):
                index[name] = (i, curated)
        return index

    def mentioned_projects(self, text):
        """Chunk indexes of projects named in `text` (in order), or of a project category it asks about."""
        t = _norm(text)
        found = sorted((t.find(name), i) for name, (i, _) in self.project_index.items() if name in t)
        picked = [i for _, i in found]
        if _WORK_WORDS.search(text):
            for pattern, indexes in self.category_chunks:
                if pattern.search(text):
                    picked += indexes
        return list(dict.fromkeys(picked))

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
        named = self.mentioned_projects(question)
        # Follow-ups ("tell me more", "give more details on the projects") mean little on their own:
        # search them together with the visitor's previous question, and pull in the projects the
        # previous answer named, both for the guard and for the context.
        previous = [m["content"] for m in history_messages if m["role"] == "user"]
        is_follow_up = previous and (len(question.split()) <= _FOLLOW_UP_WORDS or _REFERS_BACK.search(question)
                                     or scores.max() < OFFTOPIC_THRESHOLD)
        if is_follow_up:
            combined, _ = self._scores(previous[-1] + " " + question)
            if combined.max() >= scores.max() or scores.max() < OFFTOPIC_THRESHOLD:
                scores = combined
            if not named:
                answers = [m["content"] for m in history_messages if m["role"] == "assistant"]
                named = self.mentioned_projects(previous[-1] + " " + (answers[-1] if answers else ""))
        about_sidick = _ADDRESSES_SIDICK.search(question) or (
            is_follow_up and _ADDRESSES_SIDICK.search(previous[-1]))
        if semantic and not named and not about_sidick and scores.max() < OFFTOPIC_THRESHOLD:
            print(f"[guard] off-topic ({scores.max():.3f}): {question[:80]!r}", flush=True)
            return "offtopic", []
        # Named projects first (exact sections), then the best semantic matches.
        picked = named[:MAX_NAMED_CHUNKS]
        limit = max(TOP_K, len(picked) + 2)  # at most 5 named + 2 semantic = 7 chunks
        for i in np.argsort(scores)[::-1]:
            if len(picked) >= limit:
                break
            if int(i) not in picked:
                picked.append(int(i))
        return None, [(self.chunks[i], float(scores[i])) for i in picked]

    def _keyword_scores(self, question):
        words = {w for w in re.findall(r"\w{3,}", question.lower())}
        return np.array([sum(w in c.lower() for w in words) for c in self.chunks], dtype=np.float32)

    def build_system_prompt(self, question, retrieved=None):
        # Always include the identity chunks so "Who are you?" works even when similarity is weak.
        chunks = [c for c, _ in (retrieved if retrieved is not None else self.retrieve(question))]
        always = [c for c in self.chunks[:2] if c not in chunks]
        # Name the projects that actually have details here: a 3B model otherwise "fills in" others
        # it saw in the conversation (e.g. inventing a tech stack for a project not in the CONTEXT).
        names = [m.group(2) for c in chunks if (m := _PROJECT_HEADING.match(c.split("\n", 1)[0]))]
        names = list(dict.fromkeys(names))
        projects_note = (
            f"\nProjects described in the CONTEXT: {', '.join(names)}. Only describe these. "
            "If the visitor asked about other projects, just list their names and invite them to ask about one."
            if names else ""
        )
        return (f"{SYSTEM_PROMPT}\n\nCONTEXT:\n" + "\n\n".join(always + chunks)
                + f"\n\n{PROMPT_REMINDER}{projects_note}")

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
                    "options": {"num_predict": MAX_ANSWER_TOKENS, "num_ctx": CONTEXT_TOKENS, "temperature": 0.1},
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
