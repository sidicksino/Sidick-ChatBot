"""Sidick's personal chatbot: a small RAG (retrieval-augmented generation) pipeline.

Pipeline: load knowledge -> chunk -> retrieve the best chunks -> build a first-person prompt -> stream an answer.

Two ways to run it, picked with LLM_PROVIDER in .env:
- "ollama" (default): everything local. bge-m3 embeddings for retrieval, qwen2.5:3b for answers.
  Needs ~4GB of RAM (your Mac, or the Oracle Cloud server in the README).
- "openai": any OpenAI-compatible chat API (Groq by default; xAI's Grok works too). Retrieval switches to
  keyword search (BM25), because Groq has no embeddings: no model runs on the server, so it fits Render's
  free 512MB instance.
If the model can't be reached, the bot still answers straight from the best-matching knowledge chunks.
"""
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

KNOWLEDGE_DIR = Path(os.getenv("KNOWLEDGE_DIR", ROOT / "knowledge"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", ROOT / ".cache"))
PROFILE_FILE = "profile.md"  # loaded first: its opening chunks are always sent as identity context

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
if LLM_PROVIDER not in ("ollama", "openai"):
    raise ValueError(f"LLM_PROVIDER must be 'ollama' or 'openai', not {LLM_PROVIDER!r}")
USE_OLLAMA = LLM_PROVIDER == "ollama"

# Hosted API (LLM_PROVIDER=openai). Defaults are Groq's. Others that work the same way:
#   Gemini: LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai  LLM_MODEL=gemini-2.5-flash
#   xAI:    LLM_BASE_URL=https://api.x.ai/v1  LLM_MODEL=<a grok model id>
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
if not USE_OLLAMA and not LLM_API_KEY:
    raise ValueError("LLM_PROVIDER=openai needs LLM_API_KEY (your Groq key) in .env or the host's settings")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen2.5:3b") if USE_OLLAMA else LLM_MODEL
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")  # multilingual: works for English and French
KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "24h")  # keep models in memory -> no reload delay
# 350 cut French answers off mid-sentence (French runs ~25% longer); the prompt still asks for ~120 words.
MAX_ANSWER_TOKENS = int(os.getenv("MAX_ANSWER_TOKENS", "500"))
CONTEXT_TOKENS = int(os.getenv("CONTEXT_TOKENS", "4096"))

MAX_CHUNK_CHARS = 1200
HISTORY_TURNS = 6  # only the last N messages are sent, to keep prompts short (= faster on CPU)
TOP_K = 5

# Off-topic guard. A question whose best knowledge match scores below this is not about Sidick
# (celebrities, weather, homework, "write me code"...) and is declined without calling the LLM.
# Measured with bge-m3 on this knowledge: questions about Sidick scored 0.476-0.665,
# clearly off-topic ones 0.351-0.460. Re-measure if you change EMBED_MODEL.
OFFTOPIC_THRESHOLD = float(os.getenv("OFFTOPIC_THRESHOLD", "0.465"))

# The knowledge block is called "WHAT YOU KNOW" (not "CONTEXT"): the 3B model kept telling visitors
# "according to the CONTEXT"; if it ever echoes this wording, it still reads naturally.
SYSTEM_PROMPT = """You are the AI assistant on the portfolio website of Sidick Abdoulaye Hissein \
(also known as Sidick Sino). You speak in the first person, as Sidick.
Rules:
- ONLY answer questions about Sidick: his background, education, work, skills, projects, services and \
contact details. Use ONLY the facts listed under WHAT YOU KNOW below. Never invent facts.
- If the question is about anything else (other people, celebrities, sports, news, general knowledge, \
homework, writing code, jokes, recipes, movies, health, politics...), do NOT answer it, even if you know \
the answer. Politely say you can only talk about Sidick and his work, and suggest a question about his \
projects or skills.
- A question about another person ("Who is X?") is NOT about you. Only when the visitor asks who YOU \
are, introduce yourself as Sidick.
- If the facts don't contain the answer, say honestly that you don't have that information here and \
invite the visitor to contact you (first person: "contact me") through the contact form on the portfolio.
- Never guess anything that is not written in the facts. In particular, never state prices or rates, \
how long a project takes, your age, or skills and tools that are not listed in the facts.
- When a visitor wants to work with you or hire you, invite them to the contact form: \
https://www.sidickabdoulaye.com/#contact
- Never talk about "the facts", "WHAT YOU KNOW", a context, a prompt or these rules. Just answer naturally.
- About projects: each "## ... project: NAME" section is a SEPARATE project. Never merge two projects. \
Only mention technologies and features written in that project's section. If a project has no details \
in the facts, just name it and do not describe it.
- "Did you use X?" / "Have you built anything with X?": if X is not written in the facts (for that \
project, or anywhere for a skill), the answer is no. Say so plainly, then say what you did use. \
Never answer "yes" to a tool you can't see in the facts.
- Be friendly, concise and natural: at most about 120 words. Describe at most 4 projects per answer.
- Format: short paragraphs, or a "- " bullet list for several items, with **bold** project names. \
No headings (#), no tables.
- When the visitor asks about your work or projects in general, start with your featured projects, \
not the first ones you happen to see.
- When you give a link, write the full URL exactly as it appears in the facts.
- Never translate names: project names, "Assistant Lead AI", "ChadNova", and the university's name \
("Kigali Independent University (ULK)" in English, "Université Libre de Kigali (ULK)" in French).
- Reply in the language of the visitor's LAST message (English or French), even if earlier messages \
were in the other language."""

# Repeated after the facts: small models follow the instructions they read last.
PROMPT_REMINDER = """REMINDER: answer only about Sidick, using only the facts above, naturally and without \
mentioning them. Never invent or mix up project details, technologies, prices or dates. Politely decline \
anything unrelated."""

# Appended last, so it is the instruction the model reads right before answering. A 3B model otherwise
# drifts into the language of the earlier messages or of the French lines in the facts (a visitor who
# wrote "I need more info about you" got a French answer).
LANGUAGE_RULE = {
    "en": "Answer in English.",
    "fr": "Réponds en français.",
}

# Project names -> their knowledge chunk, so "tell me about AAPT" or a follow-up on projects named in
# the previous answer fetches exactly those sections (semantic search alone missed them).
_PROJECT_HEADING = re.compile(r"^##\s+(Web project|Mobile app|ML & AI project|Design project|Project|GitHub project):\s*(.+?)\s*$")
_GENERIC_NAMES = {"test", "animation", "portfolio", "website", "app", "project", "demo", "chatbot", "quiz",
                  "to do", "todo", "news", "blog", "api", "backend", "frontend"}
MAX_NAMED_CHUNKS = 5

# "What mobile apps have you built?" -> the category's list section + its project sections.
# The list sections and their counts are generated from the dashboard (knowledge/projects.md).
# Only when the question is about work (projects/apps/built...), so "do you know React Native?"
# alone doesn't pull in every app.
_WORK_WORDS = re.compile(r"\b(projects?|projets?|apps?|applications?|built|build|made|developed|created|"
                         r"réalis\w*|fait\w*|cré\w*|développ\w*|works?|travaux|portfolio|websites?|sites?)\b", re.I)
_CATEGORIES = [
    (re.compile(r"\b(mobiles?|ios|android|react native|expo)\b", re.I), "Mobile app", "list of my mobile apps"),
    (re.compile(r"\b(web|websites?|sites?|web ?apps?)\b", re.I), "Web project", "list of my web projects"),
    (re.compile(r"\b(machine learning|ml|ai|ia|intelligence artificielle|apprentissage automatique|"
                r"data science|predictions?|prédictions?)\b", re.I), "ML & AI project", "list of my ML & AI projects"),
    (re.compile(r"\b(designs?|figma|ui|ux)\b", re.I), "Design project", "list of my design projects"),
]
# General questions ("what have you built?", "I need info") get the overview with the featured list,
# and link requests ("donne-moi les liens") get the section with every link.
_OVERVIEW_TITLE = "Portfolio: overview of my projects"
_LINKS_TITLE = "Portfolio: links to my projects"
_LINK_WORDS = re.compile(r"\b(links?|liens?|urls?|live|demos?|démos?|en ligne|online)\b", re.I)
_VAGUE_INFO = re.compile(r"\b(info|infos|information|informations|details?|détails?)\b", re.I)


def _norm(text):
    return " " + re.sub(r"[\s_\-]+", " ", re.sub(r"[^\w\s\-]", " ", text.lower())).strip() + " "

OFFLINE_PREFIX = "My AI brain is resting right now, but here is what I know about that:\n\n"

# Instant replies that never reach the LLM (cheap on a CPU server, and always on-topic).
CANNED = {
    "en": {
        "offtopic": "I'm Sidick's assistant, so I can only answer questions about Sidick: his skills, "
                    "projects, services and background. Try asking me what I've built, or how to contact me!",
        "greeting": "Hi! 👋 Ask me anything about my skills, projects, services or background.",
        "howareyou": "I'm doing great, thanks for asking! 😊 Ask me anything about my projects, skills or background.",
        "unavailable": "Sorry, I can't answer that properly right now. Please try again in a moment, or reach me "
                       "directly through the contact form: https://www.sidickabdoulaye.com/#contact",
        "thanks": "You're welcome! Anything else you'd like to know about my work?",
        "bye": "Thanks for visiting! Feel free to reach out through the contact form anytime.",
        "ai": "I'm Sidick's AI assistant: I answer on his behalf using information he provided about his "
              "background, skills and projects. For a direct conversation with Sidick, use the contact form: "
              "https://www.sidickabdoulaye.com/#contact",
        "contact": "You can reach me here:\n{card}\nI'd be happy to hear about your project!",
        "business": "Yes, I'm available and open to freelance projects and new opportunities! Pricing and "
                    "timelines depend on each project, so tell me about yours through the contact form and I'll "
                    "get back to you with a quote: https://www.sidickabdoulaye.com/#contact",
        "hire": "Yes, that's what I do! I build web platforms (React, Node.js, Django), cross-platform mobile apps "
                "(React Native, Expo) and UI/UX designs (Figma). Tell me about your project through the contact "
                "form and I'll get back to you: https://www.sidickabdoulaye.com/#contact",
    },
    "fr": {
        "offtopic": "Je suis l'assistant de Sidick, je ne peux donc répondre qu'aux questions sur Sidick : "
                    "ses compétences, ses projets, ses services et son parcours. Demande-moi par exemple "
                    "ce que j'ai réalisé, ou comment me contacter !",
        "greeting": "Salut ! 👋 Pose-moi tes questions sur mes compétences, mes projets, mes services ou mon parcours.",
        "howareyou": "Ça va très bien, merci ! 😊 Pose-moi tes questions sur mes projets, mes compétences ou mon parcours.",
        "unavailable": "Désolé, je ne peux pas bien répondre pour le moment. Réessaie dans un instant, ou écris-moi "
                       "directement via le formulaire de contact : https://www.sidickabdoulaye.com/#contact",
        "thanks": "Avec plaisir ! Tu veux savoir autre chose sur mon travail ?",
        "bye": "Merci de ta visite ! N'hésite pas à me contacter via le formulaire de contact.",
        "ai": "Je suis l'assistant IA de Sidick : je réponds en son nom à partir des informations qu'il a "
              "fournies sur son parcours, ses compétences et ses projets. Pour discuter directement avec Sidick, "
              "utilise le formulaire de contact : https://www.sidickabdoulaye.com/#contact",
        "contact": "Tu peux me joindre ici :\n{card}\nAu plaisir d'échanger sur ton projet !",
        "business": "Oui, je suis disponible pour des projets freelance et de nouvelles opportunités ! Les tarifs "
                    "et les délais dépendent de chaque projet : présente-moi le tien via le formulaire de contact et "
                    "je te répondrai avec un devis : https://www.sidickabdoulaye.com/#contact",
        "hire": "Oui, c'est mon métier ! Je crée des plateformes web (React, Node.js, Django), des applications "
                "mobiles multiplateformes (React Native, Expo) et des designs UI/UX (Figma). Présente-moi ton projet "
                "via le formulaire de contact et je te répondrai : https://www.sidickabdoulaye.com/#contact",
    },
}
CARD_LABELS = {
    "en": {"email": "Email", "phone": "Phone", "form": "Contact form", "linkedin": "LinkedIn",
           "github": "GitHub", "x": "X (Twitter)"},
    "fr": {"email": "E-mail", "phone": "Téléphone", "form": "Formulaire de contact", "linkedin": "LinkedIn",
           "github": "GitHub", "x": "X (Twitter)"},
}


def parse_contact(text):
    """Contact details, read from the knowledge files (single source of truth) — never generated."""
    def find(pattern):
        m = re.search(pattern, text, re.I | re.M)
        return m.group(1).strip() if m else None
    return {
        "email": find(r"([\w.+-]+@[\w-]+\.[\w.]+)"),
        "phone": find(r"^Phone:\s*(\+?[\d ][\d ]{6,}\d)"),
        "form": find(r"(https://[^\s]+/#contact)"),
        "linkedin": find(r"(https://(?:www\.)?linkedin\.com/in/[^\s)]+)"),
        "github": find(r"^GitHub:\s*(https://github\.com/[^\s/)]+)"),
        "x": find(r"^X:\s*(@\w+)"),
    }


# Instant, exact answers for questions that must never be improvised.
_AI_QUESTION = re.compile(
    r"\b(are you (a |an )?(real|human|bot|robot|ai|a\.i\.|person|chat ?bot|machine)|are you really sidick|"
    r"is this (really )?sidick|what (ai |language )?model|which model|who (made|built|created|programmed) you|"
    r"es[- ]tu (un |une )?(vrai|humain|robot|bot|ia|chatbot|machine|vraiment sidick)|"
    r"qui t'a (créé|cree|fait|programmé)|quel modèle)\b", re.I)
_CONTACT_QUESTION = re.compile(
    r"\b(contact|reach|e-?mail|mail|phone|number|num[ée]ro|t[ée]l[ée]phone|whatsapp|call you|appeler|joindre|"
    r"contacter|linkedin|github|twitter|socials?|social media|réseaux sociaux|hire|recruit|embaucher|recruter)\b"
    r"|\bx\b(?= ?(account|handle|profile|or twitter)|\?)", re.I)
# ...but not "how many repositories on GitHub?" or "where is the code for Wallet on GitHub?"
_NOT_CONTACT = re.compile(r"repositor|repos\b|projects?|projets?|code|how many|combien|stars?", re.I)
# Price / timeline / availability: never improvised (the 3B model invented "I don't charge" and
# "a few weeks"). Sidick is available; pricing is discussed via the contact form (his choice, 2026-11-20).
_BUSINESS_QUESTION = re.compile(
    r"\b(price|prices|pricing|cost|costs|charge|charges|rates?|budget|quote|fees?|how much|tarifs?|prix|coûte?|"
    r"co[uû]t|devis|combien (ça|ca|tu)|how long|timeline|deadline|délais?|combien de temps|available|"
    r"availability|disponible|disponibilité|freelance)\b", re.I)
# "Can you build a website for me?" -> yes + services + contact form (it forgot the contact link).
_HIRE_REQUEST = re.compile(
    r"\b(build|make|create|develop|design|code|créer|creer|faire|développer|developper|concevoir)\b.*"
    r"\b(for me|for my|for our|for us|pour moi|pour mon|pour ma|pour mes|pour nous|pour notre)\b", re.I)
# Prompt-injection / jailbreak attempts: decline, never play along.
_INJECTION = re.compile(
    r"system prompt|your (instructions|rules|prompt)|tes (instructions|règles)|ton prompt|"
    r"ignore (all |your |the |any )?(previous |above |prior )?(instructions|rules|prompts?)|"
    r"forget (all |your |the )?(previous )?(rules|instructions)|oublie (tes|les) (règles|instructions)|"
    r"you are now|tu es maintenant|act as|pretend (to be|you are)|fais comme si|jailbreak|developer mode|"
    r"\bDAN\b", re.I)
# Topics that are never about Sidick (unless the visitor asks about Sidick's own: "your movie app").
_OFFTOPIC_TOPIC = re.compile(
    r"^(?!.*\b(your|yours|ton|ta|tes|votre|vos|sidick)\b).*\b(homework|devoirs?|movies?|films?|series|séries|"
    r"weather|météo|meteo|medicine|médicaments?|headache|mal de tête|symptoms?|symptômes|recipes?|recettes?|"
    r"lyrics|paroles|horoscope|bitcoin|crypto|stocks?|bourse|politics?|politique|president|président|"
    r"elections?|élections?|religion|dating|girlfriend|boyfriend)\b", re.I)
_SMALL_TALK = [
    ("greeting", re.compile(r"^(hi+|hello|hey|yo|hiya|good (morning|afternoon|evening)|bonjour|bonsoir|salut|"
                            r"coucou|slt|cc)\b[\s!.,?👋🙂😊]*(sidick|there|toi)?[\s!.,?👋🙂😊]*$", re.I)),
    ("howareyou", re.compile(r"^(hi|hello|hey|salut|bonjour)?[\s,!]*(how are you|how r u|how are u|how's it going|"
                             r"how is it going|how do you do|what's up|whats up|sup|ça va|ca va|comment vas[- ]tu|"
                             r"comment ça va|comment ca va|tu vas bien|vous allez bien)\b[\s!.,?🙂😊]*(sidick|toi)?[\s!.,?🙂😊]*$", re.I)),
    ("thanks", re.compile(r"^(thanks?( you)?( so much| a lot)?|thx|ty|merci( beaucoup| bien)?|"
                          r"great|cool|nice|super|ok(ay)?|d'accord)\b[\s!.,🙏👍🙂😊]*$", re.I)),
    ("bye", re.compile(r"^(bye|goodbye|see you|ciao|au revoir|à bientôt|a\+|bonne (journée|soirée))\b[\s!.,👋]*$", re.I)),
    # "lol", "haha", "mdr", or only emoji / punctuation: a friendly nudge, not a full biography
    ("greeting", re.compile(r"^((lol|lmao|haha+|hehe+|mdr|ptdr|xd)\b|[\W_]+$)[\s\W]*$", re.I)),
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
# "Did you use TensorFlow for the Pima app?", "Have you built anything with Flutter?". The 3B model answers
# "yes" to these whatever the facts say, so they're answered from the knowledge itself (see used_tool_reply).
_USED_TOOL_QUESTION = re.compile(
    r"\b(did you (use|build|make|code|work)|have you (ever )?(used|built|made|worked)|do you (use|know|work with)|"
    r"was it (built|made|written)|is it (built|made|written)|as[- ]tu (utilisé|fait|codé|travaillé)|"
    r"tu as (utilisé|fait|codé|travaillé)|connais[- ]tu|tu connais|tu maîtrises|utilises[- ]tu)\b", re.I)
# Tools a visitor may ask about. Anything also written in the knowledge counts as known (added at load).
_TECH_VOCAB = {
    "flutter", "dart", "kotlin", "swift", "swiftui", "java", "c#", "c++", "go", "golang", "rust", "ruby", "rails",
    "php", "laravel", "symfony", "vue", "vue.js", "angular", "svelte", "nuxt", "next.js", "nextjs", "gatsby",
    "spring", "flask", "fastapi", "django", "pytorch", "keras", "tensorflow", "scikit-learn", "xgboost", "opencv",
    "hugging face", "langchain", "postgresql", "postgres", "redis", "graphql", "firebase", "aws", "azure", "gcp",
    "kubernetes", "docker", "unity", "wordpress", "shopify", "redux", "jquery", "bootstrap", "sass", "webflow",
}


def _tech_mentions(text, vocab):
    t = " " + text.lower() + " "
    return [v for v in vocab if re.search(r"(?<![\w.#+])" + re.escape(v) + r"(?![\w#+])", t)]


_THIRD_PARTY = re.compile(r"^\s*(who|qui|c'est qui)\s+(is|was|are|est|était|sont)\b", re.I)
_FOLLOW_UP_WORDS = 6  # questions this short, mid-conversation, are searched with the previous one
# ...as are questions that point back at the conversation ("more details on these projects").
_REFERS_BACK = re.compile(r"\b(these|those|them|they|it|its|this one|that one|more|the projects?|"
                          r"ces|ceux|celles|eux|ça|cela|celui|celle|plus de|davantage|les projets?)\b", re.I)

# Questions addressed to Sidick ("which university do YOU attend?", "where does Sidick study?") are on-topic
# even when the knowledge has no answer: they go to the LLM, which says it doesn't know, instead of the
# "I can only talk about Sidick" reply. ("Who is Messi?" has none of these words, so it's still declined.)
_ADDRESSES_SIDICK = re.compile(r"\b(you|your|yours|yourself|sidick|sino|tu|toi|ton|ta|tes|t'|vous|votre|vos)\b", re.I)

_ENGLISH_HINTS = re.compile(
    r"\b(the|what|who|whats|you|your|yours|is|are|i|i'm|me|my|need|want|more|about|how|can|could|tell|give|"
    r"show|do|does|have|has|links?|projects?|please|pls|info|where|when|which|why|with|for|and|of|to|it|"
    r"hi|hello|hey|thanks?|thx|bye|whos|who's|ur|yes|no|okay)\b", re.I)
_FRENCH_HINTS = re.compile(
    r"[àâçéèêëîïôûùüÿœ]|\b(je|tu|il|nous|vous|est|es|suis|qui|quoi|quel|quels|quelle|quelles|comment|pourquoi|"
    r"où|combien|le|la|les|des|du|une|un|et|ton|ta|tes|mon|ma|mes|avec|pour|dans|sur|peux|bonjour|salut|"
    r"merci|parle|moi|toi)\b", re.I)


def detect_lang(text, fallback="en"):
    """English or French, from the words of this one message. Used for the canned replies and to tell the
    model which language to answer in. A message with no clear signal ("SinoCoffee?", "ok") falls back
    to `fallback`: the site's language, sent by the widget."""
    fr = len(_FRENCH_HINTS.findall(text))
    en = len(_ENGLISH_HINTS.findall(text))
    if fr > en:
        return "fr"
    if en > fr:
        return "en"
    return fallback if fallback in ("en", "fr") else "en"


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

# Words too common to say what a question is about (they made long GitHub READMEs win every search).
_STOPWORDS = set("""a an the and or but if of to in on at for from by with about as is are was were be been am do does did
have has had will would can could should may might i me my mine you your yours yourself he she it its we our they
them their this that these those what which who whom whose when where why how all any some more most much many
very just also not no yes so than too there here please tell give show know want need get got like
le la les un une des du de d l et ou mais si à au aux en dans sur pour par avec sans sous est es suis sont être
ai as a avez ont avoir je j tu toi te t il elle on nous vous ils elles me m mon ma mes ton ta tes son sa ses notre
votre vos leur leurs ce cet cette ces qui que qu quoi quel quelle quels quelles où quand comment pourquoi combien
ne pas plus très aussi bien moi lui y donne donner dis montre parle""".split())


def _stem(word):
    """Tiny English/French suffix stripper: "studies"/"study"/"studying" all become "stud"."""
    for suffix in ("ations", "ation", "ements", "ement", "ities", "ity", "ies", "ing", "ers", "er", "ed", "es", "s", "y"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def check_api_model():
    """At startup in API mode, ask the API which models this key can use and log it, so a wrong LLM_MODEL
    shows up in the logs right away (Groq answers 404 for a model the account can't use)."""
    try:
        r = requests.get(f"{LLM_BASE_URL}/models", headers={"Authorization": f"Bearer {LLM_API_KEY}"}, timeout=15)
        if r.status_code in (401, 403):
            print("[warn] the API rejected LLM_API_KEY (HTTP %d): check the key" % r.status_code, flush=True)
            return
        r.raise_for_status()
        # Gemini lists "models/gemini-2.5-flash" but is called with "gemini-2.5-flash".
        ids = sorted(m.get("id", "").removeprefix("models/") for m in r.json().get("data", []))
    except (requests.RequestException, ValueError) as e:
        print(f"[warn] could not list the API's models: {type(e).__name__}: {str(e)[:200]}", flush=True)
        return
    if LLM_MODEL in ids:
        print(f"API model OK: {LLM_MODEL}", flush=True)
    else:
        print(f"[warn] LLM_MODEL={LLM_MODEL!r} is not available to this key. Set LLM_MODEL to one of: "
              + ", ".join(ids), flush=True)


def _tokens(text):
    return [_stem(w) for w in re.findall(r"\w+", text.lower()) if w not in _STOPWORDS and len(w) > 1]


class BM25:
    """Keyword ranking for API mode (no embedding model on the server). Standard Okapi BM25."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(_tokens(d)) for d in docs]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=np.float32)
        self.avg = float(self.lengths.mean()) if len(docs) else 1.0
        df = Counter(t for d in self.docs for t in d)
        n = len(docs)
        self.idf = {t: float(np.log(1 + (n - f + 0.5) / (f + 0.5))) for t, f in df.items()}

    def scores(self, query):
        out = np.zeros(len(self.docs), dtype=np.float32)
        for term in set(_tokens(query)):
            idf = self.idf.get(term)
            if not idf:
                continue
            for i, d in enumerate(self.docs):
                tf = d.get(term)
                if tf:
                    out[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg))
        return out


class SidickBot:
    def __init__(self):
        self.chunks = load_chunks()
        if USE_OLLAMA:
            ensure_models()
            self.vectors = self._chunk_vectors()
        else:
            check_api_model()
            self.vectors = None
            self.bm25 = BM25(self.chunks)
            # Sidick's own pages (profile, portfolio, projects) outrank raw GitHub READMEs.
            self.keyword_weight = np.array([0.5 if c.startswith("## GitHub project:") else 1.0 for c in self.chunks],
                                           dtype=np.float32)
        self.cache = {}  # repeated first questions (e.g. example buttons) are answered instantly
        self.project_index = self._build_project_index()
        self.contact = parse_contact("\n".join(self.chunks))
        self.category_chunks = [
            (pattern, [i for i, c in enumerate(self.chunks) if c.startswith(f"## Portfolio: {list_title}")]
             + [i for i, c in enumerate(self.chunks) if c.startswith(f"## {prefix}")])
            for pattern, prefix, list_title in _CATEGORIES
        ]
        self.overview_chunk = next((i for i, c in enumerate(self.chunks) if c.startswith(f"## {_OVERVIEW_TITLE}")), None)
        listed = re.findall(r"^(?:Tech|Data Science & Visualization|Frontend Development|Mobile & Design|"
                            r"Backend Development):\s*(.+)$", "\n".join(self.chunks), re.M)
        names = [t.strip(" .") for line in listed for t in line.split(",") if t.strip(" .")]
        self.tech_display = {n.lower(): n for n in names}  # "tensorflow" -> "TensorFlow"
        self.tech_vocab = _TECH_VOCAB | set(self.tech_display)
        # First word of each portfolio project's name, when it names only one project ("pima", "wallet").
        firsts = {}
        for name, (i, curated) in self.project_index.items():
            if curated and len(name.split()[0]) >= 4:
                firsts.setdefault(name.split()[0], set()).add(i)
        self.short_names = {w: next(iter(ids)) for w, ids in firsts.items() if len(ids) == 1}
        self.links_chunk = next((i for i, c in enumerate(self.chunks) if c.startswith(f"## {_LINKS_TITLE}")), None)
        search = f"embeddings: {EMBED_MODEL}" if USE_OLLAMA else "search: keywords (BM25)"
        print(f"Loaded {len(self.chunks)} knowledge chunks, {len(self.project_index)} project names. "
              f"Chat: {LLM_PROVIDER}/{CHAT_MODEL}, {search}", flush=True)

    def contact_card(self, lang):
        labels = CARD_LABELS[lang]
        return "\n".join(f"- **{labels[k]}:** {v}" for k, v in self.contact.items() if v)

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
        if _LINK_WORDS.search(text) and self.links_chunk is not None:
            picked.insert(0, self.links_chunk)
        if _WORK_WORDS.search(text) or _VAGUE_INFO.search(text):
            categories = [indexes for pattern, indexes in self.category_chunks if pattern.search(text)]
            for indexes in categories:
                picked += indexes
            if not categories and self.overview_chunk is not None:  # "what have you built?"
                picked.append(self.overview_chunk)
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
        """Similarity of the question to every chunk. Second value: True if semantic (Ollama up).

        Keyword scores are not on the same scale, so the semantic off-topic guard is skipped for them;
        the regex guards and the prompt still decline off-topic questions."""
        if not USE_OLLAMA:
            return self.bm25.scores(question) * self.keyword_weight, False
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

        Returns (kind, retrieved): kind is "greeting"/"thanks"/"bye"/"offtopic"/"ai"/"contact" for an
        instant canned reply, or None to answer with the LLM using `retrieved` (top chunks).
        """
        for kind, pattern in _SMALL_TALK:
            if pattern.match(question):
                return kind, []
        if _INJECTION.search(question):
            print(f"[guard] prompt injection: {question[:80]!r}", flush=True)
            return "offtopic", []
        if _AI_QUESTION.search(question):
            return "ai", []
        if _CONTACT_QUESTION.search(question) and not _NOT_CONTACT.search(question):
            return "contact", []
        # ...unless it names a project ("is SinoCoffee available online?", "how long did Wallet take?")
        names_project = any(name in _norm(question) for name in self.project_index)
        if _BUSINESS_QUESTION.search(question) and not names_project:
            return "business", []
        if _HIRE_REQUEST.search(question):
            return "hire", []
        if _OFFTOPIC_REQUEST.match(question) or _OFFTOPIC_TOPIC.match(question):
            print(f"[guard] off-topic request: {question[:80]!r}", flush=True)
            return "offtopic", []

        scores, semantic = self._scores(question)
        named = self.mentioned_projects(question)
        # "Who is Messi?" is a new subject, not a follow-up, even right after a question about projects:
        # a question about another person that doesn't match the knowledge on its own is declined here,
        # before the follow-up logic can borrow the previous question's (on-topic) score.
        if (semantic and not named and _THIRD_PARTY.match(question) and not _ADDRESSES_SIDICK.search(question)
                and scores.max() < OFFTOPIC_THRESHOLD):
            print(f"[guard] off-topic person ({scores.max():.3f}): {question[:80]!r}", flush=True)
            return "offtopic", []
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

    def used_tool_reply(self, question, lang):
        """Exact yes/no for "did you use X (for project P)?", or None to let the model answer.

        Only answers "no": a tool that is in the facts goes to the model, which then has the right section."""
        if not _USED_TOOL_QUESTION.search(question):
            return None
        tools = _tech_mentions(question, self.tech_vocab)
        if not tools:
            return None
        named = [i for i in self.mentioned_projects(question) if _PROJECT_HEADING.match(self.chunks[i].split("\n", 1)[0])]
        if not named:  # "the Pima app" for "Pima Diabetes Prediction App"
            t = _norm(question)
            named = [i for word, i in self.short_names.items() if f" {word} " in t]
        scope = self.chunks[named[0]] if named else "\n".join(self.chunks)
        missing = [t for t in tools if not _tech_mentions(scope, [t])]
        if not missing:
            return None
        tool = self.tech_display.get(missing[0]) or (missing[0].upper() if len(missing[0]) <= 3 else missing[0].title())
        if named:
            m = _PROJECT_HEADING.match(self.chunks[named[0]].split("\n", 1)[0])
            stack = re.search(r"^Tech:\s*(.+?)\.?$", self.chunks[named[0]], re.M)
            used = stack.group(1) if stack else None
            if lang == "fr":
                return (f"Non, je n'ai pas utilisé {tool} pour **{m.group(2)}**."
                        + (f" Pour ce projet, j'ai utilisé : {used}." if used else ""))
            return (f"No, I didn't use {tool} for **{m.group(2)}**."
                    + (f" For that project I used {used}." if used else ""))
        if lang == "fr":
            return (f"Non, je n'ai pas encore travaillé avec {tool}. Pour le mobile j'utilise React Native et Expo, "
                    "pour le web React, Node.js et Django, et pour la data science Python, Scikit-learn et TensorFlow.")
        return (f"No, I haven't worked with {tool} yet. For mobile I use React Native and Expo, for the web React, "
                "Node.js and Django, and for data science Python, Scikit-learn and TensorFlow.")

    def _keyword_scores(self, question):
        words = {w for w in re.findall(r"\w{3,}", question.lower())}
        return np.array([sum(w in c.lower() for w in words) for c in self.chunks], dtype=np.float32)

    def build_system_prompt(self, question, retrieved=None, lang="en"):
        # Always include the identity chunks so "Who are you?" works even when similarity is weak.
        chunks = [c for c, _ in (retrieved if retrieved is not None else self.retrieve(question))]
        always = [c for c in self.chunks[:2] if c not in chunks]
        # Name the projects that actually have details here: a 3B model otherwise "fills in" others
        # it saw in the conversation (e.g. inventing a tech stack for a project not in the facts).
        names = [m.group(2) for c in chunks if (m := _PROJECT_HEADING.match(c.split("\n", 1)[0]))]
        names = list(dict.fromkeys(names))
        projects_note = (
            f"\nProjects described above: {', '.join(names)}. Only describe these. "
            "If the visitor asked about other projects, just list their names and invite them to ask about one."
            if names else ""
        )
        return (f"{SYSTEM_PROMPT}\n\nWHAT YOU KNOW:\n" + "\n\n".join(always + chunks)
                + f"\n\n{PROMPT_REMINDER}{projects_note}\n{LANGUAGE_RULE[lang]}")

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

    def offline_answer(self, question, lang="en"):
        """Answer straight from the knowledge when the model can't be reached. Only when something actually
        matches: quoting a random section ("How are you?" got an e-commerce project) is worse than saying so."""
        best = [(c, score) for c, score in self.retrieve(question, k=2) if score > 0]
        if not best:
            return CANNED[lang]["unavailable"]
        return OFFLINE_PREFIX + "\n\n".join(re.sub(r"^#+\s*", "", c, flags=re.M) for c, _ in best)

    def stream_answer(self, question, history=None, page_lang=None):
        """Yield the answer piece by piece (so visitors see text appear while a CPU server thinks).

        page_lang: the site's language ("en"/"fr"), only used when the message itself is ambiguous."""
        question = _extract_text(question).strip()
        lang = detect_lang(question, page_lang or "en")
        key = f"{lang}:{question.lower()}"
        if not history and key in self.cache:
            yield self.cache[key]
            return

        past = self._history_messages(history)
        kind, retrieved = self.check_scope(question, past)
        if kind:  # small talk, off-topic, contact, "are you a bot?": instant reply, no LLM call
            yield CANNED[lang][kind].format(card=self.contact_card(lang))
            return

        exact = self.used_tool_reply(question, lang)
        if exact:
            yield exact
            return

        messages = [{"role": "system", "content": self.build_system_prompt(question, retrieved, lang)}]
        messages += past
        # The language reminder also rides on the question itself: it's the last thing the model reads.
        messages.append({"role": "user", "content": f"{question}\n\n({LANGUAGE_RULE[lang]})"})

        parts = []
        try:
            stream = self._stream_ollama(messages) if USE_OLLAMA else self._stream_openai(messages)
            for piece in stream:
                parts.append(piece)
                yield piece
        except (requests.RequestException, RuntimeError, ValueError) as e:
            hint = ""
            if isinstance(e, requests.HTTPError) and e.response is not None and e.response.status_code in (401, 403):
                hint = " -> the API rejected the key: check LLM_API_KEY"
            elif isinstance(e, requests.HTTPError) and e.response is not None and e.response.status_code == 404:
                hint = " -> unknown model or URL: check LLM_MODEL and LLM_BASE_URL"
            detail = ""
            if isinstance(e, requests.HTTPError) and e.response is not None:
                detail = f" | API said: {e.response.text[:300]}"
            print(f"[warn] {LLM_PROVIDER} failed: {type(e).__name__}: {str(e)[:200]}{hint}{detail}", flush=True)
            if not parts:  # nothing sent yet -> fall back to a retrieval-only answer
                yield self.offline_answer(question, lang)
            return

        text = "".join(parts)
        if text and not history:
            self.cache[key] = text

    @staticmethod
    def _stream_ollama(messages):
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
                    yield piece
                if data.get("done"):
                    break

    @staticmethod
    def _stream_openai(messages):
        """OpenAI-compatible streaming (Groq, xAI...): server-sent events ending with "data: [DONE]"."""
        payload = {"model": LLM_MODEL, "messages": messages, "stream": True,
                   "max_tokens": MAX_ANSWER_TOKENS, "temperature": 0.2}
        if "gpt-oss" in LLM_MODEL or "gemini-2.5" in LLM_MODEL or "gemini-3" in LLM_MODEL:
            # Reasoning models (Groq's gpt-oss, Gemini 2.5+) spend tokens "thinking" before they answer,
            # and those count toward max_tokens: keep the thinking short and leave room for the answer.
            payload.update(reasoning_effort="low", max_tokens=MAX_ANSWER_TOKENS + 700)
        with requests.post(
                f"{LLM_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {LLM_API_KEY}"},
                json=payload,
                stream=True,
                timeout=(10, 60),
        ) as r:
            if r.status_code == 429:
                raise RuntimeError("rate limited by the API (free-tier limit reached)")
            if r.status_code >= 400:
                r.content  # read the error body so the log can show the API's message  # noqa: B018
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                choices = event.get("choices") or [{}]
                piece = (choices[0].get("delta") or {}).get("content")
                if piece:
                    yield piece

    def answer(self, question, history=None, page_lang=None):
        return "".join(self.stream_answer(question, history, page_lang))
