# Sidick-ChatBot

A personal "AI twin" for my portfolio: a small RAG chatbot (sentence-transformers retrieval + Gemini) with a Gradio UI.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your GEMINI_API_KEY (and optionally GITHUB_TOKEN)
```

## Knowledge

Everything in `knowledge/` is used: `.md` (split per `## ` section), `.txt` and `.pdf` (e.g. LinkedIn → More → Save to PDF).
`profile.md` is always loaded first, and its opening sections are sent with every question.

Pull all public GitHub repos (description, language, topics, README) into `knowledge/github.md`:

```bash
python ingest_github.py
```

Without `GITHUB_TOKEN` GitHub allows 60 requests/hour, so some READMEs may be skipped.

## Run

```bash
python app.py
```

Restart after changing anything in `knowledge/`.

## Gemini quota

Models in `CHAT_MODELS` (default `gemini-flash-latest,gemini-flash-lite-latest,gemini-2.5-flash`) are tried in order when one hits its quota.
If all fail, the bot answers directly from the best-matching knowledge chunks. Repeated first questions are cached.
