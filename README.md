# Sidick-ChatBot

A personal AI assistant for my portfolio (https://sidick.vercel.app). It's a small RAG chatbot that runs fully on
**Ollama**, with no external AI API and no quota.

```
Portfolio (Vercel)  ──HTTPS──▶  Caddy  ──▶  server.py (FastAPI)  ──▶  Ollama
widget/sidick-widget.js         (auto-HTTPS)   RAG over knowledge/      qwen2.5:3b + bge-m3
```

- `knowledge/`: what the bot knows (`profile.md`, `portfolio.md`, `github.md`, plus any `.txt`/`.pdf`).
- `chatbot.py`: chunking, Ollama embeddings (`bge-m3`, multilingual), retrieval, and the streamed answer from `qwen2.5:3b`.
  If Ollama is down, it answers straight from the best-matching knowledge.
  Off-topic questions ("Who is Messi?", "write me code", jokes) and small talk get an instant
  canned reply without calling the model; tune `OFFTOPIC_THRESHOLD` in `.env` if needed.
- `server.py`: public API (`POST /chat`, `GET /health`) with CORS limited to the portfolio, rate limits and a message length cap.
- `widget/sidick-widget.js`: the chat bubble for the portfolio (plain JS, works in React/Vite/HTML).
- `app.py`: Gradio UI for local testing.

## Run locally (Mac with Ollama)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
python app.py                      # Gradio test UI, or:
uvicorn server:app --port 8000     # the API the widget uses
```

Models are pulled into Ollama automatically if missing. To test the widget against a local page, add that page's
origin (e.g. `http://localhost:5173`) to `ALLOWED_ORIGINS` in `.env`.

## Update the knowledge

```bash
python ingest_github.py            # refresh knowledge/github.md (set GITHUB_TOKEN for all READMEs)
```

Edit `knowledge/*.md` freely. Embeddings are recomputed automatically on the next start.

## Deploy (free: Oracle Cloud Always Free + DuckDNS)

1. Create an Oracle Cloud Always Free **Ampere A1 (Arm)** VM with Ubuntu, then open ports 80 and 443
   (in the VCN security list and on the VM's firewall).
2. Create a free DuckDNS name (e.g. `sidick-bot.duckdns.org`) pointing at the VM's public IP.
3. On the VM: install Docker, copy this project, then:
   ```bash
   cp .env.example .env            # set DOMAIN=your-name.duckdns.org
   docker compose up -d --build    # first start downloads the models (a few minutes)
   curl https://your-name.duckdns.org/health
   ```
4. In the portfolio: copy `widget/sidick-widget.js` to `public/` and add this before `</body>` in `index.html`:
   ```html
   <script src="/sidick-widget.js" data-api="https://your-name.duckdns.org" defer></script>
   ```
   Then redeploy on Vercel.

After editing `knowledge/` on the server: `docker compose restart api`.
