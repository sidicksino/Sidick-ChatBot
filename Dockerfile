FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY chatbot.py server.py ingest_github.py ingest_portfolio.py ./
COPY knowledge ./knowledge
# Only Caddy (same Docker network) can reach this port, so trusting its forwarded client IP is safe.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
