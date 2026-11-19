"""Pull ALL your public GitHub repos (description, language, topics, README) into knowledge/github.md.

Run:  python ingest_github.py
Optional: put GITHUB_TOKEN in .env (no special permissions needed). Without it GitHub allows only
60 requests/hour, which is not enough for ~60 repos (one README call each).
"""
import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
USER = os.getenv("GITHUB_USER", "sidicksino")
INCLUDE_FORKS = os.getenv("INCLUDE_FORKS", "0") == "1"
README_LIMIT = 2500  # characters kept per README

HEADERS = {"Accept": "application/vnd.github+json"}
if os.getenv("GITHUB_TOKEN"):
    HEADERS["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"


# Unedited starter READMEs describe the framework, not the project. Keeping them made the bot
# describe e.g. TchadInfos as "a Learn Expo tutorial".
TEMPLATE_README_MARKERS = re.compile(
    r"This is an Expo project created with|create-expo-app|Getting Started with Create React App|"
    r"This project was bootstrapped with|This template provides a minimal setup to get React working in Vite|"
    r"This is a \[?Next\.js\]? project bootstrapped|create-next-app",
    re.I)


def is_template_readme(text):
    return bool(TEMPLATE_README_MARKERS.search(text))


def clean_readme(text):
    if is_template_readme(text):
        return ""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)        # images / badges
    text = re.sub(r"<[^>]+>", "", text)                      # html tags
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)     # links -> their text
    text = re.sub(r"^#+\s*", "", text, flags=re.M)           # headings -> plain lines (avoid '## ' splits)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:README_LIMIT]


def fetch_repos():
    repos, page = [], 1
    while True:
        r = requests.get(
            f"https://api.github.com/users/{USER}/repos",
            headers=HEADERS,
            params={"per_page": 100, "page": page, "sort": "pushed"},
            timeout=30,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            return repos
        repos += batch
        page += 1


def fetch_readme(name):
    r = requests.get(
        f"https://api.github.com/repos/{USER}/{name}/readme",
        headers={**HEADERS, "Accept": "application/vnd.github.raw+json"},
        timeout=30,
    )
    if r.status_code in (403, 429):
        raise RuntimeError("rate_limited")
    return clean_readme(r.text) if r.status_code == 200 else ""


def main():
    repos = [r for r in fetch_repos() if INCLUDE_FORKS or not r["fork"]]
    print(f"{len(repos)} repositories found for {USER}")
    sections, rate_limited = [], False
    for r in repos:
        readme = ""
        if not rate_limited:
            try:
                readme = fetch_readme(r["name"])
            except RuntimeError:
                rate_limited = True
                print("Rate limited by GitHub: continuing without READMEs. Add GITHUB_TOKEN to .env.")
        lines = [f"## GitHub project: {r['name']}", f"This is one of my GitHub repositories: {r['name']}."]
        if r.get("description"):
            lines.append(f"Description: {r['description']}")
        if r.get("language"):
            lines.append(f"Main language: {r['language']}.")
        if r.get("topics"):
            lines.append("Topics: " + ", ".join(r["topics"]) + ".")
        if r.get("homepage"):
            lines.append(f"Live demo: {r['homepage']}")
        lines.append(f"Code: {r['html_url']} (updated {r['pushed_at'][:10]}, {r['stargazers_count']} stars).")
        if readme:
            lines.append("README:\n" + readme)
        sections.append("\n".join(lines))

    out = Path(__file__).parent / "knowledge" / "github.md"
    out.write_text("# My GitHub repositories\n\n" + "\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
