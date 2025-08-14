#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import hashlib
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin
import requests
import feedparser
import frontmatter
import trafilatura

# Optional .env loader (pip install python-dotenv)
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass

# =============== CONFIG ===============

OUT_DIR         = Path(os.getenv("OUT_DIR", "./drafts"))
STATE_PATH      = Path(os.getenv("STATE_PATH", "./seen_urls.json"))
LANG            = "en-US"
MIN_LEN_CHARS   = 800
MAX_ARTICLES    = 6
MODEL_NAME      = os.getenv("LLM_MODEL", "your-llm-id")
LLM_API_KEY     = os.getenv("LLM_API_KEY", "YOUR_API_KEY")

SOURCES: List[Dict] = [
    # --- Odoo / Community ---
    {"type": "page", "url": "https://www.odoo.com/blog", "label": "Odoo Blog"},
    {"type": "page", "url": "https://odoo-community.org/blog/news-updates-1", "label": "OCA News"},
    # --- ERP / General (English queries) ---
    {"type": "rss",  "url": "https://news.google.com/rss/search?q=Odoo%20ERP&hl=en&gl=US&ceid=US:en", "label": "GoogleNews Odoo"},
    {"type": "rss",  "url": "https://news.google.com/rss/search?q=ERP%20software&hl=en&gl=US&ceid=US:en", "label": "GoogleNews ERP"},
    {"type": "page", "url": "https://news.sap.com", "label": "SAP News"},
    {"type": "page", "url": "https://erpnews.com/", "label": "ERPNews"},
]

CATEGORIES = [
    "news",
    "odoo-functional-guide",
    "odoo-technical-guide",
    "erp-insight"
]

# =============== LLM ROUTER ===============

def call_llm(prompt: str,
             model: str = None,
             api_key: str = None,
             max_tokens: int = 1200,
             temperature: float = 0.7) -> str:
    """
    Router LLM:
    - LLM_PROVIDER: openai | azure | anthropic | ollama
    - Returns English markdown with YAML front matter.
    """
    provider = (os.getenv("LLM_PROVIDER") or "openai").lower()

    if provider == "openai":
        # pip install openai
        from openai import OpenAI
        client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content.strip()

    if provider == "azure":
        # pip install openai
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        # On Azure, "model" is the deployment name
        model = model or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content.strip()

    if provider == "anthropic":
        # pip install anthropic
        import anthropic
        client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        model = model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        parts = []
        for block in resp.content:
            if getattr(block, "type", "") == "text":
                parts.append(block.text)
        return "\n".join(parts).strip()

    if provider == "ollama":
        # Local API (no extra lib)
        url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
        model = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        r = requests.post(url, json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature}
        }, timeout=120)
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or "").strip()

    raise RuntimeError(f"Unsupported LLM_PROVIDER: {provider}")

# =============== FETCH & EXTRACT ===============

def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()

def load_state() -> set:
    if STATE_PATH.exists():
        try:
            return set(json.loads(STATE_PATH.read_text()))
        except Exception:
            return set()
    return set()

def save_state(seen: set) -> None:
    STATE_PATH.write_text(json.dumps(sorted(seen), ensure_ascii=False, indent=2))

def fetch_rss(url: str) -> List[Dict]:
    feed = feedparser.parse(url)
    items = []
    for e in feed.entries:
        link = getattr(e, "link", None) or getattr(e, "id", None)
        if not link:
            continue
        title = getattr(e, "title", "") or "(untitled)"
        published = getattr(e, "published", "") or ""
        items.append({"title": title, "url": link, "published": published})
    return items

def fetch_page_listing(url: str) -> List[Dict]:
    """
    Fetch an index page and attempt to extract article links.
    Uses a simple regex as a fallback; urljoin for robust absolute URLs.
    """
    try:
        html = requests.get(url, timeout=20).text
    except Exception:
        return []
    links = set(re.findall(r'href="(/blog[^"#?]+|https?://[^"]+)"', html))
    items = []
    for href in links:
        full = href if href.startswith("http") else urljoin(url, href)
        if any(x in full for x in ["#", "/tag/", "/page/", "/feed", "/mailing", "/signup"]):
            continue
        items.append({"title": "(article)", "url": full, "published": ""})
    return list(items)[:20]

def extract_article(url: str) -> Tuple[str, str]:
    """
    Returns (title, text) via trafilatura; empty strings on failure.
    """
    try:
        downloaded = trafilatura.fetch_url(url, timeout=20)
        if not downloaded:
            return "", ""
        meta = trafilatura.extract_metadata(downloaded)
        title = meta.title if meta else ""
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False) or ""
        return title or "", text or ""
    except Exception:
        return "", ""

# =============== CLASSIFY & WRITE ===============

SYSTEM_RULES = f"""
You are an ERP-focused editor. Produce concise, accurate drafts with a practical, professional tone.
Do not invent facts: use only the information provided in the extracts.
Classify the piece into one of: {", ".join(CATEGORIES)}.
Return YAML front matter + Markdown in EN with:
- title
- date (YYYY-MM-DD)
- tags (array)
- category (one of the categories)
- sources (list of URLs)
Then write a 600–900 word article with H2/H3 sections and a final "Sources" section.
"""

def build_prompt(extract: str, url: str, desired_category: Optional[str] = None) -> str:
    today = dt.date.today().isoformat()
    guide = f"""
Context: you are preparing a blog post about Odoo/ERP.
Date: {today}
Source URL: {url}
Suggested category: {desired_category or "auto"}

Extracts (summarize and cite; no wholesale copy-paste):
---
{extract[:4000]}
---

Instructions:
- Language: English.
- If it's news: what happened, why it matters, and impact on Odoo/ERP teams.
- If it's a functional guide: clear operational steps.
- If it's a technical guide: snippets and best practices (Odoo framework, APIs, performance).
- Add 3–5 relevant tags.
- Include a "Sources" section with the link above.
"""
    return SYSTEM_RULES + "\n" + guide

def write_markdown(title: str, body_md: str, url: str, category_hint: Optional[str]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^a-z0-9\-]+", "-", title.lower()).strip("-")[:80] or _hash(title)[:12]
    fname = f"{dt.date.today().isoformat()}-{safe}.md"
    path = OUT_DIR / fname
    post = frontmatter.loads(body_md)
    meta = post.metadata or {}
    meta.setdefault("date", dt.date.today().isoformat())
    meta.setdefault("sources", [url])
    if category_hint and not meta.get("category"):
        meta["category"] = category_hint
    post.metadata = meta
    with open(path, "w", encoding="utf-8") as f:
        frontmatter.dump(post, f)
    return path

# =============== MAIN ===============

def main():
    seen = load_state()
    candidates: List[Dict] = []

    # 1) gather
    for src in SOURCES:
        try:
            if src["type"] == "rss":
                items = fetch_rss(src["url"])
            else:
                items = fetch_page_listing(src["url"])
            for it in items:
                it["source"] = src["label"]
                it["id"] = _hash(it["url"])
        except Exception:
            items = []
        for it in items:
            if it["id"] not in seen:
                candidates.append(it)

    # deduplicate by URL (strip querystring)
    uniq: Dict[str, Dict] = {}
    for c in candidates:
        k = re.sub(r"(\?.*)$", "", c["url"])
        if k not in uniq:
            uniq[k] = c

    queue = list(uniq.values())[:MAX_ARTICLES]

    written = []
    for item in queue:
        url = item["url"]

        # 2) extract
        title, text = extract_article(url)
        if len((text or "")) < MIN_LEN_CHARS:
            continue

        # 3) rough category hint
        hint = None
        url_l = url.lower()
        if any(x in url_l for x in ["blog", "tutorial", "how-to"]):
            hint = "odoo-functional-guide"
        if any(x in url_l for x in ["developer", "tech", "github", "api", "oca"]):
            hint = "odoo-technical-guide"

        # 4) LLM
        prompt = build_prompt(text, url, hint)
        body_md = call_llm(prompt, model=MODEL_NAME, api_key=LLM_API_KEY)

        # 5) write draft file
        fm_title = frontmatter.loads(body_md).metadata.get("title") if body_md else None
        final_title = title or fm_title or "ERP/Odoo Draft"
        path = write_markdown(final_title, body_md, url, hint)
        written.append(str(path))

        # 6) mark as seen
        seen.add(item["id"])
        time.sleep(0.5)

    save_state(seen)

    print(f"[OK] Drafts created: {len(written)}")
    for p in written:
        print(" -", p)

if __name__ == "__main__":
    main()

