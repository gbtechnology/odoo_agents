#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import hashlib
import datetime as dt
import lxml.html as lh
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from readability import Document
from urllib.parse import urlparse, urljoin
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

# WARMUP OLLAMA BEFORE STARTING TO USE IT
def warmup_ollama():
    """Warm up Ollama so the first real request is faster."""
    try:
        _ = call_llm("Say OK.", model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"), max_tokens=32)
    except Exception:
        pass

# =============== HELPERS TO CHECK THE ARTICLES =========================== #

ARTICLE_EXT_BLOCKLIST = (".css", ".js", ".png", ".jpg", ".jpeg", ".webp", ".svg", ".gif", ".ico", ".woff", ".woff2")

def _looks_like_article_odoo(u: str) -> bool:
    p = urlparse(u)
    if p.netloc not in ("www.odoo.com", "odoo.com"):
        return False
    path = p.path or ""
    if "/blog/" not in path:
        return False
    if re.fullmatch(r"/[a-z]{2}_[A-Z]{2}/blog/?", path):
        return False
    last = path.rstrip("/").split("/")[-1]
    return bool(re.search(r"-\d+$", last))

def _looks_like_article_oca(u: str) -> bool:
    p = urlparse(u)
    if p.netloc not in ("odoo-community.org", "www.odoo-community.org"):
        return False
    path = p.path or ""
    if not path.startswith("/blog/news-updates-1/"):
        return False
    last = path.rstrip("/").split("/")[-1]
    return bool(re.search(r"-\d+$", last))

def _is_article_url(base_url: str, candidate: str) -> bool:
    if not candidate.startswith("http"):
        return False
    if candidate.endswith(ARTICLE_EXT_BLOCKLIST):
        return False
    host = urlparse(base_url).netloc
    if "odoo.com" in host:
        return _looks_like_article_odoo(candidate)
    if "odoo-community.org" in host:
        return _looks_like_article_oca(candidate)
    path = urlparse(candidate).path or ""
    if "/blog/" in path:
        last = path.rstrip("/").split("/")[-1]
        return bool(re.search(r"-\d+$", last))
    return False


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
        # Normalize base URL (do NOT include /api in the env var)
        base = (os.getenv("OLLAMA_URL") or "http://127.0.0.1:11434/api/chat").rstrip("/")
        if base.endswith("/api"):
            base = base[:-4]
        model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

        chat_url = f"{base}/api/chat"
        generate_url = f"{base}/api/generate"

        # Read tuning from env (with safe defaults)
        num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "3072"))
        num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", str(max_tokens)))
        num_gpu = int(os.getenv("OLLAMA_NUM_GPU", "0"))
        num_thread_env = int(os.getenv("OLLAMA_NUM_THREAD", "0"))
        keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "5m")
        http_timeout = int(os.getenv("OLLAMA_TIMEOUT", "480"))

        # Common generation options (CPU-only friendly)
        opts = {
            "temperature": float(temperature),
            "top_p": 0.9,
            "repeat_penalty": 1.05,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "num_gpu": num_gpu,  # 0 = CPU only
        }
        if num_thread_env > 0:
            opts["num_thread"] = num_thread_env

        headers = {"Content-Type": "application/json"}

        # Prefer the chat endpoint; include keep_alive to keep the model loaded
        payload_chat = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": opts,
            "keep_alive": keep_alive,
        }

        try:
            r = requests.post(chat_url, json=payload_chat, headers=headers, timeout=http_timeout)
            # Fallback to /api/generate if chat endpoint is unavailable
            if r.status_code in (404, 405):
                payload_gen = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": opts,
                    "keep_alive": keep_alive,
                }
                r = requests.post(generate_url, json=payload_gen, headers=headers, timeout=http_timeout)

            r.raise_for_status()
            data = r.json()

            # Parse response: chat -> message.content, generate -> response
            text = None
            if isinstance(data.get("message"), dict):
                text = data["message"].get("content")
            if not text:
                text = data.get("response")
            if not text:
                raise RuntimeError(f"Ollama returned an empty response: {data}")

            return text.strip()

        except requests.exceptions.ReadTimeout:
            # Retry once with a shorter target length and a longer timeout
            opts["num_predict"] = max(400, num_predict // 2)  # cut length to speed up
            payload_chat["options"] = opts
            r2 = requests.post(chat_url, json=payload_chat, headers=headers, timeout=max(http_timeout, 600))
            r2.raise_for_status()
            data = r2.json()
            text = None
            if isinstance(data.get("message"), dict):
                text = data["message"].get("content")
            if not text:
                text = data.get("response")
            return (text or "").strip()

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
        resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        html = resp.text
    except Exception:
        return []

    try:
        doc = lh.fromstring(html)
    except Exception:
        return []

    found = []
    seen = set()

    for a in doc.xpath("//a[@href]"):
        href = a.get("href", "")
        if not href or href.startswith("mailto:") or href.startswith("tel:"):
            continue
        full = urljoin(url, href)
        full_noq = re.sub(r"(\?.*)$", "", full)
        if full_noq in seen:
            continue
        if _is_article_url(url, full_noq):
            seen.add(full_noq)
            found.append({"title": "(article)", "url": full_noq, "published": ""})

    return found[:20]


def extract_article(url: str) -> Tuple[str, str]:
    """

    """
def extract_article(url: str) -> Tuple[str, str]:
    """
    Returns (title, text) using several strategies:
    - Meta (og:title / twitter:title / <h1> / <title>)
    - Trafilatura
    - Readability-lxml
    - Fallback XPath (Odoo / OCA)
    """
    def _fetch_html(u: str) -> Optional[str]:
        try:
            resp = requests.get(
                u,
                timeout=25,
                headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Safari/537.36"}
            )
            if resp.status_code == 200 and resp.text:
                return resp.text
        except Exception:
            pass
        return None

    def _pick_title(doc: LH.HtmlElement) -> str:
        # 1) og:title
        node = doc.xpath("//meta[@property='og:title'][@content]")
        if node:
            t = (node[0].get("content") or "").strip()
            if t:
                return t
        # 2) twitter:title
        node = doc.xpath("//meta[@name='twitter:title'][@content]")
        if node:
            t = (node[0].get("content") or "").strip()
            if t:
                return t
        # 3) H1
        node = doc.xpath("//h1[normalize-space()]")
        if node:
            t = node[0].text_content().strip()
            if t:
                return t
        # 4) <title>
        node = doc.xpath("//title")
        if node:
            t = (node[0].text or "").strip()
            if t:
                return t
        return ""

    def _clean_text(s: str) -> str:
        s = re.sub(r"\r\n?", "\n", s)
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s.strip()

    def _extract_with_trafilatura(html: str) -> str:
        try:
            txt = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
                favor_precision=True,
                with_metadata=False,
            )
            return (txt or "").strip()
        except Exception:
            return ""

    def _extract_with_readability(html: str) -> str:
        try:
            doc = Document(html)
            summary_html = doc.summary(html_partial=True)
            if not summary_html:
                return ""
            el = LH.fromstring(summary_html)
            # prendi paragrafi ed eventuali bullet
            parts = []
            for node in el.xpath(".//p[normalize-space()] | .//li[normalize-space()]"):
                parts.append(node.text_content().strip())
            return _clean_text("\n\n".join(parts))
        except Exception:
            return ""

    def _extract_site_specific(base: str, doc: LH.HtmlElement) -> str:
        host = urlparse(base).netloc
        candidates = []

        candidates += doc.xpath("//article")
        candidates += doc.xpath("//div[contains(@class,'blog') and contains(@class,'post')]")
        candidates += doc.xpath("//div[contains(@class,'o_blog') or contains(@class,'o_wblog')]")
        candidates += doc.xpath("//div[contains(@class,'post-content') or contains(@class,'entry-content')]")

        if "odoo-community.org" in host:
            candidates += doc.xpath("//div[contains(@class,'oe_structure') or contains(@class,'s_blog_post')]")

        texts = []
        for c in candidates:
            ps = c.xpath(".//p[normalize-space()] | .//li[normalize-space()]")
            chunk = "\n\n".join(p.text_content().strip() for p in ps)
            chunk = _clean_text(chunk)
            if len(chunk) > 400:
                texts.append(chunk)

        if texts:
            return max(texts, key=len)
        return ""

    html = _fetch_html(url)
    if not html:
        return "", ""

    try:
        doc = LH.fromstring(html)
    except Exception:
        doc = None

    title = ""
    if doc is not None:
        title = _pick_title(doc)

    text = _extract_with_trafilatura(html)

    if len(text) < 600:
        alt = _extract_with_readability(html)
        if len(alt) > len(text):
            text = alt

    if len(text) < 600 and doc is not None:
        alt2 = _extract_site_specific(url, doc)
        if len(alt2) > len(text):
            text = alt2

    if not title:
        try:
            doc_r = Document(html)
            t2 = (doc_r.short_title() or "").strip()
            if t2:
                title = t2
        except Exception:
            pass

    return (title or "").strip(), _clean_text(text)


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
    # Warm up only when using Ollama
    if _current_provider() == "ollama":
        warmup_ollama()
        
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

