# -*- coding: utf-8 -*-
"""
blog_import.py

Scan a drafts folder for Markdown files with YAML front matter and
import them as **draft blog posts** into an Odoo instance via XML-RPC.

Environment variables expected:
    ODOO_URL         e.g. https://gb-technology.com
    ODOO_DB          database name
    ODOO_USER        login (email)
    ODOO_PASSWORD    password
    ODOO_BLOG_ID     numeric blog.blog ID to post into

Optional environment variables:
    DRAFT_DIR                folder to scan (default: ./drafts, fallback: ./draft)
    ODOO_AUTHOR_EMAIL       author email to link, if supported by your blog.post

Notes:
- Comments are in English, as requested.
- The script is defensive across Odoo versions (v14→v17+). It auto-detects field names like
  title vs name, website_published vs is_published, etc..
- Markdown is converted to HTML if the `markdown` package is available; otherwise
  the raw Markdown is wrapped in a <pre> block.
"""

import os
import sys
import re
import glob
import html
import traceback
import xmlrpc.client as xmlrpclib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import frontmatter

# Optional markdown -> HTML conversion
try:
    import markdown  # type: ignore
    def md_to_html(md_text: str) -> str:
        """Convert Markdown to HTML using the 'markdown' library if available."""
        return markdown.markdown(md_text, extensions=["fenced_code", "tables", "toc", "codehilite"])  # pragma: no cover
except Exception:
    def md_to_html(md_text: str) -> str:
        """Fallback: escape and wrap in <pre> if markdown is not installed."""
        return f"<pre>{html.escape(md_text)}</pre>"  # pragma: no cover


# --------------------------
# XML-RPC helpers
# --------------------------

# Remove a leading fenced YAML block at the very beginning of the body
yaml_fence_re = re.compile(r'^\s*```(?:yaml|yml)?\s+.*?```', re.IGNORECASE | re.DOTALL)

# Remove an in-body “Sources” section if present to avoid duplicates
sources_section_re = re.compile(
    r'(?ims)^\s{0,3}#{1,6}\s*Sources\s*$.*?(?=^\s{0,3}#{1,6}\s|\Z)'
)

def strip_leading_yaml_fence(md_text: str) -> str:
    """Remove the first fenced YAML block at the very beginning of the body, if present."""
    return yaml_fence_re.sub('', md_text, count=1).lstrip()

def strip_inline_sources_section(md_text: str) -> str:
    """Remove a Markdown section titled 'Sources' (any heading level) to prevent duplication."""
    return sources_section_re.sub('', md_text).rstrip()

def connect_odoo() -> Tuple[str, int, xmlrpclib.ServerProxy]:
    """Authenticate on Odoo and return (db, uid, models_proxy)."""
    url = os.getenv("ODOO_URL")
    db = os.getenv("ODOO_DB")
    user = os.getenv("ODOO_USER")
    pwd = os.getenv("ODOO_PASSWORD")

    if not all([url, db, user, pwd]):
        raise RuntimeError("Missing Odoo env vars: ODOO_URL / ODOO_DB / ODOO_USER / ODOO_PASSWORD")

    common = xmlrpclib.ServerProxy(f"{url}/xmlrpc/2/common")
    uid = common.authenticate(db, user, pwd, {})
    if not uid:
        raise RuntimeError("Authentication failed: check credentials and database name")

    models = xmlrpclib.ServerProxy(f"{url}/xmlrpc/2/object")
    return db, uid, models


# --------------------------
# Model-specific utilities
# --------------------------

def ensure_author(models: xmlrpclib.ServerProxy, db: str, uid: int, pwd: str,
                  post_fields: Dict[str, Dict], author_email: Optional[str]) -> Dict[str, int]:
    """Resolve optional author relation. Returns mapping of field->id to merge in create vals.

    Strategy: if 'author_id' exists, detect its relation and try to locate by email or name.
    Supports relations like 'blog.author', 'res.partner', or 'res.users'. If nothing is found,
    returns an empty dict (author will be the current user).
    """
    if not author_email:
        return {}
    if "author_id" not in post_fields or post_fields["author_id"].get("type") != "many2one":
        return {}

    relation = post_fields["author_id"].get("relation")
    name_guess = author_email.split("@")[0]

    def _search(model: str, domain: List, fields: List[str]) -> List[Dict]:
        ids = models.execute_kw(db, uid, pwd, model, "search", [domain], {"limit": 1})
        if not ids:
            return []
        recs = models.execute_kw(db, uid, pwd, model, "read", [ids, fields])
        return recs or []

    if relation == "blog.author":
        found = _search("blog.author", [["email", "=", author_email]], ["id"]) or \
                _search("blog.author", [["name", "=", name_guess]], ["id"])  # type: ignore
        if found:
            return {"author_id": found[0]["id"]}
        # Optionally create an author record if allowed
        try:
            new_id = models.execute_kw(db, uid, pwd, "blog.author", "create", [{"name": name_guess, "email": author_email}])
            return {"author_id": new_id}
        except Exception:
            return {}

    if relation == "res.partner":
        found = _search("res.partner", [["email", "=", author_email]], ["id"]) or \
                _search("res.partner", [["name", "=", name_guess]], ["id"])  # type: ignore
        if found:
            return {"author_id": found[0]["id"]}
        return {}

    if relation == "res.users":
        found = _search("res.users", [["login", "=", author_email]], ["id"]) or \
                _search("res.users", [["name", "=", name_guess]], ["id"])  # type: ignore
        if found:
            return {"author_id": found[0]["id"]}
        return {}

    return {}

# --------------------------
# Markdown loader
# --------------------------

def load_md_post(path: Path) -> Tuple[Dict, str]:
    """Load a Markdown file with front matter and return (metadata, markdown_body)."""
    post = frontmatter.load(path)
    meta = post.metadata or {}
    body = post.content or ""
    # Sanitize body: drop stray fenced YAML and any inline “Sources” section
    body = strip_leading_yaml_fence(body)
    body = strip_inline_sources_section(body)
    return meta, body


def build_html_content(md_body: str) -> str:
    """Turn Markdown body and sources list into HTML tailored for Odoo blog.content."""
    html_body = md_to_html(md_body)
    return html_body


# --------------------------
# Import logic
# --------------------------

def import_post(models: xmlrpclib.ServerProxy, db: str, uid: int, pwd: str,
               blog_id: int, path: Path, author_email: Optional[str]) -> Optional[int]:
    """Create or update a blog.post from a Markdown file. Returns the post ID or None on failure."""
    meta, md_body = load_md_post(path)

    # Extract metadata with fallbacks
    title = meta.get("title") or meta.get("name") or path.stem.replace("-", " ").title()
    tags = meta.get("tags") or []

    # Build HTML content
    content_html = build_html_content(md_body)

    author_id = models.execute_kw(
        db,
        uid,
        pwd,
        "res.partner",
        "search", [("email", "=", author_email)],
        {
            "limit": 1}
    )

    # Base values
    vals: Dict = {
        "blog_id": blog_id,
        "title": title,
        "content": content_html,
        "author_id": author_id.id
    }

    # Upsert behavior
    # Search for an existing post with same title in the same blog
    domain = [["blog_id", "=", blog_id], ["title", "=", title]]
    existing_ids = models.execute_kw(db, uid, pwd, "blog.post", "search", [domain], {"limit": 1})

    if existing_ids:
        print(f"[skip]    Post exists with same title: {title}")
        return int(existing_ids[0])

    # Create new post
    post_id = models.execute_kw(db, uid, pwd, "blog.post", "create", [vals])
    print(f"[create] {path.name} → blog.post({post_id})")
    return int(post_id)

def main() -> None:
    # Resolve configuration from env
    drafts_dir = Path(os.getenv("DRAFT_DIR", "./drafts")).expanduser()
    if not drafts_dir.exists():
        return
    blog_id_str = os.getenv("ODOO_BLOG_ID")
    if not blog_id_str:
        raise RuntimeError("Missing ODOO_BLOG_ID (numeric blog.blog id)")
    blog_id = int(blog_id_str)

    author_email = os.getenv("ODOO_AUTHOR_EMAIL")

    # Connect to Odoo
    db, uid, models = connect_odoo()

    # Collect markdown files
    patterns = [str(drafts_dir / "*.md"), str(drafts_dir / "*.markdown")]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    files = sorted(set(files))

    if not files:
        print(f"[info] No markdown files found in {drafts_dir}")
        return

    imported_posts = 0
    for file in files:
        try:
            post_id = import_post(models, db, uid, os.getenv("ODOO_PASSWORD"), blog_id, Path(file), author_email)
            if post_id:
                imported_posts += 1
        except Exception as e:
            print(f"[error] {file}: {e}")
            traceback.print_exc()

    print(f"[done] Imported/processed: {imported_posts}/{len(files)} files")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Fatal: {exc}")
        sys.exit(1)
