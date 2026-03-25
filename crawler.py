"""
Markdown RAG MCP Server — Web Crawler Module

Provides URL crawling, HTML→Markdown conversion,
and specialized loaders for GitHub, npm, PyPI, and ZIP sources.

Used by server.py via index_url() MCP tool.
"""

import re
import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Optional
from collections import deque
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Layer 1 — HTML → Markdown Parser (Multi-Strategy)
# ──────────────────────────────────────────────

# Minimum characters for extracted content to be considered valid
_MIN_CONTENT_LENGTH = 200


def _extract_links(html: str) -> list[str]:
    """Extract all internal links from raw HTML before any destructive parsing."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href and not href.startswith(("javascript:", "mailto:", "tel:", "#")):
            links.append(href)
    return links


def _build_header(html: str, page_url: str) -> str:
    """Extract page title and build a Markdown header."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text().strip() if title_tag else ""
    if title:
        return f"# {title}\n\nSource: {page_url}\n\n"
    return f"Source: {page_url}\n\n"


def _clean_markdown(text: str) -> str:
    """Clean up excessive whitespace and artifacts from conversion."""
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    # Remove lines that are only whitespace/special chars
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = re.sub(r"[|\\—\-\s#*`>]", "", line)
        if stripped or not line.strip():
            cleaned.append(line)
    return "\n".join(cleaned)


def _html_to_markdown(html_fragment: str) -> str:
    """Convert an HTML fragment to Markdown using html2text."""
    import html2text
    converter = html2text.HTML2Text()
    converter.ignore_images = True
    converter.body_width = 0          # no line wrapping
    converter.unicode_snob = True     # preserve unicode
    converter.ignore_links = False    # keep inline links
    converter.protect_links = True
    converter.wrap_links = False
    return converter.handle(html_fragment)


# ── Strategy 1: Trafilatura (ML-based, works on ANY site) ──

def _extract_with_trafilatura(html: str) -> str:
    """
    Use Trafilatura for intelligent main-content extraction.
    Trafilatura uses ML heuristics to identify the main content area
    regardless of DOM structure — works perfectly with React SPAs,
    custom CMSes, and any non-standard HTML.
    """
    try:
        import trafilatura
        result = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            include_links=True,
            include_formatting=True,   # preserve **bold**, *italic*, etc.
            favor_recall=True,         # prefer more content over precision
            output_format="txt",       # clean text output
        )
        return result or ""
    except ImportError:
        return ""
    except Exception as e:
        logger.debug(f"Trafilatura extraction failed: {e}")
        return ""


# ── Strategy 2: BeautifulSoup with expanded selectors ──

def _extract_with_beautifulsoup(html: str) -> str:
    """
    Traditional CSS-selector-based extraction with an expanded list
    of known content containers used by popular frameworks.
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    # Remove noise elements
    noise_selectors = [
        "nav", "footer", "header", "aside",
        ".sidebar", ".navigation", ".breadcrumb", ".toc",
        ".cookie-banner", ".cookie-consent", ".announcement-bar",
        "[role='navigation']", "[role='banner']", "[role='complementary']",
        "script", "style", "noscript", "iframe", "svg",
    ]
    for selector in noise_selectors:
        for el in soup.select(selector):
            el.decompose()

    # Expanded content selectors covering more frameworks & CMS layouts
    content_selectors = [
        # Semantic HTML5
        "main", "article",
        # Standard IDs
        "#content", "#main-content", "#main", "#app-content",
        "#docs-content", "#page-content", "#primary",
        # Standard classes
        ".content", ".main-content", ".page-content",
        ".documentation", ".docs-content", ".doc-content",
        ".markdown-body", ".article-content", ".post-content",
        ".entry-content", ".rich-text", ".prose",
        # React/Vue/Angular root containers
        "#root", "#app", "#__next", "#__nuxt",
        "[role='main']",
        # Gitbook, Docusaurus, MkDocs, ReadTheDocs
        ".gitbook-root", ".theme-doc-markdown",
        ".md-content", ".rst-content",
    ]

    main_content = None
    for selector in content_selectors:
        found = soup.select_one(selector)
        if found:
            text = found.get_text(strip=True)
            if len(text) >= _MIN_CONTENT_LENGTH:
                main_content = found
                break

    # Fallback to <body>
    if not main_content:
        body = soup.find("body")
        if body and len(body.get_text(strip=True)) >= _MIN_CONTENT_LENGTH:
            main_content = body

    if not main_content:
        return ""

    return _html_to_markdown(str(main_content))


# ── Strategy 3: Text Density Analysis (last resort) ──

def _extract_by_text_density(html: str) -> str:
    """
    Last-resort extraction: find the DOM subtree with the highest
    text density (text-to-tag ratio). This works on completely
    non-standard layouts where no known selectors match.
    """
    from bs4 import BeautifulSoup, NavigableString
    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts, styles and other non-content tags
    for tag in soup.find_all(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()

    # Score each block-level element by text density
    block_tags = {"div", "section", "article", "main", "td", "li", "p", "span"}
    best_node = None
    best_score = 0

    for element in soup.find_all(block_tags):
        text = element.get_text(strip=True)
        text_len = len(text)
        if text_len < _MIN_CONTENT_LENGTH:
            continue

        # Count descendant tags (more tags = more noise)
        tag_count = len(element.find_all()) + 1
        # Text density = characters per tag
        density = text_len / tag_count
        # Bonus for elements with paragraphs or headings (likely content)
        p_count = len(element.find_all(["p", "h1", "h2", "h3", "h4", "li"]))
        score = density * (1 + p_count * 0.5)

        if score > best_score:
            best_score = score
            best_node = element

    if not best_node:
        return ""

    return _html_to_markdown(str(best_node))


# ── Combined Parser ──

def _parse_html_page(html: str, page_url: str) -> tuple[str, list[str]]:
    """
    Parse an HTML page into clean Markdown + extract internal links.
    
    Uses a 3-strategy pipeline:
    1. Trafilatura (ML-based, handles ANY site including React SPAs)
    2. BeautifulSoup with expanded selectors (traditional approach)
    3. Text density analysis (last resort for completely unknown layouts)
    
    Returns:
        (markdown_content, list_of_links)
    """
    # Extract links from the original HTML before any destructive parsing
    links = _extract_links(html)
    header = _build_header(html, page_url)

    # Strategy 1: Trafilatura (best for unknown/SPA sites)
    content = _extract_with_trafilatura(html)
    if content and len(content.strip()) >= _MIN_CONTENT_LENGTH:
        logger.debug(f"Trafilatura extracted {len(content)} chars from {page_url}")
        return header + content, links

    # Strategy 2: BeautifulSoup with expanded selectors
    content = _extract_with_beautifulsoup(html)
    if content and len(content.strip()) >= _MIN_CONTENT_LENGTH:
        content = _clean_markdown(content)
        logger.debug(f"BeautifulSoup extracted {len(content)} chars from {page_url}")
        return header + content, links

    # Strategy 3: Text density analysis (last resort)
    content = _extract_by_text_density(html)
    if content and len(content.strip()) >= _MIN_CONTENT_LENGTH:
        content = _clean_markdown(content)
        logger.debug(f"Text density extracted {len(content)} chars from {page_url}")
        return header + content, links

    # Nothing worked — page is truly empty or too small
    logger.debug(f"No content extracted from {page_url}")
    return "", links


# ──────────────────────────────────────────────
# Layer 2 — URL Control (boundary, robots, normalize)
# ──────────────────────────────────────────────

_SKIP_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico", ".webp", ".bmp",
    ".css", ".js", ".mjs", ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".exe", ".dmg", ".pkg", ".deb", ".rpm", ".msi",
    ".mp3", ".mp4", ".avi", ".mov", ".mkv", ".wav", ".ogg",
    ".pdf",  # PDF = binary, handled separately
}


def _normalize_url(url: str) -> str:
    """Normalize URL: strip fragment, trailing slash, sort query params."""
    url = url.split("#")[0]  # remove fragment
    if url.endswith("/") and url.count("/") > 3:
        url = url.rstrip("/")
    return url


def _is_allowed_url(
    url: str,
    base_prefix: str,
    base_domain: str,
    stay_within_prefix: bool,
    exclude_patterns: list[str] | None,
) -> bool:
    """Check if a URL is allowed to be crawled."""
    parsed = urlparse(url)

    # Rule 1: only http/https
    if parsed.scheme not in ("http", "https"):
        return False

    # Rule 2: same domain only
    if parsed.netloc != base_domain:
        return False

    # Rule 3: stay within prefix path
    if stay_within_prefix:
        clean_url = url.split("?")[0].split("#")[0]
        if not clean_url.startswith(base_prefix):
            return False

    # Rule 4: user-defined exclusions
    for pattern in (exclude_patterns or []):
        if re.search(pattern, url):
            return False

    # Rule 5: skip binary/media files
    path_lower = parsed.path.lower()
    if any(path_lower.endswith(ext) for ext in _SKIP_EXTENSIONS):
        return False

    return True


def _robots_allows(url: str, cache: dict) -> bool:
    """Check robots.txt compliance. Permissive on errors."""
    parsed = urlparse(url)
    domain = f"{parsed.scheme}://{parsed.netloc}"

    if domain not in cache:
        rp = RobotFileParser()
        rp.set_url(urljoin(domain, "/robots.txt"))
        try:
            rp.read()
            cache[domain] = rp
        except Exception:
            cache[domain] = None  # on error, allow everything

    rp = cache[domain]
    return rp is None or rp.can_fetch("RAG-Indexer", url)


# ──────────────────────────────────────────────
# Layer 3 — Sitemap Parser
# ──────────────────────────────────────────────

async def _try_get_sitemap_urls(
    base_url: str,
    base_domain: str,
    base_prefix: str,
) -> list[str]:
    """
    Try to discover URLs from sitemap.xml.
    Returns list of URLs or empty list.
    """
    import httpx

    sitemap_candidates = [
        f"https://{base_domain}/sitemap.xml",
        f"https://{base_domain}/sitemap_index.xml",
    ]
    # Add prefix-level sitemap
    parsed = urlparse(base_url)
    path_parts = parsed.path.strip("/").split("/")
    if path_parts and path_parts[0]:
        sitemap_candidates.append(
            f"https://{base_domain}/{path_parts[0]}/sitemap.xml"
        )

    urls = []
    async with httpx.AsyncClient(
        timeout=10,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (compatible; RAG-Indexer/2.0)"},
    ) as client:
        for sitemap_url in sitemap_candidates:
            try:
                resp = await client.get(sitemap_url)
                if resp.status_code != 200:
                    continue
                if "xml" not in resp.headers.get("content-type", ""):
                    continue

                # Parse sitemap XML for <loc> tags
                for match in re.finditer(r"<loc>(.*?)</loc>", resp.text):
                    loc = match.group(1).strip()
                    if loc.startswith(base_prefix):
                        urls.append(loc)

                if urls:
                    break  # found a working sitemap
            except Exception:
                continue

    return list(dict.fromkeys(urls))  # deduplicate, preserve order


# ──────────────────────────────────────────────
# Layer 4 — Async BFS Web Crawler
# ──────────────────────────────────────────────

async def crawl_and_index(
    base_url: str,
    collection_name: str,
    max_pages: int = 200,
    max_depth: int = 10,
    stay_within_prefix: bool = True,
    exclude_patterns: list[str] | None = None,
    use_sitemap: bool = True,
    use_playwright: bool = False,
) -> tuple[list[dict], str]:
    """
    Crawl a website starting from base_url and return parsed pages.
    
    Returns:
        (pages_list, status_message)
        Each page = {"url": str, "content": str_markdown, "depth": int}
    """
    import httpx

    parsed_base = urlparse(base_url)
    base_domain = parsed_base.netloc
    base_prefix = base_url.split("?")[0].split("#")[0]

    robots_cache: dict = {}
    visited: set[str] = set()
    pages: list[dict] = []
    errors = 0

    # Step 1: try sitemap for fast URL discovery
    url_queue: deque[tuple[str, int]] = deque()
    sitemap_count = 0

    if use_sitemap:
        try:
            sitemap_urls = await _try_get_sitemap_urls(base_url, base_domain, base_prefix)
            if sitemap_urls:
                sitemap_count = len(sitemap_urls)
                for u in sitemap_urls[:max_pages]:
                    url_queue.append((u, 1))
        except Exception:
            pass

    # If no sitemap found, start from base URL
    if not url_queue:
        url_queue.append((base_url, 0))

    # Step 2: BFS crawl
    async with httpx.AsyncClient(
        timeout=15,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (compatible; RAG-Indexer/2.0)"},
    ) as client:
        while url_queue and len(pages) < max_pages:
            url, depth = url_queue.popleft()
            url = _normalize_url(url)

            if url in visited:
                continue
            if not _is_allowed_url(url, base_prefix, base_domain,
                                    stay_within_prefix, exclude_patterns):
                continue
            if not _robots_allows(url, robots_cache):
                continue

            visited.add(url)

            try:
                if use_playwright:
                    html = await _fetch_with_playwright(url)
                else:
                    resp = await client.get(url)
                    if resp.status_code != 200:
                        errors += 1
                        continue
                    content_type = resp.headers.get("content-type", "")
                    if "text/html" not in content_type:
                        continue
                    html = resp.text

                content_md, links = _parse_html_page(html, url)
                if content_md.strip():
                    pages.append({
                        "url": url,
                        "content": content_md,
                        "depth": depth,
                    })

                # Discover new links
                if depth < max_depth:
                    for link in links:
                        abs_link = _normalize_url(urljoin(url, link))
                        if abs_link not in visited:
                            url_queue.append((abs_link, depth + 1))

            except Exception as e:
                errors += 1
                logger.debug(f"Crawl error for {url}: {e}")
                continue

            # Rate limiting: 100ms between requests
            await asyncio.sleep(0.1)

    status = (
        f"Crawled {len(pages)} pages from {base_domain} "
        f"(visited {len(visited)}, errors {errors})"
    )
    if sitemap_count:
        status += f" | Sitemap: {sitemap_count} URLs discovered"

    return pages, status


async def _fetch_with_playwright(url: str) -> str:
    """Fetch a JS-rendered page using Playwright (optional dependency).
    
    Uses 'domcontentloaded' + explicit wait instead of 'networkidle'
    because SPAs (React, Vue, Angular) never reach 'networkidle' —
    they keep streaming analytics, websockets, and lazy-loaded assets.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise RuntimeError(
            "Playwright not installed. Run:\n"
            "  pip install playwright\n"
            "  playwright install chromium"
        )

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            # Step 1: Navigate and wait for DOM to be ready
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            # Step 2: Wait for JS frameworks to render content
            # Most SPAs finish rendering within 3 seconds
            await page.wait_for_timeout(3000)
            # Step 3: Try to wait for the body to have actual text content
            try:
                await page.wait_for_function(
                    "document.body.innerText.length > 200",
                    timeout=5000,
                )
            except Exception:
                pass  # some pages may just be small, continue anyway
            html = await page.content()
        finally:
            await browser.close()
        return html


# ──────────────────────────────────────────────
# Layer 5 — Specialized Loaders
# ──────────────────────────────────────────────

async def index_github(
    uri: str,
    collection_name: str,
) -> tuple[list[dict], str]:
    """
    Download documentation from a GitHub repository.
    
    URI formats:
        github://owner/repo           → whole repo
        github://owner/repo/docs      → only /docs folder
        github://owner/repo@v2.0.0    → specific tag/branch
    
    Returns:
        (pages_list, status_message)
    """
    import httpx
    import os

    path = uri.replace("github://", "")
    branch = "HEAD"
    sub_path = ""

    if "@" in path:
        path, branch = path.split("@", 1)

    parts = path.split("/", 2)
    if len(parts) < 2:
        return [], f"Error: invalid GitHub URI '{uri}'. Expected: github://owner/repo"

    owner, repo = parts[0], parts[1]
    sub_path = parts[2] if len(parts) > 2 else ""

    headers = {}
    if token := os.getenv("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"

    api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"

    async with httpx.AsyncClient(headers=headers, timeout=30) as client:
        resp = await client.get(api_url)
        if resp.status_code == 403:
            return [], "Error: GitHub rate limit. Set GITHUB_TOKEN env var."
        if resp.status_code == 404:
            return [], f"Error: repo {owner}/{repo} not found or is private."
        tree = resp.json().get("tree", [])

    doc_extensions = {".md", ".rst", ".txt", ".html", ".htm"}
    doc_files = [
        f for f in tree
        if f["type"] == "blob"
        and Path(f["path"]).suffix.lower() in doc_extensions
        and (not sub_path or f["path"].startswith(sub_path))
    ]

    pages = []
    async with httpx.AsyncClient(headers=headers, timeout=10) as client:
        for file_info in doc_files[:300]:  # limit to 300 files
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_info['path']}"
            try:
                r = await client.get(raw_url)
                if r.status_code == 200 and r.text.strip():
                    pages.append({
                        "url": raw_url,
                        "content": r.text,
                        "depth": 0,
                    })
            except Exception:
                continue

    status = f"GitHub: downloaded {len(pages)} docs from {owner}/{repo}"
    if sub_path:
        status += f" (path: {sub_path})"

    return pages, status


async def index_npm(
    uri: str,
    collection_name: str,
) -> tuple[list[dict], str]:
    """
    Download documentation from npm registry.
    
    URI formats:
        npm://axios@1.6   → specific version
        npm://axios       → latest version
    
    Returns:
        (pages_list, status_message)
    """
    import httpx

    pkg_spec = uri.replace("npm://", "")
    if "@" in pkg_spec and not pkg_spec.startswith("@"):
        pkg_name, version = pkg_spec.rsplit("@", 1)
    else:
        pkg_name, version = pkg_spec, "latest"

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"https://registry.npmjs.org/{pkg_name}/{version}")
        if resp.status_code != 200:
            return [], f"Error: npm package '{pkg_name}@{version}' not found"
        pkg_data = resp.json()

    pages = []
    if readme := pkg_data.get("readme", ""):
        pages.append({
            "url": f"https://www.npmjs.com/package/{pkg_name}",
            "content": readme,
            "depth": 0,
        })

    actual_version = pkg_data.get("version", version)
    status = f"npm: fetched README for {pkg_name}@{actual_version} ({len(pages)} docs)"

    return pages, status


async def index_pypi(
    uri: str,
    collection_name: str,
) -> tuple[list[dict], str]:
    """
    Download documentation from PyPI.
    
    URI formats:
        pypi://fastapi@0.110   → specific version
        pypi://fastapi         → latest
    
    Returns:
        (pages_list, status_message)
    """
    import httpx

    pkg_spec = uri.replace("pypi://", "")
    if "@" in pkg_spec:
        pkg_name, version = pkg_spec.split("@", 1)
        api_url = f"https://pypi.org/pypi/{pkg_name}/{version}/json"
    else:
        pkg_name = pkg_spec
        api_url = f"https://pypi.org/pypi/{pkg_name}/json"

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(api_url)
        if resp.status_code != 200:
            return [], f"Error: PyPI package '{pkg_name}' not found"
        data = resp.json()

    info = data.get("info", {})
    description = info.get("description", "")

    pages = []
    if description:
        pages.append({
            "url": f"https://pypi.org/project/{pkg_name}/",
            "content": description,
            "depth": 0,
        })

    actual_version = info.get("version", "latest")
    status = f"PyPI: fetched description for {pkg_name}@{actual_version} ({len(pages)} docs)"

    return pages, status


async def index_zip(
    zip_path: str,
) -> tuple[list[dict], str]:
    """
    Extract and read documentation from a ZIP archive.
    
    Returns:
        (pages_list, status_message)
    """
    import zipfile
    import tempfile

    path = Path(zip_path)
    if not path.exists():
        return [], f"Error: ZIP file not found: {zip_path}"

    doc_extensions = {".md", ".txt", ".rst", ".html", ".htm"}
    pages = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(str(path), "r") as zf:
            zf.extractall(tmp_dir)

        for file_path in Path(tmp_dir).rglob("*"):
            if file_path.suffix.lower() in doc_extensions and file_path.is_file():
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    if content.strip():
                        rel_path = file_path.relative_to(tmp_dir)
                        pages.append({
                            "url": f"zip://{path.name}/{rel_path}",
                            "content": content,
                            "depth": 0,
                        })
                except Exception:
                    continue

    status = f"ZIP: extracted {len(pages)} docs from {path.name}"
    return pages, status
