"""
Markdown RAG MCP Server v3.4 (Modular Architecture)

Features:
- Multi-format support (md, txt, rst, html, code files)
- Multilingual embeddings (BAAI/bge-m3, GPU-accelerated if available)
- Hybrid Search: ChromaDB vector + BM25 keyword scoring
- Cross-Encoder Reranking (multilingual, configurable)
- Incremental indexing (only re-indexes changed files)
- Multi-collection support
- BM25 persistence across server restarts
- Metadata filters (category, filename)
- Admin tools (list collections, remove sources)

Architecture (v3.4):
- server.py       — FastMCP tools, config, initialization (this file)
- parsers.py      — file reading, chunking, categorization
- search_engine.py — hybrid search pipeline, BM25, cross-encoder
- store.py        — ChromaDB ops, indexing, hash cache
- exceptions.py   — custom error types
"""

import os
import re
import warnings
import logging
from pathlib import Path
from typing import Optional

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from fastmcp import FastMCP
import chromadb
from chromadb.utils import embedding_functions

# Import modular components
import search_engine
import store
from parsers import (
    SUPPORTED_EXTENSIONS,
    _read_file_to_text,
    _categorize_file,
    _extract_sections,
    _extract_sections_smart,
)


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
_SERVER_DIR = Path(__file__).parent
DOCS_PATH = os.getenv(
    "RAG_DOCS_PATH",
    str(_SERVER_DIR.parent)
)
DB_PATH = os.getenv(
    "RAG_DB_PATH",
    str(_SERVER_DIR / "chroma_db")
)
COLLECTION_NAME = "docs_v4"
DEFAULT_COLLECTION = COLLECTION_NAME
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
MAX_CROSS_ENCODER_CHARS = 8000   # bge-reranker-v2-m3 supports up to 8192 tokens
MAX_RESULT_CHARS = 1500          # output truncation limit per result
_HASH_CACHE_PATH = _SERVER_DIR / "data" / "file_hashes.json"
_BM25_CACHE_PATH = _SERVER_DIR / "data" / "bm25_cache.pkl"


# ──────────────────────────────────────────────
# Device detection (GPU optional — no torch required)
# ──────────────────────────────────────────────
def _detect_device() -> str:
    """Detect compute device. Set RAG_DEVICE=cuda to force GPU."""
    requested = os.getenv("RAG_DEVICE", "auto").lower()
    if requested == "cpu":
        return "cpu"
    try:
        import torch
        if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
            return "cuda"
    except ImportError:
        pass
    return "cpu"

DEVICE = _detect_device()


# ──────────────────────────────────────────────
# ChromaDB + Models setup
# ──────────────────────────────────────────────
client = chromadb.PersistentClient(path=DB_PATH)

# GPU-aware embedding function with normalization (required for bge-m3)
try:
    from sentence_transformers import SentenceTransformer
    _embed_model = SentenceTransformer(EMBED_MODEL, device=DEVICE)

    class _EmbeddingFunction:
        """ChromaDB-compatible embedding function with normalization + optional GPU."""
        is_legacy = True  # Tell ChromaDB 0.6.0 to treat this as old-style callable

        def name(self) -> str:
            return EMBED_MODEL

        def __call__(self, input: list[str]) -> list[list[float]]:
            return _embed_model.encode(
                input,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,  # required for bge-m3 cosine similarity
            ).tolist()

        def embed_documents(self, input: list[str]) -> list[list[float]]:
            return self(input)

        def embed_query(self, input: list[str]) -> list[list[float]]:
            return self(input)


    embed_fn = _EmbeddingFunction()
except Exception:
    # Fallback: ChromaDB built-in (no normalization, CPU only)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    embed_fn.is_legacy = True  # safety patch

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embed_fn,
    metadata={"hnsw:space": "cosine"}
)

# Load BM25 from cache on startup
search_engine.load_bm25_on_startup(_BM25_CACHE_PATH, collection)

mcp = FastMCP("Markdown RAG")


# ──────────────────────────────────────────────
# Helper: common kwargs for store/search functions
# ──────────────────────────────────────────────
def _store_kwargs():
    """Common keyword arguments for store functions."""
    return dict(
        client=client,
        embed_fn=embed_fn,
        hash_cache_path=_HASH_CACHE_PATH,
        bm25_cache_path=_BM25_CACHE_PATH,
        embed_model=EMBED_MODEL,
        device=DEVICE,
        rerank_model=RERANK_MODEL,
        db_path=DB_PATH,
        default_collection=DEFAULT_COLLECTION,
    )


def _get_collection(name: str = DEFAULT_COLLECTION):
    """Get or create a named ChromaDB collection."""
    return store.get_collection(name, client, embed_fn)


def index_single_file(filepath: str, collection: str = DEFAULT_COLLECTION) -> int:
    """Safely index a single file via the store (used by Dashboard API)"""
    return store.index_single_file(
        filepath=filepath,
        collection=collection,
        **_store_kwargs()
    )


# ──────────────────────────────────────────────
# MCP Tools — Indexing
# ──────────────────────────────────────────────
@mcp.tool()
def index_documents(
    docs_path: str = DOCS_PATH,
    collection: str = DEFAULT_COLLECTION,
    force_reindex: bool = False
) -> str:
    """
    Index documents in the given directory into ChromaDB.
    Supports: .md, .txt, .rst, .html, .py, .js, .ts, and more.
    Uses incremental indexing — only re-indexes files that have changed.

    Args:
        docs_path: Path to scan for files (default: parent directory)
        collection: Collection name to index into (default: docs_v4)
        force_reindex: If True, ignore hash cache and re-index everything
    """
    return store.index_documents(
        docs_path=docs_path,
        collection=collection,
        force_reindex=force_reindex,
        **_store_kwargs(),
    )


@mcp.tool()
def reindex_collection(
    docs_path: str = DOCS_PATH,
    collection: str = DEFAULT_COLLECTION
) -> str:
    """Force full reindex of a collection. Deletes all existing chunks and re-indexes from scratch.

    Args:
        docs_path: Path to scan for files (default: parent directory)
        collection: Collection name to reindex (default: docs_v4)
    """
    return store.reindex_collection(
        docs_path=docs_path,
        collection=collection,
        **_store_kwargs(),
    )


# ──────────────────────────────────────────────
# MCP Tools — Search
# ──────────────────────────────────────────────
@mcp.tool()
def search_docs(
    query: str,
    n_results: int = 5,
    category: Optional[str] = None,
    filename: Optional[str] = None,
    collection: str = DEFAULT_COLLECTION
) -> str:
    """
    Semantic search across indexed Markdown documentation.
    Returns the most relevant chunks with source file references.

    Args:
        query: Natural language query, e.g. "how does the authentication logic work?"
        n_results: Number of results to return (default 5)
        category: Optional filter by category (dynamically based on your root folder names)
        filename: Optional filter by specific file, e.g. "architecture.md"
        collection: Collection to search in (default: docs_v4)
    """
    return search_engine.search_docs(
        query=query,
        n_results=n_results,
        category=category,
        filename=filename,
        collection_name=collection,
        get_collection_fn=_get_collection,
        embed_model=EMBED_MODEL,
        device=DEVICE,
        rerank_model=RERANK_MODEL,
        max_ce_chars=MAX_CROSS_ENCODER_CHARS,
        max_result_chars=MAX_RESULT_CHARS,
    )


# ──────────────────────────────────────────────
# MCP Tools — Admin
# ──────────────────────────────────────────────
@mcp.tool()
def rag_status(collection: str = DEFAULT_COLLECTION) -> str:
    """Show how many chunks are indexed and from which files.

    Args:
        collection: Collection to inspect (default: docs_v4)
    """
    target_col = _get_collection(collection)
    count = target_col.count()
    if count == 0:
        return f"No documents indexed in collection '{collection}'. Run index_documents() to start."

    all_data = target_col.get(include=["metadatas"])["metadatas"]

    files = {}
    categories = {}
    total_words = 0
    for m in all_data:
        fname = m.get("filename", "unknown")
        cat = m.get("category", "other")
        wc = m.get("word_count", 0)
        files[fname] = files.get(fname, 0) + 1
        categories[cat] = categories.get(cat, 0) + 1
        total_words += wc

    file_list = "\n".join(
        f"  - {name} ({chunks} chunks)"
        for name, chunks in sorted(files.items())
    )
    cat_list = "\n".join(
        f"  - {cat}: {n} chunks"
        for cat, n in sorted(categories.items())
    )

    bm25_index = search_engine._bm25_index
    bm25_corpus = search_engine._bm25_corpus
    bm25_loaded_from = search_engine._bm25_loaded_from
    cross_encoder = search_engine._cross_encoder

    if bm25_index is not None:
        bm25_status = f"Active ({len(bm25_corpus)} chunks, from {bm25_loaded_from or 'unknown'})"
    else:
        bm25_status = "Not built (re-index needed)"
    ce_status = "Loaded" if cross_encoder is not None else "Lazy (loads on first search)"

    return (
        f"## RAG Index Status\n\n"
        f"**Collection:** {collection}\n"
        f"**Total:** {count} chunks | ~{total_words:,} words\n"
        f"**Embedding:** {EMBED_MODEL} ({DEVICE})\n"
        f"**BM25:** {bm25_status}\n"
        f"**Cross-Encoder:** {ce_status} ({RERANK_MODEL})\n"
        f"**DB:** {DB_PATH}\n\n"
        f"### Files ({len(files)}):\n{file_list}\n\n"
        f"### Categories:\n{cat_list}"
    )


@mcp.tool()
def list_collections() -> str:
    """List all RAG collections in the database with their chunk counts."""
    try:
        collections = client.list_collections()
    except Exception as e:
        return f"Error listing collections: {e}"

    if not collections:
        return "No collections found. Run index_documents() to create one."

    lines = ["## RAG Collections\n"]
    for col in collections:
        try:
            count = col.count()
            lines.append(f"- **{col.name}** — {count} chunks")
        except Exception:
            lines.append(f"- **{col.name}** — (error reading)")
    return "\n".join(lines)


@mcp.tool()
def list_indexed_files(collection: str = DEFAULT_COLLECTION) -> str:
    """List all files that have been indexed in a collection.

    Args:
        collection: Collection to inspect (default: docs_v4)
    """
    target_col = _get_collection(collection)
    count = target_col.count()
    if count == 0:
        return f"No documents in collection '{collection}'."

    all_meta = target_col.get(include=["metadatas"])["metadatas"]

    files = {}
    for m in all_meta:
        fname = m.get("filename", "unknown")
        source = m.get("source", "")
        if fname not in files:
            files[fname] = {"chunks": 0, "source": source, "category": m.get("category", "")}
        files[fname]["chunks"] += 1

    lines = [f"## Indexed Files in '{collection}' ({len(files)} files, {count} chunks)\n"]
    for fname, info in sorted(files.items()):
        lines.append(f"- **{fname}** — {info['chunks']} chunks | category: {info['category']}")
        if info["source"]:
            lines.append(f"  Source: `{info['source']}`")
    return "\n".join(lines)


@mcp.tool()
def remove_source(
    filename: str,
    collection: str = DEFAULT_COLLECTION
) -> str:
    """Remove all chunks for a specific file from the index.

    Args:
        filename: Filename to remove, e.g. "api.md"
        collection: Collection to remove from (default: docs_v4)
    """
    return store.remove_source(
        filename=filename,
        collection=collection,
        client=client,
        embed_fn=embed_fn,
        bm25_cache_path=_BM25_CACHE_PATH,
        default_collection=DEFAULT_COLLECTION,
    )


@mcp.tool()
def delete_collection(
    name: str,
    confirm: bool = False
) -> str:
    """Delete an entire collection. Requires confirm=True for safety.

    Args:
        name: Collection name to delete
        confirm: Must be True to actually delete. Without it, shows what would be deleted.
    """
    return store.delete_collection(
        name=name,
        confirm=confirm,
        client=client,
        embed_fn=embed_fn,
        bm25_cache_path=_BM25_CACHE_PATH,
        default_collection=DEFAULT_COLLECTION,
    )


# ──────────────────────────────────────────────
# Web Crawling Integration
# ──────────────────────────────────────────────
@mcp.tool()
def index_url(
    uri: str,
    collection: str = DEFAULT_COLLECTION,
    max_pages: int = 200,
    max_depth: int = 10,
    stay_within_prefix: bool = True,
    exclude_patterns: list[str] = None,
    use_sitemap: bool = True,
    use_playwright: bool = False,
) -> str:
    """
    Index any web source into the RAG database. Supports:

    - URLs:    index_url("https://docs.python.org/3/library/")
    - GitHub:  index_url("github://owner/repo") or index_url("github://owner/repo/docs")
    - npm:     index_url("npm://axios@1.6")
    - PyPI:    index_url("pypi://fastapi")
    - ZIP:     index_url("file:///C:/path/to/docs.zip")

    Args:
        uri:                  Source URI to index
        collection:           Collection to index into (default: docs_v4)
        max_pages:            Max pages to crawl for URLs (default: 200)
        max_depth:            Max link depth from starting URL (default: 10)
        stay_within_prefix:   Don't leave the starting URL path (default: True)
        exclude_patterns:     Regex patterns to skip URLs (e.g. ["/blog/", "/changelog/"])
        use_sitemap:          Try sitemap.xml first for faster discovery (default: True)
        use_playwright:       Use headless browser for JS-rendered sites (default: False)
    """
    import asyncio

    try:
        from crawler import (
            crawl_and_index,
            index_github,
            index_npm,
            index_pypi,
            index_zip,
        )
    except ImportError as e:
        return (
            f"Error: crawler module not available. {e}\n"
            "Make sure crawler.py is in the same directory as server.py\n"
            "and install: pip install httpx beautifulsoup4 html2text"
        )

    try:
        try:
            asyncio.get_running_loop()
            is_running = True
        except RuntimeError:
            is_running = False

        if is_running:
            import concurrent.futures
            def _run_in_thread():
                return asyncio.run(_run_index(
                    uri, collection, max_pages, max_depth,
                    stay_within_prefix, exclude_patterns,
                    use_sitemap, use_playwright,
                ))
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(_run_in_thread)
                return future.result()
        else:
            return asyncio.run(_run_index(
                uri, collection, max_pages, max_depth,
                stay_within_prefix, exclude_patterns,
                use_sitemap, use_playwright,
            ))
    except Exception as e:
        return f"Error during indexing: {e}"


async def _run_index(
    uri: str,
    collection: str,
    max_pages: int,
    max_depth: int,
    stay_within_prefix: bool,
    exclude_patterns: list[str] | None,
    use_sitemap: bool,
    use_playwright: bool,
) -> str:
    """Async dispatcher for index_url()."""
    from crawler import (
        crawl_and_index,
        index_github,
        index_npm,
        index_pypi,
        index_zip,
    )

    if uri.startswith(("http://", "https://")):
        pages, status = await crawl_and_index(
            uri, collection, max_pages, max_depth,
            stay_within_prefix, exclude_patterns,
            use_sitemap, use_playwright,
        )
        label = f"Web: {uri}"

    elif uri.startswith("github://"):
        pages, status = await index_github(uri, collection)
        label = f"GitHub: {uri}"

    elif uri.startswith("npm://"):
        pages, status = await index_npm(uri, collection)
        label = f"npm: {uri}"

    elif uri.startswith("pypi://"):
        pages, status = await index_pypi(uri, collection)
        label = f"PyPI: {uri}"

    elif uri.lower().endswith(".zip"):
        local_path = uri.replace("file:///", "").replace("file://", "")
        pages, status = await index_zip(local_path)
        label = f"ZIP: {uri}"

    else:
        return f"Error: unknown URI scheme '{uri}'. Use http://, https://, github://, npm://, pypi://"

    if not pages:
        return f"No content found.\n{status}"

    index_result = store.index_web_pages(
        pages, collection, label,
        client=client,
        embed_fn=embed_fn,
        bm25_cache_path=_BM25_CACHE_PATH,
    )
    return f"{status}\n{index_result}"




if __name__ == "__main__":
    # Start web dashboard if configured
    if os.getenv("RAG_DASHBOARD", "").lower() in ("true", "1", "yes"):
        try:
            from dashboard import start_dashboard_thread
            start_dashboard_thread()
        except Exception:
            pass  # dashboard is optional

    # Start file watcher if configured (FEAT-09)
    _watcher_observer = None
    if os.getenv("RAG_WATCH_PATH"):
        try:
            from watcher import start_watcher
            _watcher_observer = start_watcher(
                os.getenv("RAG_WATCH_PATH"),
                os.getenv("RAG_WATCH_COLLECTION", DEFAULT_COLLECTION),
            )
        except Exception:
            pass  # watcher is optional
    mcp.run(transport="stdio")
