"""
OmniDocs RAG — Data Store

Responsible for:
- ChromaDB collection management
- File hash caching for incremental indexing
- Document indexing pipeline (scan → parse → chunk → store)
- Collection CRUD: create, delete, reindex, remove sources
"""

import os
import re
import json
import glob
import hashlib
import threading
from pathlib import Path
from typing import Optional

from parsers import (
    SUPPORTED_EXTENSIONS,
    _read_file_to_text,
    _categorize_file,
    _extract_sections,
    _extract_sections_smart,
)
from search_engine import _build_bm25
from exceptions import OmniDocsError, IndexingError, CollectionError

# Thread safety lock for concurrent access from watcher + API + MCP
_index_lock = threading.Lock()


# ──────────────────────────────────────────────
# File Hash Cache for Incremental Indexing
# ──────────────────────────────────────────────
def _load_hash_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_hash_cache(cache: dict, cache_path: Path):
    with _index_lock:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _get_file_hash(filepath: str) -> str:
    return hashlib.md5(Path(filepath).read_bytes()).hexdigest()


# ──────────────────────────────────────────────
# Collection Management
# ──────────────────────────────────────────────
def get_collection(name: str, client, embed_fn):
    """Get or create a named ChromaDB collection."""
    return client.get_or_create_collection(
        name=name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )


# ──────────────────────────────────────────────
# Main Indexing Pipeline
# ──────────────────────────────────────────────
def index_documents(
    docs_path: str,
    collection: str,
    force_reindex: bool = False,
    *,
    client,
    embed_fn,
    hash_cache_path: Path,
    bm25_cache_path: Path,
    embed_model: str,
    device: str,
    rerank_model: str,
    db_path: str,
    default_collection: str,
) -> str:
    """
    Index documents in the given directory into ChromaDB.
    Supports: .md, .txt, .rst, .html, .py, .js, .ts, and more.
    Uses incremental indexing — only re-indexes files that have changed.
    """
    with _index_lock:
        # Gather all supported files
        all_files = []
        docs_p = Path(docs_path).resolve()
        for ext in SUPPORTED_EXTENSIONS:
            for filepath in glob.glob(os.path.join(docs_path, "**", f"*{ext}"), recursive=True):
                try:
                    rel_parts = Path(filepath).resolve().relative_to(docs_p).parts
                    if any(p.startswith(".") or p in ("__pycache__", "node_modules", "venv", "env", "chroma_db", "build", "dist") for p in rel_parts):
                        continue
                    all_files.append(filepath)
                except ValueError:
                    pass  # Path not relative to docs_path
        all_files = sorted(set(all_files))  # deduplicate

        if not all_files:
            return f"No supported files found in {docs_path}\nSupported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"

        # ── Incremental indexing: detect changed files ──
        hash_cache = _load_hash_cache(hash_cache_path) if not force_reindex else {}
        changed_files = []
        unchanged_count = 0
        new_hash_cache = {}

        for filepath in all_files:
            try:
                current_hash = _get_file_hash(filepath)
            except Exception:
                continue
            norm_path = os.path.normpath(filepath)
            new_hash_cache[norm_path] = current_hash
            if not force_reindex and hash_cache.get(norm_path) == current_hash:
                unchanged_count += 1
            else:
                changed_files.append(filepath)

        # Detect deleted files (in cache but no longer on disk)
        current_paths = {os.path.normpath(f) for f in all_files}
        deleted_paths = set(hash_cache.keys()) - current_paths

        # If nothing changed and nothing deleted, skip
        if not changed_files and not deleted_paths:
            return (
                f"Nothing changed. {unchanged_count} files already up to date.\n"
                f"Collection: {collection}\n"
                f"Path: {docs_path}\n"
                f"Use force_reindex=True to rebuild everything."
            )

        target_col = get_collection(collection, client, embed_fn)

        # ── Handle deleted files: remove their chunks ──
        if deleted_paths:
            try:
                for del_path in deleted_paths:
                    del_filename = Path(del_path).name
                    old_data = target_col.get(
                        where={"filename": del_filename},
                        include=[]
                    )
                    if old_data["ids"]:
                        target_col.delete(ids=old_data["ids"])
            except Exception:
                pass

        # ── If force or first-time: full re-index ──
        if force_reindex or target_col.count() == 0:
            try:
                client.delete_collection(collection)
            except Exception:
                pass
            target_col = get_collection(collection, client, embed_fn)
            changed_files = all_files

        # ── Remove old chunks from changed files ──
        if not force_reindex:
            for filepath in changed_files:
                fname = Path(filepath).name
                try:
                    old_data = target_col.get(
                        where={"filename": fname},
                        include=[]
                    )
                    if old_data["ids"]:
                        target_col.delete(ids=old_data["ids"])
                except Exception:
                    pass

        # ── Index changed files ──
        ids, texts, metas = [], [], []

        for filepath in changed_files:
            raw = _read_file_to_text(filepath)
            if not raw:
                continue

            category = _categorize_file(filepath, raw)
            sections = _extract_sections_smart(raw, filepath)

            for sec in sections:
                path_hash = hashlib.md5(filepath.encode("utf-8")).hexdigest()[:6]
                chunk_hash = hashlib.md5(sec["text"].encode("utf-8")).hexdigest()[:10]
                chunk_id = f"{Path(filepath).stem}_{path_hash}__{chunk_hash}"
                chunk_id = re.sub(r"[^a-zA-Z0-9_]", "_", chunk_id)

                ids.append(chunk_id)
                texts.append(sec["text"])
                metas.append({
                    "source": sec["source"],
                    "filename": sec["filename"],
                    "heading": sec["heading"],
                    "parent_heading": sec["parent_heading"],
                    "category": category,
                    "word_count": sec["word_count"]
                })

        # ── Add to ChromaDB in batches ──
        batch_size = 50
        added = 0
        for i in range(0, len(ids), batch_size):
            batch_end = min(i + batch_size, len(ids))
            target_col.add(
                ids=ids[i:batch_end],
                documents=texts[i:batch_end],
                metadatas=metas[i:batch_end],
            )
            added += batch_end - i

        # ── Rebuild BM25 from full collection ──
        all_data = target_col.get(include=["documents"])
        all_ids = all_data["ids"]
        all_texts = all_data["documents"]
        _build_bm25(all_ids, all_texts, bm25_cache_path)

        # Save hash cache
        _save_hash_cache(new_hash_cache, hash_cache_path)

        categories = {}
        for m in metas:
            cat = m["category"]
            categories[cat] = categories.get(cat, 0) + 1

        cat_summary = " | ".join(f"{k}: {v}" for k, v in sorted(categories.items()))
        total_chunks = target_col.count()

        # Count file types
        ext_counts = {}
        for fp in changed_files:
            ext = Path(fp).suffix.lower()
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
        ext_str = ", ".join(f"{e}: {c}" for e, c in sorted(ext_counts.items()))

        status_parts = []
        if deleted_paths:
            status_parts.append(f"Deleted: {len(deleted_paths)} removed files")
        status_parts.append(
            f"Indexed {added} chunks from {len(changed_files)} changed files "
            f"({unchanged_count} unchanged)"
        )

        return (
            f"{' | '.join(status_parts)}\n"
            f"Total: {total_chunks} chunks in collection '{collection}'\n"
            f"File types: {ext_str}\n"
            f"Path: {docs_path}\n"
            f"Categories: {cat_summary}\n"
            f"Model: {embed_model} | Device: {device}\n"
            f"Reranker: {rerank_model}\n"
            f"DB: {db_path}"
        )


def remove_source(
    filename: str,
    collection: str,
    *,
    client,
    embed_fn,
    bm25_cache_path: Path,
    default_collection: str,
) -> str:
    """Remove all chunks for a specific file from the index."""
    with _index_lock:
        target_col = get_collection(collection, client, embed_fn)

        try:
            data = target_col.get(
                where={"filename": filename},
                include=[]
            )
        except Exception as e:
            raise IndexingError(f"Error accessing collection: {e}")

        if not data["ids"]:
            return f"No chunks found for file '{filename}' in collection '{collection}'."

        count = len(data["ids"])
        target_col.delete(ids=data["ids"])

        # Rebuild BM25 after removal
        remaining = target_col.get(include=["documents"])
        _build_bm25(remaining["ids"], remaining["documents"], bm25_cache_path)

        return f"Removed {count} chunks for '{filename}' from collection '{collection}'."


def delete_collection(
    name: str,
    confirm: bool = False,
    *,
    client,
    embed_fn,
    bm25_cache_path: Path,
    default_collection: str,
) -> str:
    """Delete an entire collection. Requires confirm=True for safety."""
    with _index_lock:
        try:
            target_col = get_collection(name, client, embed_fn)
            chunk_count = target_col.count()
        except Exception:
            chunk_count = 0

        if not confirm:
            return (
                f"⚠️ Will delete collection '{name}' ({chunk_count} chunks).\n"
                f"To confirm: delete_collection(name='{name}', confirm=True)"
            )

        try:
            client.delete_collection(name)
        except Exception as e:
            raise CollectionError(f"Error deleting collection '{name}': {e}")

        # Rebuild BM25 (collection is gone, may need cleanup)
        try:
            default_col = get_collection(default_collection, client, embed_fn)
            remaining = default_col.get(include=["documents"])
            _build_bm25(remaining["ids"], remaining["documents"], bm25_cache_path)
        except Exception:
            pass

        return f"✅ Deleted collection '{name}' ({chunk_count} chunks removed)."


def reindex_collection(
    docs_path: str,
    collection: str,
    **kwargs,
) -> str:
    """Force full reindex — delegates to index_documents with force_reindex=True."""
    return index_documents(docs_path=docs_path, collection=collection, force_reindex=True, **kwargs)


def index_web_pages(
    pages: list[dict],
    collection: str,
    source_label: str,
    *,
    client,
    embed_fn,
    bm25_cache_path: Path,
) -> str:
    """
    Bridge function: takes crawled pages and indexes them into ChromaDB.
    Reuses the existing chunking pipeline from parsers.py.
    """
    target_col = get_collection(collection, client, embed_fn)

    ids, texts, metas = [], [], []

    for page in pages:
        content = page["content"]
        url = page.get("url", "web")

        # Use the existing chunking pipeline
        sections = _extract_sections(content, url)
        category = _categorize_file(url, content)

        for sec in sections:
            url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:6]
            chunk_hash = hashlib.md5(sec["text"].encode("utf-8")).hexdigest()[:10]
            chunk_id = f"web_{url_hash}__{chunk_hash}"
            chunk_id = re.sub(r"[^a-zA-Z0-9_]", "_", chunk_id)

            ids.append(chunk_id)
            texts.append(sec["text"])
            metas.append({
                "source": url,
                "filename": url.split("/")[-1] or "index",
                "heading": sec["heading"],
                "parent_heading": sec["parent_heading"],
                "category": category,
                "word_count": sec["word_count"],
            })

    if not ids:
        return f"No content extracted from {source_label}."

    with _index_lock:
        # Add to ChromaDB in batches
        batch_size = 50
        added = 0
        for i in range(0, len(ids), batch_size):
            batch_end = min(i + batch_size, len(ids))
            target_col.add(
                ids=ids[i:batch_end],
                documents=texts[i:batch_end],
                metadatas=metas[i:batch_end],
            )
            added += batch_end - i

        # Rebuild BM25
        all_data = target_col.get(include=["documents"])
        _build_bm25(all_data["ids"], all_data["documents"], bm25_cache_path)

    return (
        f"Indexed {added} chunks from {len(pages)} pages\n"
        f"Source: {source_label}\n"
        f"Collection: {collection}\n"
        f"Total: {target_col.count()} chunks"
    )


def index_single_file(
    filepath: str,
    collection: str,
    *,
    client,
    embed_fn,
    hash_cache_path: Path,
    bm25_cache_path: Path,
) -> int:
    """Index or re-index a single file from the API/Dashboard safely."""
    with _index_lock:
        target_col = get_collection(collection, client, embed_fn)
        path_obj = Path(filepath)
        filename = path_obj.name
        
        raw = _read_file_to_text(filepath)
        if not raw:
            raise IndexingError(f"Could not read file or unsupported format: {path_obj.suffix}")
            
        category = _categorize_file(filepath, raw)
        sections = _extract_sections_smart(raw, filepath)
        
        try:
            old_data = target_col.get(where={"filename": filename}, include=[])
            if old_data["ids"]:
                target_col.delete(ids=old_data["ids"])
        except Exception:
            pass
            
        ids, texts, metas = [], [], []
        for sec in sections:
            path_hash = hashlib.md5(filepath.encode("utf-8")).hexdigest()[:6]
            chunk_hash = hashlib.md5(sec["text"].encode("utf-8")).hexdigest()[:10]
            chunk_id = f"{path_obj.stem}_{path_hash}____{chunk_hash}"
            chunk_id = re.sub(r"[^a-zA-Z0-9_]", "_", chunk_id)
            
            ids.append(chunk_id)
            texts.append(sec["text"])
            metas.append({
                "source": sec["source"],
                "filename": sec["filename"],
                "heading": sec["heading"],
                "parent_heading": sec["parent_heading"],
                "category": category,
                "word_count": sec["word_count"]
            })
            
        if ids:
            target_col.add(ids=ids, documents=texts, metadatas=metas)
            
        all_data = target_col.get(include=["documents"])
        _build_bm25(all_data["ids"], all_data["documents"], bm25_cache_path)
        
        try:
            hash_cache = _load_hash_cache(hash_cache_path)
            new_hash = _get_file_hash(filepath)
            norm_path = os.path.normpath(filepath)
            hash_cache[norm_path] = new_hash
            # Safe call because we are already in _index_lock, wait: _save_hash_cache acquires lock again!
            # Since threading.Lock() is NOT re-entrant, we must write directly!
            hash_cache_path.parent.mkdir(parents=True, exist_ok=True)
            hash_cache_path.write_text(json.dumps(hash_cache, indent=2), encoding="utf-8")
        except Exception:
            pass
            
        return len(ids)

