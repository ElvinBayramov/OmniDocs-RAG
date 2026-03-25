"""
OmniDocs RAG — Search Engine

Responsible for:
- Hybrid search pipeline (Vector + BM25 + RRF fusion)
- Cross-Encoder reranking
- Query expansion (multilingual RU/EN)
- Result formatting and deduplication
- BM25 index building and persistence
"""

import re
import pickle
import threading
from pathlib import Path
from typing import Optional
from difflib import SequenceMatcher


# ──────────────────────────────────────────────
# BM25 Index State
# ──────────────────────────────────────────────
_bm25_index = None
_bm25_corpus = None   # list of (chunk_id, text)
_bm25_loaded_from = None  # tracks how BM25 was initialized
_bm25_lock = threading.Lock()

# Cross-encoder (lazy loaded)
_cross_encoder = None


def _get_cross_encoder(rerank_model: str):
    """Lazy-load the cross-encoder reranker (~1.1GB on first use)."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(rerank_model)
    return _cross_encoder


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in text.split() if len(w) > 2]


def _build_bm25(ids: list[str], texts: list[str], cache_path: Path):
    """Build BM25 index and persist to disk for restart survival."""
    global _bm25_index, _bm25_corpus, _bm25_loaded_from
    with _bm25_lock:
        if not ids:
            _bm25_index = None
            _bm25_corpus = None
            return
        from rank_bm25 import BM25Okapi
        tokenized = [_tokenize(t) for t in texts]
        _bm25_index = BM25Okapi(tokenized)
        _bm25_corpus = list(zip(ids, texts))
        _bm25_loaded_from = "indexed"
        # Persist to disk so BM25 survives server restarts
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump({"corpus": _bm25_corpus, "tokenized": tokenized}, f)
        except Exception:
            pass  # non-critical: BM25 still works in-memory


def load_bm25_on_startup(cache_path: Path, collection):
    """Restore BM25 index from disk cache if available and still valid."""
    global _bm25_index, _bm25_corpus, _bm25_loaded_from
    if not cache_path.exists():
        return
    try:
        from rank_bm25 import BM25Okapi
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        # Validate: cache must match current ChromaDB collection size
        chroma_count = collection.count()
        cache_count = len(data["corpus"])
        if chroma_count > 0 and cache_count != chroma_count:
            return  # stale cache — skip, will rebuild on next index_documents()
        _bm25_index = BM25Okapi(data["tokenized"])
        _bm25_corpus = data["corpus"]
        _bm25_loaded_from = "cache"
    except Exception:
        pass  # corrupt cache — will rebuild on next index_documents()


# ──────────────────────────────────────────────
# Query Expansion
# ──────────────────────────────────────────────
def _expand_query(query: str) -> list[str]:
    """
    Generate query variations to improve recall.
    Especially important for RU documentation with EN API terms.
    """
    queries = [query]
    has_cyrillic = any('\u0400' <= c <= '\u04FF' for c in query)

    if has_cyrillic:
        queries.append(query.lower())
        api_synonyms = {
            "создать": "create post make new",
            "получить": "get fetch list read",
            "обновить": "update patch put",
            "удалить": "delete remove drop",
            "найти": "find search query",
            "добавить": "add insert append",
            "ошибка": "error exception fail bug",
        }
        en_terms = []
        for ru, en in api_synonyms.items():
            if ru in query.lower():
                en_terms.append(en)
        if en_terms:
            queries.append(" ".join(en_terms))

    return queries[:3]


# ──────────────────────────────────────────────
# Deduplication
# ──────────────────────────────────────────────
def _deduplicate_results(candidates: dict) -> dict:
    """Remove near-duplicates (>80% similar by first 200 chars)."""
    items = sorted(candidates.values(), key=lambda x: x.get("final_score", 0), reverse=True)
    kept = []
    for item in items:
        item_text = item["doc"][:200]
        is_dup = any(
            SequenceMatcher(None, item_text, k["doc"][:200]).ratio() > 0.8
            for k in kept
        )
        if not is_dup:
            kept.append(item)
    return {str(i): item for i, item in enumerate(kept)}


# ──────────────────────────────────────────────
# Result Formatting
# ──────────────────────────────────────────────
def _format_result(doc: str, meta: dict, score: float, rank: int, max_chars: int = 1500) -> str:
    """Format a single search result cleanly."""
    heading = meta.get("heading", "")
    parent = meta.get("parent_heading", "")
    filename = meta.get("filename", "unknown")
    category = meta.get("category", "")

    breadcrumb = filename
    if parent and parent != Path(filename).stem:
        breadcrumb += f" > {parent}"
    if heading and heading != parent:
        breadcrumb += f" > {heading}"

    content = doc.strip()
    # Remove overlap prefix markers
    content = re.sub(r"^\[\.\.\.]\s*", "", content)
    content = re.sub(r"\n{3,}", "\n\n", content)

    if len(content) > max_chars:
        # Don't truncate inside ```code blocks```
        code_block_end = content.find("```", max_chars - 200)
        if 0 < code_block_end < max_chars + 500:
            content = content[:code_block_end + 3]
        else:
            cut = content[:max_chars].rfind(". ")
            if cut > max_chars // 2:
                content = content[:cut + 1]
            else:
                content = content[:max_chars] + "..."

    return (
        f"### [{rank}] {breadcrumb}\n"
        f"**Relevance:** {score:.0%} | **Category:** {category}\n\n"
        f"{content}\n"
    )


# ──────────────────────────────────────────────
# Main Search Pipeline
# ──────────────────────────────────────────────
def search_docs(
    query: str,
    n_results: int = 5,
    category: Optional[str] = None,
    filename: Optional[str] = None,
    collection_name: str = "docs_v4",
    *,
    get_collection_fn,
    embed_model: str,
    device: str,
    rerank_model: str,
    max_ce_chars: int = 8000,
    max_result_chars: int = 1500,
) -> str:
    """
    Semantic search across indexed documentation.
    Uses Hybrid (Vector + BM25) + Cross-Encoder Reranking.
    """
    target_col = get_collection_fn(collection_name)
    count = target_col.count()
    if count == 0:
        return f"No documents indexed in collection '{collection_name}'. Call index_documents() first."

    # Validate inputs
    n_results = max(1, n_results)

    # ── Step 1: Vector search (ChromaDB) with Query Expansion ──
    where_filter = None
    conditions = []
    if category:
        conditions.append({"category": category})
    if filename:
        conditions.append({"filename": filename})

    if len(conditions) == 1:
        where_filter = conditions[0]
    elif len(conditions) > 1:
        where_filter = {"$and": conditions}

    fetch_count = min(n_results * 4, count)  # fetch more for reranking
    expanded_queries = _expand_query(query)
    candidates: dict[str, dict] = {}

    try:
        for q in expanded_queries:
            vector_results = target_col.query(
                query_texts=[q],
                n_results=fetch_count,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )

            if not vector_results["documents"][0]:
                continue

            result_ids = vector_results.get("ids", [[]])[0]
            for i, (doc, meta, dist) in enumerate(zip(
                vector_results["documents"][0],
                vector_results["metadatas"][0],
                vector_results["distances"][0]
            )):
                chunk_id = result_ids[i] if i < len(result_ids) else f"_vec_{i}"
                if chunk_id not in candidates:
                    candidates[chunk_id] = {
                        "doc": doc, "meta": meta,
                        "vec_score": 1 - dist, "bm25_score": 0.0,
                        "min_vec_rank": i
                    }
                else:
                    candidates[chunk_id]["min_vec_rank"] = min(
                        candidates[chunk_id]["min_vec_rank"], i
                    )
    except Exception as e:
        from exceptions import SearchError
        raise SearchError(f"Vector search failed: {str(e)}")

    if not candidates:
        return f"No results found for: '{query}'"

    # ── Step 2: BM25 keyword search ──
    if _bm25_index is not None and _bm25_corpus is not None:
        query_tokens = _tokenize(query)
        if query_tokens:
            bm25_scores = _bm25_index.get_scores(query_tokens)
            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0

            # Match BM25 results to vector candidates by chunk_id
            for idx, (chunk_id, text) in enumerate(_bm25_corpus):
                if chunk_id in candidates:
                    candidates[chunk_id]["bm25_score"] = bm25_scores[idx] / max_bm25

    # ── Step 3: Combine scores (RRF - Reciprocal Rank Fusion) ──
    vec_sorted = sorted(candidates.values(), key=lambda x: x.get("min_vec_rank", 999))
    for rank, c in enumerate(vec_sorted):
        c["vec_rank"] = rank + 1

    bm25_sorted = sorted(candidates.values(), key=lambda x: x["bm25_score"], reverse=True)
    for rank, c in enumerate(bm25_sorted):
        c["bm25_rank"] = rank + 1

    k = 60  # standard RRF constant
    for c in candidates.values():
        c["hybrid_score"] = (1.0 / (k + c["vec_rank"])) + (1.0 / (k + c["bm25_rank"]))

    hybrid_sorted = sorted(candidates.values(), key=lambda x: x["hybrid_score"], reverse=True)
    top_candidates = hybrid_sorted[:min(n_results * 2, len(hybrid_sorted))]

    # ── Step 4: Cross-Encoder Reranking ──
    try:
        cross_encoder = _get_cross_encoder(rerank_model)
        pairs = [(query, c["doc"][:max_ce_chars]) for c in top_candidates]
        ce_scores = cross_encoder.predict(pairs)

        min_ce = min(ce_scores)
        max_ce = max(ce_scores) if max(ce_scores) != min(ce_scores) else min(ce_scores) + 1
        for i, c in enumerate(top_candidates):
            c["ce_score"] = (ce_scores[i] - min_ce) / (max_ce - min_ce)

        # Final score: 40% hybrid + 60% cross-encoder
        h_scores = [c["hybrid_score"] for c in top_candidates]
        max_hybrid, min_hybrid = max(h_scores), min(h_scores)
        hybrid_range = max_hybrid - min_hybrid if max_hybrid != min_hybrid else 1.0

        for c in top_candidates:
            norm_hybrid = (c["hybrid_score"] - min_hybrid) / hybrid_range
            c["final_score"] = 0.4 * norm_hybrid + 0.6 * c["ce_score"]

    except Exception:
        # Fallback: use hybrid score directly
        if top_candidates:
            max_h = max(c["hybrid_score"] for c in top_candidates)
            for c in top_candidates:
                c["final_score"] = c["hybrid_score"] / max_h if max_h > 0 else 0

    # Sort by final score and apply deduplication
    final_sorted = sorted(top_candidates, key=lambda x: x["final_score"], reverse=True)
    deduped_candidates = _deduplicate_results({str(i): c for i, c in enumerate(final_sorted)})
    final_results = list(deduped_candidates.values())[:n_results]

    # ── Step 5: Format output ──
    output_parts = [f"## Results for: \"{query}\"\n"]

    if category or filename:
        filters = []
        if category:
            filters.append(f"category={category}")
        if filename:
            filters.append(f"file={filename}")
        output_parts.append(f"**Filters:** {', '.join(filters)}\n")

    output_parts.append(f"**Method:** Hybrid (Vector + BM25) + Cross-Encoder Reranking\n")

    for rank, c in enumerate(final_results, 1):
        output_parts.append(_format_result(c["doc"], c["meta"], c["final_score"], rank, max_result_chars))

    return "\n".join(output_parts)
