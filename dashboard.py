"""
RAG MCP Server — Web Dashboard (localhost:6280)

Provides a visual management interface for:
- Viewing system status (GPU, models, collections)
- Browsing collections and indexed files
- Adding new sources (local paths, URLs, GitHub, npm, PyPI)
- Searching the knowledge base
- Managing collections (delete, reindex, remove files)

Usage:
    Standalone:  python dashboard.py
    Embedded:    Set RAG_DASHBOARD=true in env (starts with server.py)
"""

import os
import sys
import json
import asyncio
import threading
from pathlib import Path

# Ensure server module is importable
sys.path.insert(0, str(Path(__file__).parent))

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
except ImportError:
    print("Dashboard requires: pip install fastapi uvicorn")
    sys.exit(1)

# Import server functions (shared logic with MCP tools)
import server
import search_engine

app = FastAPI(title="RAG Dashboard", docs_url=None, redoc_url=None)

STATIC_DIR = Path(__file__).parent / "static"


# ── API Endpoints ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the SPA."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>Dashboard files not found</h1><p>Ensure static/index.html exists.</p>", 404)
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


@app.get("/api/status")
def api_status():
    """System status: GPU, models, BM25, cross-encoder."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            vram_used = round(torch.cuda.memory_allocated(0) / 1e9, 1)
            gpu = {"available": True, "name": gpu_name, "vram_total": vram_total, "vram_used": vram_used}
        else:
            gpu = {"available": False, "name": "CPU only"}
    except ImportError:
        gpu = {"available": False, "name": "torch not installed"}

    return {
        "embed_model": server.EMBED_MODEL,
        "rerank_model": server.RERANK_MODEL,
        "device": server.DEVICE,
        "gpu": gpu,
        "bm25_loaded": search_engine._bm25_index is not None,
        "bm25_chunks": len(search_engine._bm25_corpus) if search_engine._bm25_corpus else 0,
        "cross_encoder_loaded": search_engine._cross_encoder is not None,
        "db_path": server.DB_PATH,
    }


@app.get("/api/collections")
def api_collections():
    """List all collections with stats."""
    try:
        collections = server.client.list_collections()
    except Exception:
        collections = []

    result = []
    for col in collections:
        try:
            count = col.count()
            # Get file stats
            get_res = col.get(include=["metadatas"])
            meta = get_res.get("metadatas") or []
            files = {}
            categories = {}
            for m in meta:
                if not m: m = {}
                fname = m.get("filename", "unknown")
                cat = m.get("category", "other")
                files[fname] = files.get(fname, 0) + 1
                categories[cat] = categories.get(cat, 0) + 1
            result.append({
                "name": col.name,
                "chunks": count,
                "files": len(files),
                "categories": categories,
                "top_files": sorted(files.items(), key=lambda x: -x[1])[:10],
            })
        except Exception:
            result.append({"name": col.name, "chunks": 0, "files": 0, "categories": {}, "top_files": []})
    return result


@app.get("/api/collections/{name}/files")
def api_collection_files(name: str):
    """List files in a specific collection."""
    try:
        col = server._get_collection(name)
        if col.count() == 0:
            return []
        get_res = col.get(include=["metadatas"])
        meta = get_res.get("metadatas") or []
        files = {}
        for m in meta:
            if not m: m = {}
            fname = m.get("filename", "unknown")
            source = m.get("source", "")
            cat = m.get("category", "")
            wc = m.get("word_count", 0)
            if fname not in files:
                files[fname] = {"filename": fname, "source": source, "category": cat, "chunks": 0, "words": 0}
            files[fname]["chunks"] += 1
            files[fname]["words"] += wc
        return sorted(files.values(), key=lambda x: x["filename"])
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.post("/api/index")
async def api_index(request: Request):
    """Index a local path or URL."""
    body = await request.json()
    source = body.get("source", "").strip()
    collection = body.get("collection", server.DEFAULT_COLLECTION)

    if not source:
        return JSONResponse({"error": "source is required"}, 400)

    try:
        # Detect source type
        if source.startswith(("http://", "https://", "github://", "npm://", "pypi://")):
            result = await asyncio.to_thread(server.index_url, uri=source, collection=collection)
        else:
            # Local path — heavy CPU+IO, run in threadpool
            result = await asyncio.to_thread(server.index_documents, docs_path=source, collection=collection)
        return {"result": result}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, 500)


@app.post("/api/search")
async def api_search(request: Request):
    """Search across indexed documents."""
    body = await request.json()
    query = body.get("query", "").strip()
    n_results = body.get("n_results", 5)
    category = body.get("category")
    filename = body.get("filename")
    collection = body.get("collection", server.DEFAULT_COLLECTION)

    if not query:
        return JSONResponse({"error": "query is required"}, 400)

    try:
        # Heavy CPU: embedding generation + BM25 + cross-encoder reranking — run in threadpool
        result = await asyncio.to_thread(
            server.search_docs,
            query=query,
            n_results=n_results,
            category=category,
            filename=filename,
            collection=collection,
        )
        return {"result": result}
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.delete("/api/collections/{name}")
def api_delete_collection(name: str):
    """Delete a collection."""
    try:
        result = server.delete_collection(name=name, confirm=True)
        return {"result": result}
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.delete("/api/sources/{filename}")
def api_remove_source(filename: str, collection: str = None):
    """Remove a file from the index."""
    col = collection or server.DEFAULT_COLLECTION
    try:
        result = server.remove_source(filename=filename, collection=col)
        return {"result": result}
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.post("/api/reindex")
async def api_reindex(request: Request):
    """Force full reindex."""
    body = await request.json()
    docs_path = body.get("docs_path", server.DOCS_PATH)
    collection = body.get("collection", server.DEFAULT_COLLECTION)
    try:
        # Heavy CPU+IO — run in threadpool
        result = await asyncio.to_thread(server.reindex_collection, docs_path=docs_path, collection=collection)
        return {"result": result}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, 500)


@app.get("/api/browse")
async def api_browse(type: str = "folder"):
    """Open a native OS file dialog. Returns empty path if not supported (headless/WSL)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        import asyncio

        def _open_dialog():
            try:
                root = tk.Tk()
                root.attributes("-topmost", True)
                root.withdraw()
                path = ""
                try:
                    if type == "folder":
                        path = filedialog.askdirectory(
                            parent=root, title="Select Directory to Index", mustexist=True
                        )
                    else:
                        path = filedialog.askopenfilename(
                            parent=root,
                            title="Select File to Index",
                            filetypes=[
                                ("Supported Files", "*.md;*.txt;*.pdf;*.csv;*.docx;*.py;*.js;*.ts;*.html;*.json;*.yaml"),
                                ("All Files", "*.*"),
                            ],
                        )
                finally:
                    root.destroy()
                return path
            except Exception:
                return ""

        path = await asyncio.to_thread(_open_dialog)
        return {"path": path, "supported": True}

    except ImportError:
        return {"path": "", "supported": False, "error": "tkinter not available"}
    except Exception as e:
        return {"path": "", "supported": False, "error": str(e)}


@app.get("/api/gpu-check")
def api_gpu_check():
    """Detailed GPU compatibility check."""
    info = {
        "torch_installed": False,
        "cuda_available": False,
        "cuda_version": None,
        "gpu_name": None,
        "vram_total_gb": None,
        "vram_used_gb": None,
        "current_device": server.DEVICE,
        "recommendation": "",
        "how_to_enable": "",
    }
    try:
        import torch
        info["torch_installed"] = True
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            used = torch.cuda.memory_allocated(0) / 1e9
            info["vram_total_gb"] = round(total, 1)
            info["vram_used_gb"] = round(used, 1)
            if server.DEVICE == "cuda":
                info["recommendation"] = "GPU is active. Maximum performance."
            else:
                info["recommendation"] = "GPU available but not active."
                info["how_to_enable"] = 'Set RAG_DEVICE=cuda in your environment or MCP config env block.'
        else:
            info["recommendation"] = "No CUDA GPU detected. Running on CPU."
            info["how_to_enable"] = "Install CUDA-enabled PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --upgrade --force-reinstall"
    except ImportError:
        info["recommendation"] = "PyTorch not installed."
        info["how_to_enable"] = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --upgrade --force-reinstall"
    return info


@app.post("/api/index-file")
async def api_index_file(request: Request):
    """Index a single file directly (not the whole parent folder)."""
    body = await request.json()
    filepath = body.get("filepath", "").strip()
    collection = body.get("collection", server.DEFAULT_COLLECTION)

    if not filepath:
        return JSONResponse({"error": "filepath is required"}, 400)

    path = Path(filepath)
    if not path.exists():
        return JSONResponse({"error": f"File not found: {filepath}"}, 404)
    if not path.is_file():
        return JSONResponse({"error": "Path must be a file, not a directory. Use /api/index for folders."}, 400)

    try:
        from exceptions import OmniDocsError
        try:
            # Heavy CPU+IO — run in threadpool
            chunk_count = await asyncio.to_thread(server.index_single_file, filepath, collection)
        except OmniDocsError as e:
            return JSONResponse({"error": str(e)}, 400)
            
        return {"result": f"Indexed {chunk_count} chunks from {path.name} into '{collection}'"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, 500)


# ── Startup Modes ──────────────────────────────────────────────

DASHBOARD_PORT = int(os.getenv("RAG_DASHBOARD_PORT", "6280"))


def start_dashboard_thread(port: int = DASHBOARD_PORT):
    """Start dashboard as a background thread (for embedding in server.py)."""
    import sys
    def _run():
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="error")
    t = threading.Thread(target=_run, daemon=True, name="rag-dashboard")
    t.start()
    sys.stderr.write(f"📊 Dashboard: http://localhost:{port}\n")
    return t


if __name__ == "__main__":
    print(f"📊 RAG Dashboard starting on http://localhost:{DASHBOARD_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=DASHBOARD_PORT, log_level="info")
