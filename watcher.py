"""
File Watcher — Auto-reindex on filesystem changes (FEAT-09).

Uses watchdog to monitor a directory for file changes and triggers
incremental re-indexing with debounce (waits 2s after last change).

Usage:
    Set env vars before starting server.py:
        RAG_WATCH_PATH=C:/projects/my-app/docs
        RAG_WATCH_COLLECTION=my-project

    Or use programmatically:
        from watcher import start_watcher, stop_watcher
        observer = start_watcher("./docs", "my-collection")
        # ... later ...
        stop_watcher(observer)
"""

import logging
import threading
from pathlib import Path

logger = logging.getLogger("rag-watcher")

# Extensions worth watching (matches server.py SUPPORTED_EXTENSIONS)
WATCHED_EXTENSIONS = {
    # Text / markdown
    ".md", ".txt", ".rst", ".text", ".log",
    # Code
    ".py", ".js", ".ts", ".jsx", ".tsx", ".css", ".scss",
    ".java", ".go", ".rs", ".c", ".cpp", ".h", ".hpp",
    ".rb", ".php", ".swift", ".kt", ".lua", ".sh", ".bash",
    # Config
    ".json", ".yaml", ".yml", ".toml", ".xml", ".csv", ".ini", ".cfg",
    # Web
    ".html", ".htm",
    # Binary docs
    ".pdf", ".docx", ".xlsx", ".pptx", ".ipynb",
}


def start_watcher(docs_path: str, collection_name: str = "docs_v4",
                  debounce_seconds: float = 2.0):
    """
    Start watching a directory for file changes.
    Triggers incremental re-indexing after debounce period.

    Args:
        docs_path: Directory to watch (recursive)
        collection_name: ChromaDB collection to index into
        debounce_seconds: Wait time after last change before re-indexing

    Returns:
        Observer instance (call stop_watcher() to stop)
    """
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        logger.error(
            "watchdog not installed. Install with: pip install watchdog\n"
            "File watcher disabled."
        )
        return None

    class _DocsChangeHandler(FileSystemEventHandler):
        def __init__(self):
            self._timer: threading.Timer | None = None
            self._pending_files: set[str] = set()
            self._lock = threading.Lock()

        def on_modified(self, event):
            self._handle(event)

        def on_created(self, event):
            self._handle(event)

        def on_deleted(self, event):
            self._handle(event)

        def _handle(self, event):
            if event.is_directory:
                return
            path = Path(event.src_path)
            if path.suffix.lower() not in WATCHED_EXTENSIONS:
                return

            with self._lock:
                self._pending_files.add(str(path))
                # Debounce: reset timer on each change
                if self._timer:
                    self._timer.cancel()
                self._timer = threading.Timer(debounce_seconds, self._reindex)
                self._timer.daemon = True
                self._timer.start()

        def _reindex(self):
            with self._lock:
                count = len(self._pending_files)
                self._pending_files.clear()

            if count == 0:
                return

            logger.info(f"Auto-reindexing {count} changed files in '{docs_path}'...")
            try:
                # Import here to avoid circular imports
                from server import index_documents
                result = index_documents(
                    docs_path=docs_path,
                    collection=collection_name,
                )
                logger.info(f"Auto-reindex complete: {result[:200]}")
            except Exception as e:
                logger.error(f"Auto-reindex failed: {e}")

    handler = _DocsChangeHandler()
    observer = Observer()
    observer.schedule(handler, docs_path, recursive=True)
    observer.daemon = True
    observer.start()
    logger.info(f"📂 Watching: {docs_path} → collection '{collection_name}' (debounce={debounce_seconds}s)")
    return observer


def stop_watcher(observer):
    """Stop a running file watcher."""
    if observer is None:
        return
    try:
        observer.stop()
        observer.join(timeout=5)
        logger.info("File watcher stopped.")
    except Exception as e:
        logger.warning(f"Error stopping watcher: {e}")
