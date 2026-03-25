"""
OmniDocs RAG — Custom Exceptions

Provides structured error types for better error handling
across MCP tools, FastAPI routes, and CLI commands.
"""


class OmniDocsError(Exception):
    """Base class for all OmniDocs errors."""


class IndexingError(OmniDocsError):
    """Error during document indexing."""


class UnsupportedFormatError(OmniDocsError):
    """Unsupported file format for indexing."""


class SearchError(OmniDocsError):
    """Error during search execution."""


class CollectionError(OmniDocsError):
    """Error during collection operations."""
