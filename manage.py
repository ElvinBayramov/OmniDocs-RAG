"""
RAG MCP Server — CLI Management Tool

Usage:
    python manage.py status                       # System status
    python manage.py list                         # List collections
    python manage.py files [collection]           # List files in a collection
    python manage.py index <path>                 # Index local path
    python manage.py index-url <url>              # Index URL / GitHub / npm / PyPI
    python manage.py search <query> [-n 5]        # Search
    python manage.py remove <filename> [-c col]   # Remove file from index
    python manage.py delete <collection>          # Delete collection
    python manage.py reindex [collection]          # Force full reindex
"""

import sys
import os
import argparse
from pathlib import Path

# Ensure server module is importable
sys.path.insert(0, str(Path(__file__).parent))


def cmd_status(_args):
    import server
    print(server.rag_status())


def cmd_list(_args):
    import server
    print(server.list_collections())


def cmd_files(args):
    import server
    col = args.collection or server.DEFAULT_COLLECTION
    print(server.list_indexed_files(collection=col))


def cmd_index(args):
    import server
    result = server.index_documents(docs_path=args.path, collection=args.collection or server.DEFAULT_COLLECTION)
    print(result)


def cmd_index_url(args):
    import server
    import asyncio
    result = asyncio.run(server.index_url(uri=args.url, collection=args.collection or server.DEFAULT_COLLECTION))
    print(result)


def cmd_search(args):
    import server
    result = server.search_docs(
        query=args.query,
        n_results=args.n,
        collection=args.collection or server.DEFAULT_COLLECTION,
    )
    print(result)


def cmd_remove(args):
    import server
    result = server.remove_source(filename=args.filename, collection=args.collection or server.DEFAULT_COLLECTION)
    print(result)


def cmd_delete(args):
    import server
    # Ask confirmation
    answer = input(f"Delete collection '{args.name}'? Type 'yes' to confirm: ")
    if answer.strip().lower() != "yes":
        print("Cancelled.")
        return
    result = server.delete_collection(name=args.name, confirm=True)
    print(result)


def cmd_reindex(args):
    import server
    col = args.collection or server.DEFAULT_COLLECTION
    path = args.path or server.DOCS_PATH
    result = server.reindex_collection(docs_path=path, collection=col)
    print(result)


def main():
    parser = argparse.ArgumentParser(
        prog="manage.py",
        description="RAG MCP Server — CLI Management",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # status
    sub.add_parser("status", help="Show system status")

    # list
    sub.add_parser("list", help="List all collections")

    # files
    p = sub.add_parser("files", help="List files in a collection")
    p.add_argument("collection", nargs="?", default=None, help="Collection name")

    # index
    p = sub.add_parser("index", help="Index local files")
    p.add_argument("path", help="Path to scan for files")
    p.add_argument("-c", "--collection", default=None, help="Target collection")

    # index-url
    p = sub.add_parser("index-url", help="Index URL / GitHub / npm / PyPI")
    p.add_argument("url", help="URI to index (https://, github://, npm://, pypi://)")
    p.add_argument("-c", "--collection", default=None, help="Target collection")

    # search
    p = sub.add_parser("search", help="Search the knowledge base")
    p.add_argument("query", help="Search query")
    p.add_argument("-n", type=int, default=5, help="Number of results (default: 5)")
    p.add_argument("-c", "--collection", default=None, help="Target collection")

    # remove
    p = sub.add_parser("remove", help="Remove a file from the index")
    p.add_argument("filename", help="Filename to remove")
    p.add_argument("-c", "--collection", default=None, help="Target collection")

    # delete
    p = sub.add_parser("delete", help="Delete an entire collection")
    p.add_argument("name", help="Collection name to delete")

    # reindex
    p = sub.add_parser("reindex", help="Force full reindex of a collection")
    p.add_argument("collection", nargs="?", default=None, help="Collection name")
    p.add_argument("--path", default=None, help="Path to reindex (default: RAG_DOCS_PATH env var)")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    commands = {
        "status": cmd_status,
        "list": cmd_list,
        "files": cmd_files,
        "index": cmd_index,
        "index-url": cmd_index_url,
        "search": cmd_search,
        "remove": cmd_remove,
        "delete": cmd_delete,
        "reindex": cmd_reindex,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
