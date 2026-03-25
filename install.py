"""
Markdown RAG MCP Server — Quick Install Script
Installs dependencies, downloads models, and auto-configures your IDE.
"""

import sys
import os
import json
import subprocess
import platform

REQUIRED_PYTHON = (3, 10)
PACKAGES = [
    "chromadb",
    "sentence-transformers",
    "fastmcp",
    "rank-bm25",
    "fastapi",
    "uvicorn",
    "httpx",
    "beautifulsoup4",
    "html2text",
    "lxml",
    "trafilatura",
]
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# Known MCP config paths per IDE (Windows / macOS / Linux)
CONFIG_LOCATIONS = {
    "Antigravity": {
        "Windows": os.path.expandvars(r"%USERPROFILE%\.gemini\antigravity\mcp_config.json"),
        "Darwin":  os.path.expanduser("~/.gemini/antigravity/mcp_config.json"),
        "Linux":   os.path.expanduser("~/.gemini/antigravity/mcp_config.json"),
    },
    "Claude Desktop": {
        "Windows": os.path.expandvars(r"%APPDATA%\Claude\claude_desktop_config.json"),
        "Darwin":  os.path.expanduser("~/Library/Application Support/Claude/claude_desktop_config.json"),
        "Linux":   os.path.expanduser("~/.config/Claude/claude_desktop_config.json"),
    },
    "Windsurf": {
        "Windows": os.path.expanduser("~/.codeium/windsurf/mcp_config.json"),
        "Darwin":  os.path.expanduser("~/.codeium/windsurf/mcp_config.json"),
        "Linux":   os.path.expanduser("~/.codeium/windsurf/mcp_config.json"),
    },
}


def check_python():
    version = sys.version_info
    if version < REQUIRED_PYTHON:
        print(f"  ERROR: Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}+ required.")
        print(f"         You have Python {version.major}.{version.minor}")
        sys.exit(1)
    print(f"  [OK] Python {version.major}.{version.minor}.{version.micro}")


def install_packages():
    print("\n[2/4] Installing Python packages...")
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(req_file):
        print("  Installing from requirements.txt... ", end="", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", req_file, "--quiet"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("OK")
        else:
            print(f"FAILED\n{result.stderr}")
            sys.exit(1)
    else:
        for pkg in PACKAGES:
            print(f"  Installing {pkg}...", end=" ", flush=True)
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print("OK")
            else:
                print(f"FAILED\n{result.stderr}")
                sys.exit(1)


def download_models():
    print("\n[3/4] Downloading AI models (one-time, ~2.3GB)...")
    print("  This may take 5-10 minutes on first run.")

    try:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
        from sentence_transformers import SentenceTransformer, CrossEncoder

        print(f"\n  [1/2] {EMBED_MODEL} (~1.1GB)...", flush=True)
        SentenceTransformer(EMBED_MODEL)
        print("        Done.")

        print(f"\n  [2/2] {RERANK_MODEL} (~1.1GB)...", flush=True)
        CrossEncoder(RERANK_MODEL)
        print("        Done.")

    except Exception as e:
        print(f"\n  WARNING: Could not pre-download models: {e}")
        print("  Models will download on first search instead.")


def _build_server_entry():
    """Build the MCP config entry with real absolute paths."""
    server_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "server.py"))
    python_path = sys.executable

    entry = {
        "command": python_path if platform.system() == "Windows" else "python",
        "args": [server_path],
    }
    return entry


def _detect_existing_configs():
    """Detect which IDE config files already exist on disk."""
    system = platform.system()
    found = []
    for ide_name, paths in CONFIG_LOCATIONS.items():
        path = paths.get(system)
        if path and os.path.isfile(path):
            found.append((ide_name, path))
    return found


def _inject_into_config(config_path, entry):
    """
    Read existing MCP config, inject 'markdown-rag' server entry, write back.
    Returns True on success, False on failure.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        config = {}

    if "mcpServers" not in config:
        config["mcpServers"] = {}

    config["mcpServers"]["markdown-rag"] = entry

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return True


def auto_configure():
    """
    Auto-detect installed IDEs and offer to inject the MCP config entry.
    Always asks before writing. Shows existing entry if one is already configured.
    """
    print("\n[4/4] Configuring MCP connection...")

    entry = _build_server_entry()
    server_path = entry["args"][0]

    # --- Try auto-detect ---
    found = _detect_existing_configs()

    if found:
        print(f"\n  Detected {len(found)} IDE config(s):\n")
        for i, (ide, path) in enumerate(found, 1):
            print(f"    {i}. {ide}: {path}")

        print()
        for ide, path in found:
            try:
                # Check if markdown-rag already exists
                with open(path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                existing_entry = existing.get("mcpServers", {}).get("markdown-rag")

                if existing_entry:
                    existing_path = existing_entry.get("args", [""])[0]
                    if os.path.normpath(existing_path) == os.path.normpath(server_path):
                        print(f"  [OK] {ide} — already configured with this server. Skipping.")
                        continue

                    # Different path — warn user
                    print(f"  [!!] {ide} — 'markdown-rag' already exists:")
                    print(f"       Current : {existing_path}")
                    print(f"       New     : {server_path}")
                    answer = input(f"       Overwrite? (y/N): ").strip().lower()
                    if answer != "y":
                        print(f"       Skipped. Existing config preserved.")
                        continue
                else:
                    # No existing entry — ask to add
                    answer = input(f"  Add 'markdown-rag' to {ide} config? (Y/n): ").strip().lower()
                    if answer == "n":
                        print(f"       Skipped.")
                        continue

                _inject_into_config(path, entry)
                print(f"  [OK] {ide} — config updated!")

            except Exception as e:
                print(f"  [!!] {ide} — could not process: {e}")
                print(f"       Please add manually to: {path}")
    else:
        print("  No IDE config files detected automatically.")

    # --- Always show the manual config as well ---
    server_path = entry["args"][0]
    command = entry["command"]

    # Format for JSON display
    if platform.system() == "Windows":
        server_display = server_path.replace("\\", "\\\\")
        command_display = command.replace("\\", "\\\\")
    else:
        server_display = server_path
        command_display = command

    manual_config = f'''{{
  "mcpServers": {{
    "markdown-rag": {{
      "command": "{command_display}",
      "args": ["{server_display}"]
    }}
  }}
}}'''

    print("\n" + "=" * 60)
    print("INSTALLATION COMPLETE!")
    print("=" * 60)

    if found:
        print(f"\n  Your config was auto-injected into {len(found)} IDE(s).")
        print("  Just RESTART your IDE and you're ready to go!\n")
    else:
        print("\n  Copy this into your IDE's MCP config file:\n")
        print(manual_config)
        print("\n  Config file locations:")
        if platform.system() == "Windows":
            print(r"    Antigravity : %USERPROFILE%\.gemini\antigravity\mcp_config.json")
            print(r"    Claude      : %APPDATA%\Claude\claude_desktop_config.json")
            print(r"    Windsurf    : %USERPROFILE%\.codeium\windsurf\mcp_config.json")
        else:
            print("    Antigravity : ~/.gemini/antigravity/mcp_config.json")
            print("    Claude      : ~/Library/Application Support/Claude/claude_desktop_config.json")
            print("    Windsurf    : ~/.codeium/windsurf/mcp_config.json")
        print()

    print("  After restarting your IDE, ask your AI agent:")
    print('    > "Index my documentation"     ← scans parent folder for .md files')
    print('    > "Search: how does auth work?" ← semantic search')
    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("  Markdown RAG MCP Server — Installer")
    print("=" * 60)

    print("\n[1/4] Checking Python version...")
    check_python()
    install_packages()
    download_models()
    auto_configure()
