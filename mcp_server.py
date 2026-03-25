"""
FastCode MCP Server - Expose repo-level code understanding via MCP protocol.

Usage:
    python mcp_server.py                    # stdio transport (default)
    python mcp_server.py --transport sse    # SSE transport (default host/port)
    python mcp_server.py --transport streamable-http --host 0.0.0.0 --port 5555

MCP config example (for Claude Code / Cursor):
    {
      "mcpServers": {
        "fastcode": {
          "command": "python",
          "args": ["/path/to/FastCode/mcp_server.py"],
          "env": {
            "MODEL": "your-model",
            "BASE_URL": "your-base-url",
            "OPENAI_API_KEY": "your-api-key"
          }
        }
      }
    }
"""

import os
import sys
import logging
import asyncio
import uuid
import inspect
import hashlib
import subprocess
import pickle
import yaml
from typing import Optional, List, Set

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Logging (file only – stdout is reserved for MCP JSON-RPC in stdio mode)
# ---------------------------------------------------------------------------
log_dir = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(os.path.join(log_dir, "mcp_server.log"))],
)
logger = logging.getLogger("fastcode.mcp")

# ---------------------------------------------------------------------------
# Lazy FastCode singleton
# ---------------------------------------------------------------------------
_fastcode_instance = None


def _get_fastcode():
    """Lazy-init the FastCode engine (heavy imports happen here)."""
    global _fastcode_instance
    if _fastcode_instance is None:
        logger.info("Initializing FastCode engine …")
        from fastcode import FastCode

        _fastcode_instance = FastCode()
        logger.info("FastCode engine ready.")
    return _fastcode_instance


def _repo_name_from_source(source: str, is_url: bool) -> str:
    """Derive a canonical repo name from a URL or local path."""
    if is_url:
        return _repo_name_from_url(source)

    normalized = _normalize_local_source(source)
    base_name = os.path.basename(os.path.normpath(normalized)) or "local_repo"
    wt_info = _detect_worktree_context(normalized)
    if wt_info:
        # Worktree-safe key: stable per worktree path/common-dir context.
        token = wt_info["token"]
        return f"{base_name}__wt__{token}"

    return base_name


def _legacy_repo_name_from_source(source: str, is_url: bool) -> str:
    """Legacy repo key for compatibility with pre-worktree indexes."""
    if is_url:
        return _repo_name_from_url(source)
    normalized = _normalize_local_source(source)
    return os.path.basename(os.path.normpath(normalized)) or "local_repo"


def _normalize_local_source(source: str) -> str:
    """Normalize local source path (expand ~/$VARS + absolute path)."""
    expanded = os.path.expandvars(os.path.expanduser((source or "").strip()))
    return os.path.realpath(os.path.abspath(expanded))


def _repo_name_from_url(url: str) -> str:
    """Extract repository name from URL in a lightweight way."""
    cleaned = (url or "").strip().rstrip("/")
    if cleaned.endswith(".git"):
        cleaned = cleaned[:-4]
    parts = cleaned.split("/")
    return parts[-1] if parts else "unknown_repo"


def _get_persist_dir() -> str:
    """Resolve vector-store persist directory without FastCode initialization."""
    config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    default_rel = os.path.join("data", "vector_store")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f) or {}
    except Exception:
        return os.path.join(PROJECT_ROOT, default_rel)

    vector_cfg = raw_cfg.get("vector_store", {}) if isinstance(raw_cfg, dict) else {}
    persist = vector_cfg.get("persist_directory", default_rel)
    if os.path.isabs(persist):
        return os.path.realpath(persist)
    return os.path.realpath(os.path.join(PROJECT_ROOT, persist))


def _scan_available_indexes_light() -> list[dict]:
    """Lightweight index scanner that avoids FastCode/model initialization."""
    persist_dir = _get_persist_dir()
    if not os.path.isdir(persist_dir):
        return []

    repos = []
    for file_name in os.listdir(persist_dir):
        if not file_name.endswith(".faiss"):
            continue
        name = file_name[:-6]
        metadata_file = os.path.join(persist_dir, f"{name}_metadata.pkl")
        if not os.path.exists(metadata_file):
            continue

        element_count = 0
        size_mb = 0.0
        try:
            size_mb = (
                os.path.getsize(os.path.join(persist_dir, file_name))
                + os.path.getsize(metadata_file)
            ) / (1024 * 1024)
            with open(metadata_file, "rb") as f:
                meta_data = pickle.load(f)
            element_count = len(meta_data.get("metadata", []))
        except Exception:
            pass

        repos.append(
            {
                "name": name,
                "element_count": element_count,
                "size_mb": round(size_mb, 2),
            }
        )

    return sorted(repos, key=lambda x: x["name"])


def _run_git(path: str, *args: str) -> Optional[str]:
    """Run git command with -C path; return stripped stdout or None."""
    try:
        proc = subprocess.run(
            ["git", "-C", path, *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    if proc.returncode != 0:
        return None

    out = (proc.stdout or "").strip()
    return out or None


def _detect_worktree_context(path: str) -> Optional[dict]:
    """
    Detect git worktree context for a local path.

    Returns:
        {"toplevel": str, "common_dir": str, "branch": str, "token": str}
        or None when the path is not inside a git worktree.
    """
    toplevel = _run_git(path, "rev-parse", "--show-toplevel")
    if not toplevel:
        return None

    common_dir = _run_git(path, "rev-parse", "--git-common-dir")
    if not common_dir:
        common_dir = os.path.join(toplevel, ".git")
    elif not os.path.isabs(common_dir):
        common_dir = os.path.abspath(os.path.join(toplevel, common_dir))

    absolute_git_dir = _run_git(path, "rev-parse", "--absolute-git-dir")
    if absolute_git_dir:
        absolute_git_dir = os.path.realpath(absolute_git_dir)
    else:
        absolute_git_dir = os.path.realpath(common_dir)

    branch = _run_git(path, "rev-parse", "--abbrev-ref", "HEAD") or "detached"

    token_src = f"{absolute_git_dir}::{os.path.realpath(toplevel)}"
    token = hashlib.sha1(token_src.encode("utf-8")).hexdigest()[:10]

    return {
        "toplevel": os.path.realpath(toplevel),
        "common_dir": os.path.realpath(common_dir),
        "branch": branch,
        "token": token,
    }


def _is_path_like(value: str) -> bool:
    """Heuristic: whether a repo query likely represents a local path."""
    v = (value or "").strip()
    if not v:
        return False
    return (
        os.path.isabs(v)
        or v.startswith(".")
        or v.startswith("~")
        or os.sep in v
        or (os.altsep is not None and os.altsep in v)
    )


def _set_loaded_repo_identity(fc, repo_name: str) -> None:
    """Override loaded repository identity before indexing."""
    if getattr(fc, "loader", None) is not None:
        fc.loader.repo_name = repo_name
    if isinstance(getattr(fc, "repo_info", None), dict):
        fc.repo_info["name"] = repo_name


def _indexed_repo_names(fc) -> list[str]:
    """Get all currently indexed repository names."""
    available = _scan_available_indexes_light()
    return [r.get("name", r.get("repo_name", "")) for r in available if r]


def _resolve_repo_query(repo_query: str) -> tuple[Optional[str], Optional[str]]:
    """
    Resolve user repo query (name/path/worktree key) to an indexed repo name.

    Returns:
        (resolved_repo_name, error_message)
    """
    indexed = [r.get("name", "") for r in _scan_available_indexes_light()]
    if not indexed:
        return None, "No indexed repositories found."

    q = (repo_query or "").strip()
    if not q:
        return None, "Repository name/path cannot be empty."

    if q in indexed:
        return q, None

    # Path-aware exact candidate first
    path_candidates: list[str] = []
    if _is_path_like(q):
        normalized = _normalize_local_source(q)
        if os.path.isdir(normalized):
            preferred = _repo_name_from_source(normalized, is_url=False)
            legacy = _legacy_repo_name_from_source(normalized, is_url=False)
            path_candidates = [c for c in [preferred, legacy] if c in indexed]
            if len(path_candidates) == 1:
                return path_candidates[0], None
            if len(path_candidates) > 1:
                choices = ", ".join(sorted(path_candidates))
                return None, (
                    f"Repository path '{repo_query}' maps to multiple indexed repositories. "
                    f"Use one of: {choices}"
                )

    # Name fallback: exact legacy basename / worktree key prefix
    q_lower = q.lower()
    basename_matches = [n for n in indexed if n.lower() == q_lower]
    if basename_matches:
        if len(basename_matches) == 1:
            return basename_matches[0], None
        choices = ", ".join(sorted(basename_matches))
        return None, (
            f"Repository name '{repo_query}' is ambiguous. Use one of: {choices}"
        )

    worktree_matches = [n for n in indexed if n.lower().startswith(f"{q_lower}__wt__")]
    if len(worktree_matches) == 1:
        return worktree_matches[0], None
    if len(worktree_matches) > 1:
        choices = ", ".join(sorted(worktree_matches))
        return None, (
            f"Repository name '{repo_query}' is ambiguous across worktrees. "
            f"Use one of: {choices}"
        )

    # Convenience: if single indexed repo and query is '.' or current-directory-like.
    if q in {".", "./"} and len(indexed) == 1:
        return indexed[0], None

    return (
        None,
        f"Repository '{repo_query}' is not indexed. Use code_qa or reindex_repo first.",
    )


def _is_repo_indexed(repo_name: str) -> bool:
    """Check whether a repo already has a persisted FAISS index."""
    persist_dir = _get_persist_dir()
    faiss_path = os.path.join(persist_dir, f"{repo_name}.faiss")
    meta_path = os.path.join(persist_dir, f"{repo_name}_metadata.pkl")
    return os.path.exists(faiss_path) and os.path.exists(meta_path)


def _apply_forced_env_excludes(fc) -> None:
    """
    Force-ignore environment-related paths before indexing.

    Always excludes virtual environment folders. Optionally excludes
    site-packages when FASTCODE_EXCLUDE_SITE_PACKAGES=1.
    """
    repo_cfg = fc.config.setdefault("repository", {})
    ignore_patterns = list(repo_cfg.get("ignore_patterns", []))

    forced_patterns = [
        ".venv",
        "venv",
        ".env",
        "env",
        "**/.venv/**",
        "**/venv/**",
        "**/.env/**",
        "**/env/**",
    ]

    # Optional (opt-in): site-packages can be huge/noisy in some repos.
    if os.getenv("FASTCODE_EXCLUDE_SITE_PACKAGES", "0").lower() in {"1", "true", "yes"}:
        forced_patterns.extend(
            [
                "site-packages",
                "**/site-packages/**",
            ]
        )

    added = []
    for pattern in forced_patterns:
        if pattern not in ignore_patterns:
            ignore_patterns.append(pattern)
            added.append(pattern)

    repo_cfg["ignore_patterns"] = ignore_patterns
    # Keep loader in sync when FastCode instance already exists.
    fc.loader.ignore_patterns = ignore_patterns

    if added:
        logger.info(f"Added forced ignore patterns: {added}")


def _ensure_repos_ready(
    repos: List[str], allow_incremental: bool = True, ctx=None
) -> List[str]:
    """
    For each repo source string:
      - If already indexed → skip
      - If URL and not on disk → clone + index
      - If local path → load + index

    Returns the list of canonical repo names that are ready.
    """
    fc = _get_fastcode()
    _apply_forced_env_excludes(fc)
    ready_names: List[str] = []

    for source in repos:
        source_clean = (source or "").strip()
        if not source_clean:
            continue

        source_for_infer = source_clean
        if _is_path_like(source_clean):
            source_for_infer = _normalize_local_source(source_clean)

        resolved_is_url = fc._infer_is_url(source_for_infer)
        name = _repo_name_from_source(source_for_infer, resolved_is_url)
        legacy_name = _legacy_repo_name_from_source(source_for_infer, resolved_is_url)

        if not resolved_is_url:
            normalized_local = _normalize_local_source(source_for_infer)
            if not os.path.isdir(normalized_local):
                logger.error(f"Local path does not exist: {normalized_local}")
                continue

        # Already indexed
        indexed_name = name if _is_repo_indexed(name) else None
        if (
            indexed_name is None
            and legacy_name != name
            and _is_repo_indexed(legacy_name)
        ):
            indexed_name = legacy_name

        if indexed_name is not None:
            # Try incremental update for local repos
            if not resolved_is_url and allow_incremental:
                abs_path = _normalize_local_source(source_for_infer)
                if os.path.isdir(abs_path):
                    try:
                        result = fc.incremental_reindex(
                            indexed_name, repo_path=abs_path
                        )
                        if result and result.get("changes", 0) > 0:
                            logger.info(
                                f"Incremental update for '{indexed_name}': {result}"
                            )
                            # Force reload since on-disk data changed
                            fc.repo_indexed = False
                            fc.loaded_repositories.clear()
                    except Exception as e:
                        logger.warning(
                            f"Incremental reindex failed for '{indexed_name}': {e}"
                        )
            logger.info(f"Repo '{indexed_name}' ready.")
            ready_names.append(indexed_name)
            continue

        # Need to index
        logger.info(f"Repo '{name}' not indexed. Preparing …")

        if resolved_is_url:
            # Clone and index
            logger.info(f"Cloning {source_for_infer} …")
            fc.load_repository(source_for_infer, is_url=True)
            _set_loaded_repo_identity(fc, name)
        else:
            # Local path
            abs_path = _normalize_local_source(source_for_infer)
            fc.load_repository(abs_path, is_url=False)
            _set_loaded_repo_identity(fc, name)

        logger.info(f"Indexing '{name}' …")
        fc.index_repository(force=False)
        logger.info(f"Indexing '{name}' complete.")
        ready_names.append(name)

    return ready_names


def _ensure_loaded(fc, ready_names: List[str]) -> bool:
    """Ensure repos are loaded into memory (vectors + BM25 + graphs)."""
    if not fc.repo_indexed or set(ready_names) != set(fc.loaded_repositories.keys()):
        logger.info(f"Loading repos into memory: {ready_names}")
        return fc._load_multi_repo_cache(repo_names=ready_names)
    return True


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
MCP_SERVER_DESCRIPTION = (
    "Repo-level code understanding - ask questions about any codebase."
)
_fastmcp_kwargs = {}
try:
    sig = inspect.signature(FastMCP.__init__).parameters
    fastmcp_host = os.getenv("FASTMCP_HOST", "0.0.0.0")
    try:
        fastmcp_port = int(os.getenv("FASTMCP_PORT", "5555"))
    except ValueError:
        logger.warning("Invalid FASTMCP_PORT; falling back to 5555")
        fastmcp_port = 5555

    if "description" in sig:
        _fastmcp_kwargs["description"] = MCP_SERVER_DESCRIPTION
    if "port" in sig:
        _fastmcp_kwargs["port"] = fastmcp_port
    if "host" in sig:
        _fastmcp_kwargs["host"] = fastmcp_host
    if "stateless_http" in sig:
        _fastmcp_kwargs["stateless_http"] = True
except (TypeError, ValueError):
    pass

mcp = FastMCP("FastCode", **_fastmcp_kwargs)


@mcp.tool()
def code_qa(
    question: str,
    repos: list[str],
    multi_turn: bool = True,
    session_id: str | None = None,
) -> str:
    """Ask a question about one or more code repositories.

    This is the core tool for repo-level code understanding. FastCode will
    automatically clone (if URL) and index repositories that haven't been
    indexed yet, then answer your question using hybrid retrieval + LLM.

    Args:
        question: The question to ask about the code.
        repos: List of repository sources. Each can be:
               - A GitHub/GitLab URL (e.g. "https://github.com/user/repo")
               - A local filesystem path (e.g. "/home/user/projects/myrepo")
               If the repo is already indexed, it won't be re-indexed.
        multi_turn: Enable multi-turn conversation mode. When True, previous
                    Q&A context from the same session_id is used. Default: True.
        session_id: Session identifier for multi-turn conversations. If not
                    provided, a new session is created automatically. Pass the
                    same session_id across calls to continue a conversation.

    Returns:
        The answer to your question, with source references.
    """
    fc = _get_fastcode()

    # 1. Ensure all repos are indexed
    ready_names = _ensure_repos_ready(repos)
    if not ready_names:
        return "Error: None of the specified repositories could be loaded or indexed."

    # 2. Load indexed repos into memory (multi-repo merge)
    if not _ensure_loaded(fc, ready_names):
        return "Error: Failed to load repository indexes."

    # 3. Session management
    sid = session_id or str(uuid.uuid4())[:8]

    # 4. Query
    result = fc.query(
        question=question,
        # Always enforce repository filtering for both single-repo and
        # multi-repo queries to avoid cross-repo source leakage.
        repo_filter=ready_names,
        session_id=sid,
        enable_multi_turn=multi_turn,
    )

    answer = result.get("answer", "")
    sources = result.get("sources", [])

    # Format output
    parts = [answer]

    if sources:
        parts.append("\n\n---\nSources:")
        for s in sources[:]:
            file_path = s.get("file", s.get("relative_path", ""))
            repo = s.get("repo", s.get("repository", ""))
            name = s.get("name", "")
            start = s.get("start_line", "")
            end = s.get("end_line", "")
            if (not start or not end) and s.get("lines"):
                lines = str(s.get("lines", ""))
                if "-" in lines:
                    parsed_start, parsed_end = lines.split("-", 1)
                    start = start or parsed_start
                    end = end or parsed_end
            loc = f"L{start}-L{end}" if start and end else ""
            parts.append(
                f"  - {repo}/{file_path}:{loc} ({name})"
                if repo
                else f"  - {file_path}:{loc} ({name})"
            )

    parts.append(f"\n[session_id: {sid}]")
    return "\n".join(parts)


@mcp.tool()
def list_sessions() -> str:
    """List all existing conversation sessions.

    Returns a list of sessions with their IDs, titles (first query),
    turn counts, and timestamps. Useful for finding a session_id to
    continue a previous conversation.
    """
    fc = _get_fastcode()
    sessions = fc.list_sessions()

    if not sessions:
        return "No sessions found."

    lines = ["Sessions:"]
    for s in sessions:
        sid = s.get("session_id", "?")
        title = s.get("title", "Untitled")
        turns = s.get("total_turns", 0)
        mode = "multi-turn" if s.get("multi_turn", False) else "single-turn"
        lines.append(f'  - {sid}: "{title}" ({turns} turns, {mode})')

    return "\n".join(lines)


@mcp.tool()
def get_session_history(session_id: str) -> str:
    """Get the full conversation history for a session.

    Args:
        session_id: The session identifier to retrieve history for.

    Returns:
        The complete Q&A history of the session.
    """
    fc = _get_fastcode()
    history = fc.get_session_history(session_id)

    if not history:
        return f"No history found for session '{session_id}'."

    lines = [f"Session {session_id} history:"]
    for turn in history:
        turn_num = turn.get("turn_number", "?")
        query = turn.get("query", "")
        answer = turn.get("answer", "")
        # Truncate long answers for readability
        if len(answer) > 500:
            answer = answer[:500] + " …"
        lines.append(f"\n--- Turn {turn_num} ---")
        lines.append(f"Q: {query}")
        lines.append(f"A: {answer}")

    return "\n".join(lines)


@mcp.tool()
def delete_session(session_id: str) -> str:
    """Delete a conversation session and all its history.

    Args:
        session_id: The session identifier to delete.

    Returns:
        Confirmation message.
    """
    fc = _get_fastcode()
    success = fc.delete_session(session_id)
    if success:
        return f"Session '{session_id}' deleted."
    return f"Failed to delete session '{session_id}'. It may not exist."


@mcp.tool()
def list_indexed_repos() -> str:
    """List all repositories that have been indexed and are available for querying.

    Returns:
        A list of indexed repository names with metadata.
    """
    available = _scan_available_indexes_light()

    if not available:
        return "No indexed repositories found."

    lines = ["Indexed repositories:"]
    for repo in available:
        name = repo.get("name", repo.get("repo_name", "?"))
        elements = repo.get("element_count", repo.get("elements", "?"))
        size = repo.get("size_mb", "?")
        lines.append(f"  - {name} ({elements} elements, {size} MB)")

    return "\n".join(lines)


@mcp.tool()
def delete_repo_metadata(repo_name: str) -> str:
    """Delete indexed metadata for a repository while keeping source code.

    This removes vector/BM25/graph index artifacts and the repository's
    overview entry from repo_overviews.pkl, but does NOT delete source files
    from the configured repository workspace.

    Args:
        repo_name: Repository name to clean metadata for.

    Returns:
        Confirmation message with deleted artifacts and freed disk space.
    """
    fc = _get_fastcode()
    result = fc.remove_repository(repo_name, delete_source=False)

    deleted_files = result.get("deleted_files", [])
    freed_mb = result.get("freed_mb", 0)

    if not deleted_files:
        return (
            f"No metadata files found for repository '{repo_name}'. "
            "Source code was not modified."
        )

    lines = [f"Deleted metadata for repository '{repo_name}' (source code kept)."]
    lines.append(f"Freed: {freed_mb} MB")
    lines.append("Removed artifacts:")
    for fname in deleted_files:
        lines.append(f"  - {fname}")
    return "\n".join(lines)


@mcp.tool()
def search_symbol(
    symbol_name: str,
    repos: list[str],
    symbol_type: str | None = None,
) -> str:
    """Search for a symbol (function, class, method) by name across repositories.

    Finds definitions matching the given name with case-insensitive search.
    Results are ranked: exact match > prefix match > contains match (top 20).

    Args:
        symbol_name: Name of the symbol to search for (e.g. "FastCode", "query").
        repos: List of repository sources (URLs or local paths).
        symbol_type: Optional filter: "function", "class", "file", or "documentation".

    Returns:
        Matching definitions with file path, line range, and signature.
    """
    fc = _get_fastcode()
    ready_names = _ensure_repos_ready(repos, allow_incremental=False)
    if not ready_names:
        return "Error: None of the specified repositories could be loaded."
    if not _ensure_loaded(fc, ready_names):
        return "Error: Failed to load repository indexes."

    query_lower = symbol_name.lower()
    exact, prefix, contains = [], [], []

    for meta in fc.vector_store.metadata:
        name = meta.get("name", "")
        elem_type = meta.get("type", "")
        if elem_type == "repository_overview":
            continue
        if symbol_type and elem_type != symbol_type:
            continue

        name_lower = name.lower()
        if name_lower == query_lower:
            exact.append(meta)
        elif name_lower.startswith(query_lower):
            prefix.append(meta)
        elif query_lower in name_lower:
            contains.append(meta)

    ranked = (exact + prefix + contains)[:20]
    if not ranked:
        return f"No symbols matching '{symbol_name}' found."

    lines = [f"Found {len(ranked)} result(s) for '{symbol_name}':"]
    for meta in ranked:
        name = meta.get("name", "")
        etype = meta.get("type", "")
        repo = meta.get("repo_name", "")
        rel_path = meta.get("relative_path", "")
        start = meta.get("start_line", "")
        end = meta.get("end_line", "")
        sig = meta.get("signature", "")
        loc = f"L{start}-L{end}" if start and end else ""
        line = f"  - [{etype}] {name}"
        if sig:
            line += f"  |  {sig}"
        line += f"\n    {repo}/{rel_path}:{loc}" if repo else f"\n    {rel_path}:{loc}"
        lines.append(line)

    return "\n".join(lines)


@mcp.tool()
def get_repo_structure(repo_name: str) -> str:
    """Get the high-level structure and summary of an indexed repository.

    Returns the repository summary, directory tree, and language statistics.
    Does not require loading the full index into memory.

    Args:
        repo_name: Name of an indexed repository (see list_indexed_repos).

    Returns:
        Repository summary, directory structure, and language breakdown.
    """
    fc = _get_fastcode()
    resolved_repo_name, error = _resolve_repo_query(repo_name)
    if error:
        return error

    assert resolved_repo_name is not None

    if not _is_repo_indexed(resolved_repo_name):
        return (
            f"Repository '{repo_name}' is not indexed. "
            "Use code_qa or reindex_repo first."
        )

    overviews = fc.vector_store.load_repo_overviews()
    overview = overviews.get(resolved_repo_name)
    if not overview:
        return f"No overview found for repository '{resolved_repo_name}'. It may need re-indexing."

    metadata = overview.get("metadata", {})
    summary = metadata.get("summary", "No summary available.")
    structure_text = metadata.get("structure_text", "")
    file_structure = metadata.get("file_structure", {})
    languages = file_structure.get("languages", {})

    parts = [f"Repository: {resolved_repo_name}", ""]
    parts.append(f"Summary:\n{summary}")

    if languages:
        parts.append("\nLanguages:")
        for lang, count in sorted(languages.items(), key=lambda x: -x[1]):
            parts.append(f"  - {lang}: {count} files")

    if structure_text:
        parts.append(f"\nDirectory Structure:\n{structure_text}")

    return "\n".join(parts)


@mcp.tool()
def get_file_summary(file_path: str, repos: list[str]) -> str:
    """Get the structure summary of a specific file (classes, functions, imports).

    Args:
        file_path: Path to the file (e.g. "fastcode/main.py").
                   Flexible matching: endswith or contains.
        repos: List of repository sources to search in.

    Returns:
        File structure: classes (with methods), top-level functions, and import count.
    """
    fc = _get_fastcode()
    ready_names = _ensure_repos_ready(repos, allow_incremental=False)
    if not ready_names:
        return "Error: None of the specified repositories could be loaded."
    if not _ensure_loaded(fc, ready_names):
        return "Error: Failed to load repository indexes."

    # Find matching elements by relative_path
    matching = []
    for meta in fc.vector_store.metadata:
        rel = meta.get("relative_path", "")
        if meta.get("type") == "repository_overview":
            continue
        if rel.endswith(file_path) or file_path in rel:
            matching.append(meta)

    if not matching:
        return f"No elements found for file path '{file_path}'."

    files = [m for m in matching if m.get("type") == "file"]
    classes = [m for m in matching if m.get("type") == "class"]
    functions = [m for m in matching if m.get("type") == "function"]

    file_meta = files[0] if files else matching[0]
    actual_path = file_meta.get("relative_path", file_path)
    repo = file_meta.get("repo_name", "")

    parts = [f"File: {repo}/{actual_path}" if repo else f"File: {actual_path}"]

    if files:
        fm = files[0]
        parts.append(f"Language: {fm.get('language', '?')}")
        mi = fm.get("metadata", {})
        parts.append(
            f"Lines: {mi.get('total_lines', '?')} (code: {mi.get('code_lines', '?')})"
        )
        num_imports = mi.get("num_imports", 0)
        if num_imports:
            parts.append(f"Imports: {num_imports}")

    if classes:
        parts.append(f"\nClasses ({len(classes)}):")
        for c in classes:
            sig = c.get("signature", c.get("name", ""))
            mi = c.get("metadata", {})
            methods = mi.get("methods", [])
            loc = f"L{c.get('start_line', '')}-L{c.get('end_line', '')}"
            parts.append(f"  - {sig} ({loc})")
            for m in methods:
                parts.append(f"      .{m}")

    if functions:
        top_level = [
            f for f in functions if not f.get("metadata", {}).get("class_name")
        ]
        if top_level:
            parts.append(f"\nFunctions ({len(top_level)}):")
            for fn in top_level:
                sig = fn.get("signature", fn.get("name", ""))
                loc = f"L{fn.get('start_line', '')}-L{fn.get('end_line', '')}"
                parts.append(f"  - {sig} ({loc})")

    return "\n".join(parts)


def _walk_call_chain(
    gb,
    element_id: str,
    direction: str,
    hops_left: int,
    parts: list,
    indent: int = 2,
    visited: Optional[Set[str]] = None,
):
    """Recursively walk the call chain and format output."""
    if visited is None:
        visited = {element_id}

    neighbors = (
        gb.get_callers(element_id)
        if direction == "callers"
        else gb.get_callees(element_id)
    )

    if not neighbors:
        parts.append(f"{'  ' * indent}(none)")
        return

    for nid in neighbors:
        if nid in visited:
            continue
        visited.add(nid)
        elem = gb.element_by_id.get(nid)
        if elem:
            loc = (
                f"{elem.relative_path}:L{elem.start_line}" if elem.relative_path else ""
            )
            parts.append(f"{'  ' * indent}- {elem.name} [{loc}]")
            if hops_left > 1:
                _walk_call_chain(
                    gb, nid, direction, hops_left - 1, parts, indent + 1, visited
                )


@mcp.tool()
def get_call_chain(
    symbol_name: str,
    repos: list[str],
    direction: str = "both",
    max_hops: int = 2,
) -> str:
    """Trace the call chain for a function or method.

    Shows who calls this symbol (callers) and/or what it calls (callees),
    up to max_hops levels deep.

    Args:
        symbol_name: Name of the function/method to trace.
        repos: List of repository sources.
        direction: "callers", "callees", or "both" (default: "both").
        max_hops: Maximum depth of the call chain (default: 2, max: 5).

    Returns:
        Formatted call chain showing callers and/or callees.
    """
    fc = _get_fastcode()
    ready_names = _ensure_repos_ready(repos, allow_incremental=False)
    if not ready_names:
        return "Error: None of the specified repositories could be loaded."
    if not _ensure_loaded(fc, ready_names):
        return "Error: Failed to load repository indexes."

    max_hops = min(max_hops, 5)
    gb = fc.graph_builder
    name_lower = symbol_name.lower()
    target_id = None
    target_elem = None

    # Exact match via element_by_name
    elem = gb.element_by_name.get(symbol_name)
    if elem:
        target_elem, target_id = elem, elem.id

    # Fallback: case-insensitive search
    if not target_id:
        for eid, elem in gb.element_by_id.items():
            if elem.name.lower() == name_lower:
                target_elem, target_id = elem, eid
                break

    # Fallback: partial match
    if not target_id:
        for eid, elem in gb.element_by_id.items():
            if name_lower in elem.name.lower():
                target_elem, target_id = elem, eid
                break

    if not target_id or target_elem is None:
        return f"Symbol '{symbol_name}' not found in call graph."

    parts = [
        f"Call chain for '{target_elem.name}' ({target_elem.type})"
        f" at {target_elem.relative_path}:L{target_elem.start_line}"
    ]

    if direction in ("callers", "both"):
        parts.append("\n  Callers (who calls this):")
        _walk_call_chain(gb, target_id, "callers", max_hops, parts, indent=2)

    if direction in ("callees", "both"):
        parts.append("\n  Callees (what this calls):")
        _walk_call_chain(gb, target_id, "callees", max_hops, parts, indent=2)

    return "\n".join(parts)


@mcp.tool()
def reindex_repo(repo_source: str) -> str:
    """Force a full re-index of a repository.

    Clones (if URL) or loads (if local path) the repository and rebuilds
    all indexes from scratch.

    Args:
        repo_source: Repository URL or local filesystem path.

    Returns:
        Confirmation with element count.
    """
    fc = _get_fastcode()
    _apply_forced_env_excludes(fc)

    source_clean = (repo_source or "").strip()
    source_for_infer = source_clean
    if _is_path_like(source_clean):
        source_for_infer = _normalize_local_source(source_clean)

    resolved_is_url = fc._infer_is_url(source_for_infer)
    name = _repo_name_from_source(source_for_infer, resolved_is_url)
    logger.info(f"Force re-indexing '{name}' from {source_for_infer}")

    if resolved_is_url:
        fc.load_repository(source_for_infer, is_url=True)
        _set_loaded_repo_identity(fc, name)
    else:
        abs_path = _normalize_local_source(source_for_infer)
        if not os.path.isdir(abs_path):
            return f"Error: Local path does not exist: {abs_path}"
        fc.load_repository(abs_path, is_url=False)
        _set_loaded_repo_identity(fc, name)

    fc.index_repository(force=True)
    count = fc.vector_store.get_count()

    # Reset in-memory state so next _ensure_loaded does a clean load
    fc.repo_indexed = False
    fc.loaded_repositories.clear()

    return f"Successfully re-indexed '{name}': {count} elements indexed."


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    default_transport = os.getenv("FASTMCP_TRANSPORT", "stdio")
    if default_transport not in {"stdio", "sse", "streamable-http"}:
        logger.warning(
            "Invalid FASTMCP_TRANSPORT=%s; falling back to stdio",
            default_transport,
        )
        default_transport = "stdio"

    try:
        default_port = int(os.getenv("FASTMCP_PORT", "5555"))
    except ValueError:
        logger.warning("Invalid FASTMCP_PORT; falling back to 5555")
        default_port = 5555

    default_host = os.getenv("FASTMCP_HOST", "0.0.0.0")

    parser = argparse.ArgumentParser(description="FastCode MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default=default_transport,
        help=f"MCP transport (default: {default_transport})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=default_port,
        help=f"Port for HTTP transport (default: {default_port})",
    )
    parser.add_argument(
        "--host",
        default=default_host,
        help=f"Host for HTTP transport (default: {default_host})",
    )
    args = parser.parse_args()

    def _run_http_transport(transport: str) -> None:
        # FastMCP v2/v3 style: host/port are constructor settings, not run() params.
        if hasattr(mcp, "settings"):
            mcp.settings.host = args.host
            mcp.settings.port = args.port
        mcp.run(transport=transport)

    if args.transport == "sse":
        _run_http_transport("sse")
    elif args.transport == "streamable-http":
        _run_http_transport("streamable-http")
    else:
        mcp.run(transport="stdio")
