---
name: fastcode
description: "FastCode MCP — repo-level code understanding with hybrid retrieval + LLM. Use for discovery, navigation, and answering questions about unfamiliar codebases."
compatibility: opencode

mcp:
  fastcode:
    type: http
    url: http://localhost:5555/mcp
---

## MCP Endpoint

- **URL**: `http://localhost:5555/mcp`
- **Transport**: `streamable-http`

## Available Tools

| Tool                                                        | Purpose                                                     |
| ----------------------------------------------------------- | ----------------------------------------------------------- |
| `code_qa(question, repos, multi_turn?, session_id?)`        | Ask source-backed code questions across one or more repos   |
| `list_sessions()`                                           | List existing conversation sessions                         |
| `get_session_history(session_id)`                           | Retrieve full conversation history for a session            |
| `delete_session(session_id)`                                | Delete a session                                            |
| `list_indexed_repos()`                                      | Show indexed repos available for querying                   |
| `delete_repo_metadata(repo_name)`                           | Clear index artifacts for a repo (keeps source code)        |
| `search_symbol(symbol_name, repos, symbol_type?)`           | Find symbol definitions (function/class/file/documentation) |
| `get_repo_structure(repo_name)`                             | Show high-level repo summary/tree/language breakdown        |
| `get_file_summary(file_path, repos)`                        | Show classes/functions/import stats for a file              |
| `get_call_chain(symbol_name, repos, direction?, max_hops?)` | Trace callers/callees from call graph                       |
| `reindex_repo(repo_source)`                                 | Force full re-index from scratch                            |

## Core Workflows

### 0. New Project Bootstrap (First-Time Setup)

Use this sequence when FastCode has never indexed the project yet:

```typescript
// 1) Trigger indexing for the current OpenCode project root
reindex_repo((repo_source = "."));

// 2) Confirm the repository now appears in indexed list
list_indexed_repos();

// 3) Copy the exact repo name from list_indexed_repos() and inspect structure
get_repo_structure((repo_name = "<exact-name-from-list>"));

// 4) Start normal Q&A on the current project
code_qa(
  (question = "Summarize this repository architecture"),
  (repos = ["."]),
  (multi_turn = true),
);
```

If you see `Repository '.' is not indexed`, run `reindex_repo(".")` first and retry with the exact indexed name from `list_indexed_repos()`.

### 1. Understanding an Unfamiliar Codebase

```typescript
// First: get repo overview from indexed metadata
get_repo_structure((repo_name = "myproject"));

// Then: inspect a specific file summary
get_file_summary((file_path = "src/main.py"), (repos = ["/path/to/repo"]));

// Finally: ask targeted architectural questions
code_qa(
  (question = "How does authentication work?"),
  (repos = ["/path/to/repo"]),
);
```

### 2. Finding Specific Symbols

```typescript
// Find symbol definitions
search_symbol(
  (symbol_name = "AuthService"),
  (repos = ["/path/to/repo"]),
  (symbol_type = "class"),
);

// Trace callers/callees
get_call_chain(
  (symbol_name = "validate_token"),
  (repos = ["/path/to/repo"]),
  (direction = "both"),
  (max_hops = 3),
);
```

### 3. Multi-Repository Analysis

```typescript
code_qa(
  (question = "How do these repos handle error propagation?"),
  (repos = ["/home/kevin/workspace/api", "/home/kevin/workspace/frontend"]),
  (multi_turn = true),
);
```

## Usage Patterns

### Repository Path Passing (Important)

- Always pass `repos` explicitly on tools that require it (`code_qa`, `search_symbol`, `get_file_summary`, `get_call_chain`).
- For the **current OpenCode project**, pass `repos=["."]` to target the current working directory (project root).
- Use absolute paths when querying repos outside the current project, for example:
  - `repos=["/home/kevin/workspace/my-other-repo"]`
- `~` shortcuts are not guaranteed to expand in all MCP clients. Prefer absolute paths.
- Git URLs are also supported in `repos`, and FastCode can index them when needed.
- For Git worktrees with similar folder names, FastCode now generates a worktree-aware internal index key automatically; keep calling with path-based `repos=["."]` or absolute paths.
- If `get_repo_structure` reports ambiguity across worktrees, copy a concrete name from `list_indexed_repos()` and use that exact value.

Example (current project root):

```typescript
code_qa(
  (question = "Where is auth middleware defined?"),
  (repos = ["."]),
  (multi_turn = true),
);
```

### Session Continuity

Use `multi_turn=true` for iterative exploration. Reuse the returned `session_id` to continue context:

```typescript
code_qa(
  (question = "Find the auth middleware"),
  (repos = ["/path/to/repo"]),
  (multi_turn = true),
);
// Returns [session_id: ...] in output — reuse it on follow-up calls
```

### Symbol Search Types

- `symbol_type: "function"` — functions/methods
- `symbol_type: "class"` — classes/structs
- `symbol_type: "file"` — file-level symbols
- `symbol_type: "documentation"` — docs/comments

### Call Chain Directions

- `direction: "callers"` — who calls this symbol
- `direction: "callees"` — what this symbol calls
- `direction: "both"` (default) — bidirectional trace

## Index Management

```typescript
// Check indexed repositories
list_indexed_repos();

// Re-index a local path or URL
reindex_repo((repo_source = "/path/to/repo"));

// Delete index artifacts only (keeps source files)
delete_repo_metadata((repo_name = "myproject"));
```

## OpenCode Remote Setup

Use this MCP config in your OpenCode setup:

```yaml
mcp:
  fastcode:
    type: http
    url: http://localhost:5555/mcp
```

## Tips

- FastCode auto-clones repos from Git URLs when needed
- Multi-turn sessions preserve context across related questions
- Use `search_symbol` and `get_file_summary` before deep `code_qa` questions for faster grounding
- If you see `BertModel LOAD REPORT ... UNEXPECTED embeddings.position_ids`: this comes from the sentence-transformers embedding stack, not your Ollama generation model. It is typically benign and does not indicate query failure by itself.
