import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FASTCODE_MAIN = ROOT / "fastcode" / "main.py"
MCP_SERVER = ROOT / "mcp_server.py"


def _null_logger():
    return SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
    )


def _load_functions(path, names, *, class_name=None, global_ns=None):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    if class_name:
        class_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        lookup = {
            node.name: node
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
    else:
        lookup = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    selected = []
    for name in names:
        node = copy.deepcopy(lookup[name])
        node.decorator_list = []
        selected.append(node)

    future_import = ast.parse("from __future__ import annotations").body
    module = ast.Module(body=future_import + selected, type_ignores=[])
    ast.fix_missing_locations(module)

    namespace = {}
    if global_ns:
        namespace.update(global_ns)
    exec(compile(module, str(path), "exec"), namespace)
    return [namespace[name] for name in names]


class StubCodeElement:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to_dict(self):
        return dict(self.__dict__)


def _element_meta(element_id, file_path, relative_path, embedding):
    return {
        "id": element_id,
        "type": "file",
        "name": relative_path,
        "file_path": file_path,
        "relative_path": relative_path,
        "language": "python",
        "start_line": 1,
        "end_line": 10,
        "code": "print('hello')",
        "signature": None,
        "docstring": None,
        "summary": None,
        "metadata": {"embedding": embedding},
        "repo_name": "repo",
        "repo_url": None,
    }


def _make_incremental_reindex(globals_override=None):
    base_globals = {
        "os": os,
        "np": np,
        "CodeElement": StubCodeElement,
    }
    if globals_override:
        base_globals.update(globals_override)
    return _load_functions(
        FASTCODE_MAIN,
        ["incremental_reindex"],
        class_name="FastCode",
        global_ns=base_globals,
    )[0]


def test_incremental_reindex_uses_loader_repo_path_when_rebuilding_graphs(tmp_path):
    captured = {}

    class FakeTempStore:
        def __init__(self, config):
            self.config = config

        def initialize(self, dimension):
            self.dimension = dimension

        def add_vectors(self, vectors, metadata_list):
            self.vectors = vectors
            self.metadata_list = metadata_list

    class FakeRetriever:
        def __init__(self, config, vector_store, embedder, graph_builder, repo_root=None):
            self.repo_root = repo_root

        def index_for_bm25(self, elements):
            self.elements = elements

    class FakeGraphBuilder:
        def __init__(self, config):
            self.config = config

        def build_graphs(self, elements, module_resolver, symbol_resolver):
            self.elements = elements

    class FakeGlobalIndexBuilder:
        def __init__(self, config):
            self.config = config

        def build_maps(self, elements, repo_root):
            captured["repo_root"] = repo_root

    incremental_reindex = _make_incremental_reindex(
        {
            "VectorStore": FakeTempStore,
            "HybridRetriever": FakeRetriever,
            "CodeGraphBuilder": FakeGraphBuilder,
            "GlobalIndexBuilder": FakeGlobalIndexBuilder,
            "ModuleResolver": lambda gib: ("module_resolver", gib),
            "SymbolResolver": lambda gib, module_resolver: (
                "symbol_resolver",
                gib,
                module_resolver,
            ),
        }
    )

    original_repo = tmp_path / "source-repo"
    copied_repo = tmp_path / "workspace-copy" / "repo"
    original_repo.mkdir()
    copied_repo.mkdir(parents=True)

    file_path = copied_repo / "a.py"
    meta = _element_meta("elem-1", str(file_path), "a.py", [0.1, 0.2, 0.3])

    class FakeLoader:
        def __init__(self):
            self.repo_path = None

        def load_from_path(self, path):
            self.repo_path = str(copied_repo)

        def scan_files(self):
            return []

    fc = SimpleNamespace(
        logger=_null_logger(),
        loader=FakeLoader(),
        config={},
        embedder=SimpleNamespace(embedding_dim=3),
        loaded_repositories={},
        indexer=SimpleNamespace(index_files=lambda file_infos, repo_name, repo_url=None: []),
        _load_file_manifest=lambda repo_name: {"files": {"a.py": {"element_ids": ["elem-1"]}}},
        _detect_file_changes=lambda repo_name, current_files: {
            "added": [],
            "modified": [],
            "deleted": ["deleted.py"],
            "unchanged": ["a.py"],
            "manifest": {"files": {"a.py": {"element_ids": ["elem-1"]}}},
            "current_lookup": {},
        },
        _load_existing_metadata=lambda repo_name: [meta],
        _collect_unchanged_elements=lambda manifest, unchanged_files, existing_metadata: (
            existing_metadata,
            ["elem-1"],
        ),
        _should_persist_indexes=lambda: False,
    )

    incremental_reindex(fc, "repo", repo_path=str(original_repo))

    assert captured["repo_root"] == str(copied_repo)


def test_incremental_reindex_regenerates_repository_overview_after_changes(tmp_path):
    class FakeTempStore:
        def __init__(self, config):
            self.config = config

        def initialize(self, dimension):
            self.dimension = dimension

        def add_vectors(self, vectors, metadata_list):
            self.vectors = vectors
            self.metadata_list = metadata_list

        def save(self, repo_name):
            self.saved_repo = repo_name

    class FakeRetriever:
        def __init__(self, config, vector_store, embedder, graph_builder, repo_root=None):
            self.repo_root = repo_root

        def index_for_bm25(self, elements):
            self.elements = elements

        def save_bm25(self, repo_name):
            self.saved_repo = repo_name

    class FakeGraphBuilder:
        def __init__(self, config):
            self.config = config

        def build_graphs(self, elements, module_resolver, symbol_resolver):
            self.elements = elements

        def save(self, repo_name):
            self.saved_repo = repo_name

    class FakeGlobalIndexBuilder:
        def __init__(self, config):
            self.config = config

        def build_maps(self, elements, repo_root):
            self.elements = elements
            self.repo_root = repo_root

    incremental_reindex = _make_incremental_reindex(
        {
            "VectorStore": FakeTempStore,
            "HybridRetriever": FakeRetriever,
            "CodeGraphBuilder": FakeGraphBuilder,
            "GlobalIndexBuilder": FakeGlobalIndexBuilder,
            "ModuleResolver": lambda gib: ("module_resolver", gib),
            "SymbolResolver": lambda gib, module_resolver: (
                "symbol_resolver",
                gib,
                module_resolver,
            ),
        }
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    changed_file = repo_root / "a.py"

    current_file_info = {
        "path": str(changed_file),
        "relative_path": "a.py",
        "size": 10,
        "extension": ".py",
    }
    existing_meta = _element_meta(
        "old-elem",
        str(repo_root / "old.py"),
        "old.py",
        [0.1, 0.2, 0.3],
    )
    new_element = StubCodeElement(
        id="new-elem",
        type="file",
        name="a.py",
        file_path=str(changed_file),
        relative_path="a.py",
        language="python",
        start_line=1,
        end_line=20,
        code="print('updated')",
        signature=None,
        docstring=None,
        summary=None,
        metadata={"embedding": [0.4, 0.5, 0.6]},
        repo_name="repo",
        repo_url="https://example.com/repo.git",
    )

    class FakeLoader:
        def __init__(self):
            self.repo_path = None

        def load_from_path(self, path):
            self.repo_path = str(repo_root)

        def scan_files(self):
            return [current_file_info]

    vector_store = SimpleNamespace(
        persist_dir=str(tmp_path / "persist"),
        save_repo_overview=Mock(),
    )

    class FakeIndexer:
        def __init__(self):
            self.overview_generator = SimpleNamespace(
                parse_file_structure=lambda repo_path, files: {"languages": {"python": len(files)}},
                generate_overview=lambda repo_path, repo_name, file_structure: {
                    "repo_name": repo_name,
                    "summary": "updated summary",
                    "structure_text": "a.py",
                    "file_structure": file_structure,
                    "readme_content": "",
                    "has_readme": False,
                },
            )

        def index_files(self, file_infos, repo_name, repo_url=None):
            return [new_element]

        def _save_repository_overview(self, overview):
            vector_store.save_repo_overview(
                overview["repo_name"],
                overview["summary"],
                np.array([0.1, 0.2, 0.3], dtype=np.float32),
                {
                    "summary": overview["summary"],
                    "structure_text": overview["structure_text"],
                    "file_structure": overview["file_structure"],
                },
            )

    fc = SimpleNamespace(
        logger=_null_logger(),
        loader=FakeLoader(),
        vector_store=vector_store,
        indexer=FakeIndexer(),
        config={},
        embedder=SimpleNamespace(embedding_dim=3),
        loaded_repositories={"repo": {"url": "https://example.com/repo.git"}},
        _load_file_manifest=lambda repo_name: {"files": {"old.py": {"element_ids": ["old-elem"]}}},
        _detect_file_changes=lambda repo_name, current_files: {
            "added": ["a.py"],
            "modified": [],
            "deleted": [],
            "unchanged": [],
            "manifest": {"files": {"old.py": {"element_ids": ["old-elem"]}}},
            "current_lookup": {"a.py": {"file_info": current_file_info}},
        },
        _load_existing_metadata=lambda repo_name: [existing_meta],
        _collect_unchanged_elements=lambda manifest, unchanged_files, existing_metadata: ([], []),
        _should_persist_indexes=lambda: True,
        _build_file_manifest=lambda elements, repo_root: {"files": {}},
        _save_file_manifest=lambda repo_name, manifest: None,
    )

    incremental_reindex(fc, "repo", repo_path=str(repo_root))

    assert vector_store.save_repo_overview.called


def test_incremental_reindex_rejects_incompatible_preserved_embeddings(tmp_path):
    class GuardedTempStore:
        def __init__(self, config):
            self.config = config
            self.dimension = None

        def initialize(self, dimension):
            self.dimension = dimension

        def add_vectors(self, vectors, metadata_list):
            if vectors.shape[1] != self.dimension:
                raise AssertionError(
                    "incremental_reindex attempted to rebuild with incompatible "
                    "preserved embeddings"
                )

    class FakeRetriever:
        def __init__(self, config, vector_store, embedder, graph_builder, repo_root=None):
            self.repo_root = repo_root

        def index_for_bm25(self, elements):
            self.elements = elements

    class FakeGraphBuilder:
        def __init__(self, config):
            self.config = config

        def build_graphs(self, elements, module_resolver, symbol_resolver):
            self.elements = elements

    class FakeGlobalIndexBuilder:
        def __init__(self, config):
            self.config = config

        def build_maps(self, elements, repo_root):
            self.elements = elements

    incremental_reindex = _make_incremental_reindex(
        {
            "VectorStore": GuardedTempStore,
            "HybridRetriever": FakeRetriever,
            "CodeGraphBuilder": FakeGraphBuilder,
            "GlobalIndexBuilder": FakeGlobalIndexBuilder,
            "ModuleResolver": lambda gib: ("module_resolver", gib),
            "SymbolResolver": lambda gib, module_resolver: (
                "symbol_resolver",
                gib,
                module_resolver,
            ),
        }
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    existing_meta = _element_meta(
        "elem-1",
        str(repo_root / "a.py"),
        "a.py",
        [0.1, 0.2],
    )

    class FakeLoader:
        def __init__(self):
            self.repo_path = None

        def load_from_path(self, path):
            self.repo_path = str(repo_root)

        def scan_files(self):
            return []

    fc = SimpleNamespace(
        logger=_null_logger(),
        loader=FakeLoader(),
        config={},
        embedder=SimpleNamespace(embedding_dim=3),
        loaded_repositories={},
        indexer=SimpleNamespace(index_files=lambda file_infos, repo_name, repo_url=None: []),
        _load_file_manifest=lambda repo_name: {"files": {"a.py": {"element_ids": ["elem-1"]}}},
        _detect_file_changes=lambda repo_name, current_files: {
            "added": [],
            "modified": [],
            "deleted": ["deleted.py"],
            "unchanged": ["a.py"],
            "manifest": {"files": {"a.py": {"element_ids": ["elem-1"]}}},
            "current_lookup": {},
        },
        _load_existing_metadata=lambda repo_name: [existing_meta],
        _collect_unchanged_elements=lambda manifest, unchanged_files, existing_metadata: (
            existing_metadata,
            ["elem-1"],
        ),
        _should_persist_indexes=lambda: False,
    )

    incremental_reindex(fc, "repo", repo_path=str(repo_root))


def test_ensure_repos_ready_falls_back_to_full_reindex_when_manifest_is_missing(tmp_path):
    ensure_repos_ready = _load_functions(
        MCP_SERVER,
        ["_ensure_repos_ready"],
        global_ns={"os": os, "logger": _null_logger()},
    )[0]

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    fc = SimpleNamespace(
        _infer_is_url=lambda source: False,
        incremental_reindex=Mock(return_value={"status": "no_manifest", "changes": 0}),
        load_repository=Mock(),
        index_repository=Mock(),
        repo_indexed=True,
        loaded_repositories={},
    )
    full_reindex = Mock(return_value={"status": "success", "count": 12})

    ensure_repos_ready.__globals__.update(
        {
            "_get_fastcode": lambda: fc,
            "_apply_forced_env_excludes": lambda fc: None,
            "_repo_name_from_source": lambda source, is_url: "repo",
            "_is_repo_indexed": lambda repo_name: True,
            "_run_full_reindex": full_reindex,
            "_invalidate_loaded_state": lambda fc: None,
        }
    )

    ensure_repos_ready([str(repo_dir)])

    assert full_reindex.called
    assert not fc.load_repository.called
    assert not fc.index_repository.called


def test_lookup_tools_do_not_disable_incremental_refresh():
    search_symbol, get_file_summary, get_call_chain = _load_functions(
        MCP_SERVER,
        ["search_symbol", "get_file_summary", "get_call_chain"],
        global_ns={},
    )

    calls = []

    def fake_ensure_repos_ready(repos, allow_incremental=True, ctx=None):
        calls.append(allow_incremental)
        return ["repo"]

    fake_fc = SimpleNamespace(
        vector_store=SimpleNamespace(metadata=[]),
        graph_builder=SimpleNamespace(
            element_by_name={},
            element_by_id={},
            get_callers=lambda element_id: [],
            get_callees=lambda element_id: [],
        ),
    )

    shared_globals = {
        "_get_fastcode": lambda: fake_fc,
        "_ensure_repos_ready": fake_ensure_repos_ready,
        "_ensure_loaded": lambda fc, ready_names: True,
    }

    search_symbol.__globals__.update(shared_globals)
    get_file_summary.__globals__.update(shared_globals)
    get_call_chain.__globals__.update(shared_globals)

    search_symbol("FastCode", ["/tmp/repo"])
    get_file_summary("fastcode/main.py", ["/tmp/repo"])
    get_call_chain("query", ["/tmp/repo"])

    assert calls == [True, True, True]


def test_reindex_repo_uses_clean_full_reindex_helper():
    reindex_repo = _load_functions(
        MCP_SERVER,
        ["reindex_repo"],
        global_ns={"os": os, "logger": _null_logger()},
    )[0]

    fc = SimpleNamespace(_infer_is_url=lambda source: False)
    full_reindex = Mock(return_value={"status": "success", "count": 42})

    reindex_repo.__globals__.update(
        {
            "_get_fastcode": lambda: fc,
            "_repo_name_from_source": lambda source, is_url: "repo",
            "_run_full_reindex": full_reindex,
        }
    )

    message = reindex_repo("/tmp/repo")

    assert "42 elements indexed" in message
    assert full_reindex.called


def test_load_multi_repo_cache_replaces_loaded_repository_set(tmp_path):
    load_multi_repo_cache = _load_functions(
        FASTCODE_MAIN,
        ["_load_multi_repo_cache"],
        class_name="FastCode",
        global_ns={"os": os, "pickle": __import__("pickle")},
    )[0]

    persist_dir = tmp_path / "persist"
    persist_dir.mkdir()
    for repo_name in ("repo_a", "repo_b"):
        (persist_dir / f"{repo_name}.faiss").write_bytes(b"index")
        (persist_dir / f"{repo_name}_metadata.pkl").write_bytes(b"meta")

    class FakeVectorStore:
        def __init__(self):
            self.persist_dir = str(persist_dir)
            self.merged = []

        def initialize(self, dimension):
            self.dimension = dimension

        def merge_from_index(self, repo_name):
            self.merged.append(repo_name)
            return True

        def get_count(self):
            return len(self.merged)

    fake_fc = SimpleNamespace(
        logger=_null_logger(),
        vector_store=FakeVectorStore(),
        embedder=SimpleNamespace(embedding_dim=3),
        loaded_repositories={"repo_a": {"name": "repo_a"}, "repo_b": {"name": "repo_b"}},
        retriever=SimpleNamespace(
            persist_dir=str(persist_dir),
            build_repo_overview_bm25=lambda: None,
            index_for_bm25=lambda elements: None,
        ),
        graph_builder=SimpleNamespace(load=lambda repo_name: False, merge_from_file=lambda repo_name: False),
        _reconstruct_elements_from_metadata=lambda: [],
    )

    ok = load_multi_repo_cache(fake_fc, repo_names=["repo_a"])

    assert ok is True
    assert set(fake_fc.loaded_repositories) == {"repo_a"}
