"""Tests for build_index.py: manifest assembly, verify gate, input gates.

These exercise the pure helpers directly and the input-validation gates via
subprocess. They do not need chromadb / sentence-transformers installed: the
heavy imports in build_index.py are lazy (inside main(), after the gates).
"""

import argparse
import json
import os
import subprocess
import sys

from build_index import VERIFY_QUERIES, build_manifest, verify_index

BUILD_INDEX = os.path.join(os.path.dirname(os.path.dirname(__file__)), "build_index.py")


def _valid_chunk(**overrides):
    chunk = {
        "id": "chunk_000",
        "content": "body text",
        "title": "Page Title",
        "section_heading": "Section",
        "breadcrumb": "Integration > X",
        "source_url": "https://help.boomi.com/docs/x",
        "page_key": "https://help.boomi.com/docs/x",
        "chunk_index": 0,
        "category": "Integration",
        "token_estimate": 10,
    }
    chunk.update(overrides)
    return chunk


# --- build_manifest -----------------------------------------------------------

def _args(model="all-MiniLM-L6-v2", artifact_tag="kb-42", builder_commit="abc1234"):
    return argparse.Namespace(
        model=model, artifact_tag=artifact_tag, builder_commit=builder_commit
    )


def test_build_manifest_has_all_schema_v1_keys():
    chunks = [_valid_chunk(id="a_000"), _valid_chunk(id="a_001")]
    manifest = build_manifest(chunks, _args(), embedding_dim=384)
    assert set(manifest) == {
        "schema_version", "collection_name", "embedding_model", "embedding_dim",
        "build_timestamp", "chunk_count", "page_count", "category_counts",
        "source_roots", "artifact_tag", "builder_commit", "builder_version",
    }
    assert manifest["schema_version"] == "1"
    assert manifest["collection_name"] == "boomi_docs"
    assert manifest["embedding_model"] == "all-MiniLM-L6-v2"
    assert manifest["embedding_dim"] == 384


def test_build_manifest_counts_chunks_pages_categories():
    chunks = [
        _valid_chunk(id="a_000", page_key="P1", category="Integration"),
        _valid_chunk(id="a_001", page_key="P1", category="Integration"),
        _valid_chunk(id="b_000", page_key="P2", category="EDI"),
    ]
    manifest = build_manifest(chunks, _args(), embedding_dim=384)
    assert manifest["chunk_count"] == 3
    assert manifest["page_count"] == 2
    assert manifest["category_counts"] == {"Integration": 2, "EDI": 1}


def test_build_manifest_derives_source_roots():
    chunks = [
        _valid_chunk(id="a_000", source_url="https://help.boomi.com/docs/x"),
        _valid_chunk(id="b_000", source_url="https://developer.boomi.com/docs/y"),
        _valid_chunk(id="c_000", source_url="https://help.boomi.com/docs/z"),
    ]
    manifest = build_manifest(chunks, _args(), embedding_dim=384)
    assert manifest["source_roots"] == [
        "https://developer.boomi.com",
        "https://help.boomi.com",
    ]


def test_build_manifest_source_roots_fallback_when_all_empty():
    chunks = [_valid_chunk(id="a_000", source_url=""), _valid_chunk(id="b_000", source_url="")]
    manifest = build_manifest(chunks, _args(), embedding_dim=384)
    assert manifest["source_roots"] == ["https://help.boomi.com"]


def test_build_manifest_echoes_artifact_tag_and_commit():
    chunks = [_valid_chunk()]
    manifest = build_manifest(
        chunks, _args(artifact_tag="kb-99", builder_commit="deadbee"), embedding_dim=384
    )
    assert manifest["artifact_tag"] == "kb-99"
    assert manifest["builder_commit"] == "deadbee"
    assert manifest["builder_version"]  # non-empty placeholder


# --- verify_index gate --------------------------------------------------------

class FakeCollection:
    """Minimal stand-in for a Chroma collection in verify_index tests."""

    def __init__(self, distances_by_query):
        self._distances = distances_by_query

    def query(self, query_texts, n_results):
        distances = self._distances.get(query_texts[0], [])
        ids = [f"id-{i}" for i in range(len(distances))]
        metas = [{"title": "T", "section_heading": "S"} for _ in distances]
        return {"ids": [ids], "distances": [distances], "metadatas": [metas]}


def test_verify_index_passes_when_all_queries_have_close_hit():
    collection = FakeCollection({q: [0.20, 0.30, 0.40] for q in VERIFY_QUERIES})
    assert verify_index(collection) is True


def test_verify_index_fails_when_a_query_is_too_far():
    distances = {q: [0.20, 0.30] for q in VERIFY_QUERIES}
    distances[VERIFY_QUERIES[0]] = [0.80, 0.90]  # best hit above 0.45 threshold
    collection = FakeCollection(distances)
    assert verify_index(collection) is False


def test_verify_index_fails_when_a_query_has_no_results():
    distances = {q: [0.20] for q in VERIFY_QUERIES}
    distances[VERIFY_QUERIES[-1]] = []  # no hits at all
    collection = FakeCollection(distances)
    assert verify_index(collection) is False


# --- input-validation gates (subprocess) --------------------------------------

def _run_build_index(input_path, tmp_path):
    return subprocess.run(
        [sys.executable, BUILD_INDEX, "--input", str(input_path),
         "--output", str(tmp_path / "out")],
        capture_output=True, text=True,
    )


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )


def test_main_fails_on_missing_input_file(tmp_path):
    result = _run_build_index(tmp_path / "does_not_exist.jsonl", tmp_path)
    assert result.returncode == 1
    assert "Input file not found" in result.stdout


def test_main_fails_on_empty_input(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    result = _run_build_index(empty, tmp_path)
    assert result.returncode == 1
    assert "No chunks found" in result.stdout


def test_main_fails_on_missing_required_field(tmp_path):
    bad = _valid_chunk()
    del bad["page_key"]
    jsonl = tmp_path / "bad.jsonl"
    _write_jsonl(jsonl, [bad])
    result = _run_build_index(jsonl, tmp_path)
    assert result.returncode == 1
    assert "missing required field" in result.stdout
    assert "page_key" in result.stdout


def test_main_fails_on_duplicate_chunk_ids(tmp_path):
    jsonl = tmp_path / "dupes.jsonl"
    _write_jsonl(jsonl, [
        _valid_chunk(id="chunk_000"),
        _valid_chunk(id="chunk_000"),  # duplicate id
    ])
    result = _run_build_index(jsonl, tmp_path)
    assert result.returncode == 1
    assert "duplicate chunk id" in result.stdout
