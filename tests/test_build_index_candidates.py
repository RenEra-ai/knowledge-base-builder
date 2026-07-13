"""Tests for candidate builds (C0/B5/B56/B567): embedding inputs, S5 chunk
validation, schema-v2 manifests, semantic-input hashes, and the CLI gates.

Pure helpers are tested directly; heavy paths (SentenceTransformer/Chroma)
stay behind main()'s lazy imports and are exercised only up to the gates.
"""
import argparse
import json
import os
import subprocess
import sys

import pytest

from build_index import (
    EMBEDDING_TEXT_VERSIONS,
    assert_embedding_inputs_within_budget,
    b56_embedding_input,
    build_manifest,
    build_manifest_v2,
    chunk_metadata,
    compute_embedding_plan,
    embedding_input_for,
    validate_chunks,
    write_semantic_inputs,
)
from kb_tokenizer import FakeWordPieceTokenizer
from s7_cache import S7Cache, generate_missing, generator_contract
from s7_eligibility import chunk_identifiers

BUILD_INDEX = os.path.join(os.path.dirname(os.path.dirname(__file__)), "build_index.py")
_TOK = FakeWordPieceTokenizer()

ELIGIBLE_CONTENT = (
    "Document property\n```xml\n<FunctionStep type=\"DocumentPropertySet\"/>\n```"
)


def _s5_chunk(**overrides):
    chunk = {
        "id": "companion_mf_abc_000",
        "content": ELIGIBLE_CONTENT,
        "title": "Map Component Functions",
        "section_heading": "Document property",
        "breadcrumb": "Companion Reference > Components > Map Component Functions",
        "source_url": "https://github.com/OfficialBoomi/x",
        "page_key": "companion://OfficialBoomi/boomi-integration/references/"
                    "components/map_component_functions.md",
        "chunk_index": 0,
        "category": "Companion Reference",
        "token_estimate": 20,
        "source_type": "companion_reference",
        "verification_status": "companion_unverified",
        "upstream_repo": "OfficialBoomi/boomi-integration",
        "upstream_commit": "19aacdd0aa4c9c83f5d87e9b89fb213044447f52",
        "source_path": "references/components/map_component_functions.md",
        "raw_url": "https://raw.githubusercontent.com/OfficialBoomi/x",
        "latest_url": "https://github.com/OfficialBoomi/x",
        # S5 fields
        "source_token_count": 18,
        "content_token_count": 20,
        "embedding_token_count": 30,
        "source_record_id": "companion://OfficialBoomi/boomi-integration/"
                            "references/components/map_component_functions.md",
        "source_start_byte": 40,
        "source_end_byte": 160,
        "synthetic_wrapper_metadata": {"heading_line": True, "fence_open": None,
                                       "fence_close": None, "stripped_xml": False,
                                       "spans": [[40, 160]]},
        "s6_header": "Companion Reference > Components > Map Component Functions",
    }
    chunk.update(overrides)
    return chunk


def _official_s5_chunk(**overrides):
    return _s5_chunk(
        id="db_page_000",
        content="Connection settings\nHost and port live on the connection.",
        title="Database Connector",
        section_heading="Connection settings",
        breadcrumb="Integration > Connectors",
        page_key="https://help.boomi.com/docs/connectors/database",
        category="Integration",
        source_type="official",
        verification_status="official",
        upstream_repo="", upstream_commit="", source_path="",
        raw_url="", latest_url="",
        source_record_id="official:https://help.boomi.com/docs/connectors/database",
        s6_header="Database Connector",
        **overrides,
    )


def _contract(**overrides):
    kwargs = {"prompt_version": "v1", "system_prompt": "sys",
              "user_prompt_template": "u {raw_document}"}
    kwargs.update(overrides)
    return generator_contract(**kwargs)


def _s7(tmp_path, llm=None):
    cache = S7Cache(str(tmp_path / "cache.jsonl"))
    contract = _contract()
    if llm is not None:
        generate_missing(
            cache, [_s5_chunk()], contract, llm,
            validator=lambda d, c, i: [],
            identifiers_fn=chunk_identifiers,
        )
    return {"cache": cache, "contract": contract,
            "identifiers_fn": chunk_identifiers}


def _ok_llm(system, user, model_id):
    return model_id, "A Set Document Property function uses FunctionStep."


def _failing_llm(system, user, model_id):
    return "claude-other", "wrong model output"


# --- embedding_input_for matrix ------------------------------------------------

def test_c0_and_b5_embed_the_raw_document_only():
    chunk = _s5_chunk()
    for candidate in ("c0", "b5"):
        text, source = embedding_input_for(chunk, candidate, tokenizer=_TOK)
        assert text == chunk["content"]
        assert source == "raw"


def test_b56_prepends_the_deduplicated_header():
    chunk = _s5_chunk()
    text, source = embedding_input_for(chunk, "b56", tokenizer=_TOK)
    assert text == chunk["s6_header"] + "\n" + chunk["content"]
    assert source == "raw"
    # A fully-deduplicated (empty) header embeds the raw document alone.
    bare = _s5_chunk(s6_header="")
    assert b56_embedding_input(bare) == bare["content"]


def test_b567_official_always_embeds_the_b56_input(tmp_path):
    chunk = _official_s5_chunk()
    text, source = embedding_input_for(chunk, "b567", tokenizer=_TOK,
                                       s7=_s7(tmp_path))
    assert text == b56_embedding_input(chunk)
    assert source == "raw"


def test_b567_non_eligible_companion_embeds_the_b56_input(tmp_path):
    chunk = _s5_chunk(content="Plain prose only, nothing structural at all.")
    text, source = embedding_input_for(chunk, "b567", tokenizer=_TOK,
                                       s7=_s7(tmp_path))
    assert text == b56_embedding_input(chunk)
    assert source == "raw"


def test_b567_eligible_ok_embeds_the_validated_description_only(tmp_path):
    chunk = _s5_chunk()
    text, source = embedding_input_for(chunk, "b567", tokenizer=_TOK,
                                       s7=_s7(tmp_path, llm=_ok_llm))
    assert text == "A Set Document Property function uses FunctionStep."
    assert source == "description"


def test_b567_raw_fallback_embeds_the_exact_b56_input(tmp_path):
    """Attribution rule: a B567-vs-B56 regression must be attributable to
    successful description vectors — fallback chunks embed the EXACT B56
    input, byte for byte."""
    chunk = _s5_chunk()
    text, source = embedding_input_for(chunk, "b567", tokenizer=_TOK,
                                       s7=_s7(tmp_path, llm=_failing_llm))
    assert text == b56_embedding_input(chunk)
    assert source == "raw_fallback"


def test_b567_missing_cache_entry_for_eligible_chunk_raises(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        embedding_input_for(_s5_chunk(), "b567", tokenizer=_TOK,
                            s7=_s7(tmp_path))


# --- budget guard ------------------------------------------------------------------

def test_budget_guard_names_the_offending_chunk():
    plan = [
        {"id": "ok", "embedding_input": "short text"},
        {"id": "too_big", "embedding_input": " ".join(["word"] * 300)},
    ]
    with pytest.raises(ValueError, match="too_big"):
        assert_embedding_inputs_within_budget(plan, _TOK)


# --- metadata ----------------------------------------------------------------------

def test_legacy_chunk_metadata_shape_is_unchanged():
    legacy = {k: v for k, v in _s5_chunk().items()
              if k not in ("source_token_count", "content_token_count",
                           "embedding_token_count", "source_record_id",
                           "source_start_byte", "source_end_byte",
                           "synthetic_wrapper_metadata", "s6_header")}
    meta = chunk_metadata(legacy)
    assert len(meta) == 15
    assert "source_record_id" not in meta


def test_s5_chunk_metadata_adds_fields_and_json_encodes_wrapper():
    meta = chunk_metadata(_s5_chunk())
    assert meta["source_record_id"].startswith("companion://")
    assert meta["source_start_byte"] == 40
    assert meta["source_end_byte"] == 160
    assert meta["source_token_count"] == 18
    assert meta["embedding_token_count"] == 30
    assert meta["s6_header"].startswith("Companion Reference")
    decoded = json.loads(meta["synthetic_wrapper_metadata"])
    assert decoded["stripped_xml"] is False
    assert decoded["spans"] == [[40, 160]]


def test_compute_embedding_plan_carries_source_description_and_contract(tmp_path):
    s7 = _s7(tmp_path, llm=_ok_llm)
    plan = compute_embedding_plan([_s5_chunk(), _official_s5_chunk()], "b567",
                                  _TOK, s7=s7)
    by_id = {p["id"]: p for p in plan}
    companion = by_id["companion_mf_abc_000"]
    assert companion["embedding_source"] == "description"
    assert companion["document"] == ELIGIBLE_CONTENT  # raw document stored
    assert companion["metadata"]["embedding_source"] == "description"
    assert companion["metadata"]["description"].startswith("A Set Document")
    assert len(companion["metadata"]["generator_contract_sha256"]) == 64
    official = by_id["db_page_000"]
    assert official["embedding_source"] == "raw"
    assert "description" not in official["metadata"]


# --- validate_chunks: conditional S5 rules ------------------------------------------

def test_valid_s5_chunk_passes_validation():
    assert validate_chunks([_s5_chunk()]) == []


@pytest.mark.parametrize("field", [
    "source_token_count", "content_token_count", "embedding_token_count",
    "source_start_byte", "source_end_byte", "synthetic_wrapper_metadata",
    "s6_header",
])
def test_s5_chunk_missing_field_fails_validation(field):
    chunk = _s5_chunk()
    del chunk[field]
    errors = validate_chunks([chunk])
    assert errors and field in errors[0]


@pytest.mark.parametrize("field,value", [
    ("source_start_byte", -1),
    ("source_start_byte", "40"),
    ("source_end_byte", 10),          # end < start
    ("embedding_token_count", True),
    ("synthetic_wrapper_metadata", "not a dict"),
    ("s6_header", None),
])
def test_s5_chunk_bad_field_value_fails_validation(field, value):
    errors = validate_chunks([_s5_chunk(**{field: value})])
    assert errors, f"{field}={value!r} should fail"


# --- schema-v2 manifest --------------------------------------------------------------

def _args(**overrides):
    values = {"model": "all-MiniLM-L6-v2", "artifact_tag": "kb-25-rc1",
              "builder_commit": "abc123"}
    values.update(overrides)
    return argparse.Namespace(**values)


def test_manifest_v2_carries_the_embedding_contract_per_candidate():
    chunks = [_s5_chunk(), _official_s5_chunk()]
    for candidate, text_version, s7_enabled in (
        ("b56", "s5-s6-v1", False), ("b567", "s5-s6-s7-v1", True),
    ):
        manifest = build_manifest_v2(
            chunks, _args(), 384, candidate,
            source_snapshot_sha256="f" * 64,
        )
        assert manifest["schema_version"] == "2"
        assert manifest["source_snapshot_sha256"] == "f" * 64
        contract = manifest["embedding_contract"]
        assert contract == {
            "version": 1,
            "model_id": "all-MiniLM-L6-v2",
            "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            "max_seq_length": 256,
            "distance_metric": "cosine",
            "normalize_embeddings": False,
            "embedding_text_version": text_version,
            "s7_enabled": s7_enabled,
        }
        assert EMBEDDING_TEXT_VERSIONS[candidate] == text_version
        # v1 base fields survive.
        assert manifest["collection_name"] == "boomi_docs"
        assert manifest["embedding_model"] == "all-MiniLM-L6-v2"


def test_manifest_v2_requires_a_source_snapshot_hash():
    with pytest.raises(ValueError, match="source_snapshot_sha256"):
        build_manifest_v2([_s5_chunk()], _args(), 384, "b56",
                          source_snapshot_sha256="")


def test_manifest_v2_b567_records_the_s7_hashes():
    manifest = build_manifest_v2(
        [_s5_chunk()], _args(), 384, "b567",
        source_snapshot_sha256="f" * 64,
        s7_hashes={"generator_contract_sha256": "a" * 64,
                   "s7_cache_sha256": "b" * 64,
                   "usage_map_sha256": "c" * 64},
    )
    assert manifest["generator_contract_sha256"] == "a" * 64
    assert manifest["s7_cache_sha256"] == "b" * 64
    assert manifest["usage_map_sha256"] == "c" * 64


# --- semantic-inputs sidecar ----------------------------------------------------------

def test_write_semantic_inputs_rows_and_stable_summary_hash(tmp_path):
    plan = compute_embedding_plan([_official_s5_chunk(), _s5_chunk(s6_header="H")],
                                  "b56", _TOK)
    out = tmp_path / "semantic_inputs.jsonl"
    combined = {"source_snapshot_sha256": "f" * 64}
    summary = write_semantic_inputs(str(out), plan, combined)

    lines = [json.loads(line) for line in out.read_text().splitlines()]
    rows, trailer = lines[:-1], lines[-1]
    assert len(rows) == 2
    for row in rows:
        assert sorted(row) == ["canonical_metadata", "embedding_input_sha256",
                               "id", "raw_document"]
        assert len(row["embedding_input_sha256"]) == 64
    assert trailer["semantic_inputs_sha256"] == summary["semantic_inputs_sha256"]
    assert trailer["combined_hash_inputs"]["source_snapshot_sha256"] == "f" * 64

    # Stable: identical plan -> identical summary hash.
    plan2 = compute_embedding_plan([_official_s5_chunk(), _s5_chunk(s6_header="H")],
                                   "b56", _TOK)
    summary2 = write_semantic_inputs(str(tmp_path / "s2.jsonl"), plan2, combined)
    assert summary2["semantic_inputs_sha256"] == summary["semantic_inputs_sha256"]


# --- CLI gates --------------------------------------------------------------------------

def _write_chunks(path, chunks):
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")


def test_cli_candidate_requires_s5_chunks(tmp_path):
    legacy = {k: v for k, v in _s5_chunk().items()
              if k not in ("source_token_count", "content_token_count",
                           "embedding_token_count", "source_record_id",
                           "source_start_byte", "source_end_byte",
                           "synthetic_wrapper_metadata", "s6_header")}
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path, [legacy])
    result = subprocess.run(
        [sys.executable, BUILD_INDEX, "--input", str(chunks_path),
         "--output", str(tmp_path / "db"), "--candidate", "b56",
         "--source-snapshot-sha256", "f" * 64],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "S5 chunk" in result.stdout


def test_cli_candidate_runs_fixture_integrity_before_embedding(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path, [_s5_chunk()])
    fixtures = tmp_path / "fixtures.json"
    fixtures.write_text(json.dumps([{
        "query_id": "GX1", "query": "q",
        "groups": [{"page_match": "map_component_functions.md",
                    "markers": ["THIS MARKER DOES NOT EXIST"]}],
    }]), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, BUILD_INDEX, "--input", str(chunks_path),
         "--output", str(tmp_path / "db"), "--candidate", "b56",
         "--source-snapshot-sha256", "f" * 64,
         "--eval-fixtures", str(fixtures)],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "fixture_marker_missing" in result.stdout


def test_cli_b567_requires_cache_and_contract(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path, [_s5_chunk()])
    result = subprocess.run(
        [sys.executable, BUILD_INDEX, "--input", str(chunks_path),
         "--output", str(tmp_path / "db"), "--candidate", "b567",
         "--source-snapshot-sha256", "f" * 64],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "--s7-cache" in result.stdout
