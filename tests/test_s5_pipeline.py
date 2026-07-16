"""Tests for the S5 tokenizer-aware document pipeline (chunk records)."""
import json
import os
import subprocess
import sys

import pytest

from kb_tokenizer import EFFECTIVE_BUDGET, FakeWordPieceTokenizer
from s5_pipeline import (
    S5_CHUNK_FIELDS,
    build_official_chunks,
    build_s5_chunks,
    chunk_companion_text,
)

_TOK = FakeWordPieceTokenizer()

_ENTRY = {
    "path": "references/components/map_component_functions.md",
    "area": "Components",
    "title": "Map Component Functions",
    "sections": None,
    "drop_sections": None,
    "blob_url": "https://github.com/OfficialBoomi/boomi-integration/blob/19aacdd/x.md",
    "raw_url": "https://raw.githubusercontent.com/OfficialBoomi/x.md",
    "latest_url": "https://github.com/OfficialBoomi/boomi-integration/blob/main/x.md",
}
_UPSTREAM = {
    "repo": "OfficialBoomi/boomi-integration",
    "commit": "19aacdd0aa4c9c83f5d87e9b89fb213044447f52",
}


def _companion(text, entry_overrides=None):
    entry = dict(_ENTRY)
    entry.update(entry_overrides or {})
    return chunk_companion_text(entry, _UPSTREAM, text, _TOK)


SMALL_MD = """# Map Component Functions

## Function architecture

Functions run in order. The order matters for chained outputs.

```xml
<Functions optimizeExecutionOrder="true"/>
```
"""


def test_content_is_raw_markdown_not_html():
    chunks = _companion(SMALL_MD)
    arch = [c for c in chunks if c["section_heading"] == "Function architecture"]
    assert arch
    for chunk in arch:
        assert "<p>" not in chunk["content"]
        assert "```xml" in chunk["content"]
        # HTML is rendered ONLY after final fragments exist, from the fragment.
        assert "<code>" in chunk["content_html"] or "<pre>" in chunk["content_html"]


def test_heading_is_first_raw_content_line_of_every_fragment():
    body = " ".join(f"word{i:04d}" for i in range(200))  # 400 fake pieces -> splits
    md = f"# Map Component Functions\n\n## Function architecture\n\n{body}\n"
    chunks = _companion(md)
    arch = [c for c in chunks if c["section_heading"] == "Function architecture"]
    assert len(arch) >= 2, "expected the oversized section to split"
    for chunk in arch:
        assert chunk["content"].split("\n", 1)[0] == "Function architecture"
        assert chunk["content"].count("Function architecture") == 1


def test_every_embedding_input_fits_the_wordpiece_budget():
    body = " ".join(f"identifier{i:05d}" for i in range(300))
    md = f"# Map Component Functions\n\n## Key observations\n\n{body}\n"
    for chunk in _companion(md):
        assert chunk["embedding_token_count"] <= EFFECTIVE_BUDGET
        # Recompute: header + raw document is what gets embedded for B56.
        embedding_input = (
            chunk["s6_header"] + "\n" + chunk["content"]
            if chunk["s6_header"] else chunk["content"]
        )
        assert _TOK.count(embedding_input) == chunk["embedding_token_count"]


def test_spans_tile_the_section_body_across_chunks():
    body = " ".join(f"word{i:04d}" for i in range(200))
    md = f"# Map Component Functions\n\n## Function architecture\n\n{body}\n"
    record_bytes = md.encode("utf-8")
    chunks = [c for c in _companion(md)
              if c["section_heading"] == "Function architecture"]
    assert len(chunks) >= 2
    # Adjacent chunks tile: each starts where the previous ended, and the byte
    # range decodes to the section body's exact bytes.
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev["source_end_byte"] == nxt["source_start_byte"]
    joined = record_bytes[
        chunks[0]["source_start_byte"]:chunks[-1]["source_end_byte"]
    ].decode("utf-8")
    assert joined == body


def test_stripped_xml_section_records_original_span_and_flag():
    big_xml = "<Component>" + ("<x/>" * 500) + "</Component>"  # >1500 chars
    md = (
        "# Map Component Functions\n\n## Complete example\n\nIntro text.\n\n"
        f"```xml\n{big_xml}\n```\n"
    )
    chunks = _companion(md, {"strip_xml_blocks": True})
    example = [c for c in chunks if c["section_heading"] == "Complete example"]
    assert example
    for chunk in example:
        meta = chunk["synthetic_wrapper_metadata"]
        assert meta["stripped_xml"] is True
        assert "<Component>" not in chunk["content"]
    # Original section body byte range (pre-strip) is recorded.
    body_start = md.encode("utf-8").index(b"Intro text.")
    assert example[0]["source_start_byte"] == body_start


def test_chunk_records_carry_all_s5_fields_with_correct_types():
    chunks = _companion(SMALL_MD)
    assert chunks
    for chunk in chunks:
        assert S5_CHUNK_FIELDS <= set(chunk)
        assert isinstance(chunk["source_token_count"], int)
        assert isinstance(chunk["content_token_count"], int)
        assert isinstance(chunk["embedding_token_count"], int)
        assert chunk["source_record_id"].startswith("companion://")
        assert isinstance(chunk["source_start_byte"], int)
        assert isinstance(chunk["source_end_byte"], int)
        assert chunk["source_start_byte"] <= chunk["source_end_byte"]
        assert isinstance(chunk["synthetic_wrapper_metadata"], dict)
        assert isinstance(chunk["s6_header"], str)
        # Diagnostics-only estimate is retained for backward compatibility.
        assert chunk["token_estimate"] == len(chunk["content"]) // 4
        # Provenance contract matches the legacy pipeline.
        assert chunk["source_type"] == "companion_reference"
        assert chunk["verification_status"] == "companion_unverified"
        assert chunk["upstream_repo"] == _UPSTREAM["repo"]
        assert chunk["upstream_commit"] == _UPSTREAM["commit"]


def test_companion_header_is_breadcrumb_when_title_repeats_in_heading():
    chunks = _companion(SMALL_MD)
    title_chunk = [c for c in chunks
                   if c["section_heading"] == "Map Component Functions"]
    # The h1 section repeats the title as its heading line, so the S6 header
    # dedups the title and keeps only the breadcrumb.
    if title_chunk:
        header = title_chunk[0]["s6_header"]
        assert "Companion Reference > Components" in header
        assert header.count("Map Component Functions") == 1


def test_section_allowlist_and_deny_still_filter():
    md = (
        "# Map Component Functions\n\n## Keep me\n\nkeep body\n\n"
        "## Drop me\n\ndrop body\n"
    )
    chunks = _companion(md, {"sections": ["keep me"]})
    headings = {c["section_heading"] for c in chunks}
    assert "Keep me" in headings
    assert "Drop me" not in headings


def test_chunk_ids_unique_and_page_key_stable():
    chunks = _companion(SMALL_MD)
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(
        c["page_key"] == "companion://OfficialBoomi/boomi-integration/"
                         "references/components/map_component_functions.md"
        for c in chunks
    )


# --- official HTML path ---------------------------------------------------------

_OFFICIAL_HTML = """<h1>Database Connector</h1>
<p><strong>Path:</strong> <a href='https://help.boomi.com/docs/connectors/database'>Database Connector</a></p>
<p>The database connector lets processes communicate with your database.</p>
<h2>Connection settings</h2>
<p>Host, port, and credentials are configured on the connection component.</p>
"""


def test_official_chunks_have_records_spans_and_title_header(tmp_path):
    (tmp_path / "db.html").write_text(_OFFICIAL_HTML, encoding="utf-8")
    chunks = build_official_chunks(str(tmp_path), _TOK)
    assert chunks
    for chunk in chunks:
        assert chunk["source_type"] == "official"
        assert chunk["source_record_id"].startswith("official:")
        assert S5_CHUNK_FIELDS <= set(chunk)
        assert chunk["content"].split("\n", 1)[0] == chunk["section_heading"]
    settings = [c for c in chunks if c["section_heading"] == "Connection settings"]
    assert settings
    assert "configured on the connection component" in settings[0]["content"]
    # Official header = page title (deduped when the heading repeats it).
    assert settings[0]["s6_header"] == "Database Connector"


def test_official_section_spans_decode_from_the_page_record(tmp_path):
    (tmp_path / "db.html").write_text(_OFFICIAL_HTML, encoding="utf-8")
    chunks = build_official_chunks(str(tmp_path), _TOK)
    records = {c["source_record_id"]: c for c in chunks}
    assert len({c["source_record_id"] for c in chunks}) == 1
    # Spans are non-overlapping and increasing in document order.
    ordered = sorted(chunks, key=lambda c: c["source_start_byte"])
    for prev, nxt in zip(ordered, ordered[1:]):
        assert prev["source_end_byte"] <= nxt["source_start_byte"]


# Styled-components pages (developer.boomi.com) carry their stylesheet inline in
# <style> tags between the headings. BeautifulSoup only hides a Stylesheet string
# from get_text() when it is called on an ANCESTOR: called on the <style> tag
# itself — which is what elements_to_text does for every top-level section
# element — it returns the CSS as if it were prose.
_STYLED_HTML = """<h1>File Sharing API</h1>
<p>Upload a file to the shared folder.</p>
<style data-styled="active">html:not([data-theme='dark']) .eqlrrV{width:calc(100% - 40%);padding:0 40px;}/*!sc*/
@media print,screen and (max-width: 130rem){html .fidpWx{width:100%;}}/*!sc*/</style>
<h2>Response codes</h2>
<script>window.__DOCS__ = {build: 1};</script>
<p>A 200 response returns the file metadata.</p>
"""


def test_style_and_script_blocks_never_reach_the_official_corpus(tmp_path):
    """Browser assets are not documentation: a <style>/<script> body must never
    be embedded as document text (it floods retrieval with CSS vectors)."""
    (tmp_path / "api.html").write_text(_STYLED_HTML, encoding="utf-8")
    chunks = build_official_chunks(str(tmp_path), _TOK)
    assert chunks
    blob = "\n".join(c["content"] for c in chunks)
    assert "/*!sc*/" not in blob
    assert "data-theme" not in blob
    assert "padding" not in blob
    assert "window.__DOCS__" not in blob
    # The real prose on the same page still survives.
    assert "Upload a file to the shared folder." in blob
    assert "A 200 response returns the file metadata." in blob


def test_json_ld_metadata_is_stripped_with_the_other_script_assets(tmp_path):
    """schema.org metadata is machine annotation, not page prose: it rides the
    same <script> strip rather than being embedded as a chunk of JSON."""
    (tmp_path / "ld.html").write_text(
        "<h1>Deploy a process</h1>\n"
        '<script type="application/ld+json">'
        '{"@context":"https://schema.org","@type":"TechArticle"}</script>\n'
        "<p>Attach the process to an environment first.</p>\n",
        encoding="utf-8",
    )
    chunks = build_official_chunks(str(tmp_path), _TOK)
    blob = "\n".join(c["content"] for c in chunks)
    assert "schema.org" not in blob
    assert "TechArticle" not in blob
    assert "Attach the process to an environment first." in blob


# --- full pipeline + CLI ----------------------------------------------------------

def _stage_companion(tmp_path):
    from companion import sha256_hex

    staging = tmp_path / "companion_sources"
    rel = "references/components/map_component_functions.md"
    md_abs = staging / rel
    md_abs.parent.mkdir(parents=True)
    md_abs.write_text(SMALL_MD, encoding="utf-8")
    manifest = {
        "repo": _UPSTREAM["repo"], "commit": _UPSTREAM["commit"],
        "files": [dict(_ENTRY, path=rel, sha256=sha256_hex(SMALL_MD))],
    }
    manifest_path = tmp_path / "companion_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config_path = tmp_path / "companion_sources.json"
    config_path.write_text(json.dumps(
        {"repo": _UPSTREAM["repo"], "files": [{k: v for k, v in _ENTRY.items()
                                               if k in ("path", "area", "title",
                                                        "sections", "drop_sections")}]}
    ), encoding="utf-8")
    return str(staging), str(manifest_path), str(config_path)


def test_build_s5_chunks_combines_both_corpora_with_contiguous_indices(tmp_path):
    official = tmp_path / "kb"
    official.mkdir()
    (official / "db.html").write_text(_OFFICIAL_HTML, encoding="utf-8")
    staging, manifest_path, config_path = _stage_companion(tmp_path)

    chunks = build_s5_chunks(
        official_input_dir=str(official),
        companion_manifest_path=manifest_path,
        companion_staging=staging,
        companion_config_path=config_path,
        tokenizer=_TOK,
    )
    assert {c["source_type"] for c in chunks} == {"official", "companion_reference"}
    by_page = {}
    for chunk in chunks:
        by_page.setdefault(chunk["page_key"], []).append(chunk["chunk_index"])
    for page_key, indices in by_page.items():
        assert sorted(indices) == list(range(len(indices))), page_key


def test_cli_writes_s5_chunks_jsonl(tmp_path):
    official = tmp_path / "kb"
    official.mkdir()
    (official / "db.html").write_text(_OFFICIAL_HTML, encoding="utf-8")
    staging, manifest_path, config_path = _stage_companion(tmp_path)
    output = tmp_path / "chunks_s5.jsonl"

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(
        [sys.executable, os.path.join(repo, "s5_pipeline.py"),
         "--input", str(official), "--companion-input", staging,
         "--companion-manifest", manifest_path,
         "--companion-config", config_path,
         "--output", str(output), "--tokenizer", "fake"],
        capture_output=True, text=True, cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    chunks = [json.loads(line) for line in output.read_text().splitlines() if line]
    assert chunks
    assert all(S5_CHUNK_FIELDS <= set(c) for c in chunks)


def test_cli_snapshot_flag_is_mutually_exclusive_with_inputs(tmp_path):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(
        [sys.executable, os.path.join(repo, "s5_pipeline.py"),
         "--snapshot", str(tmp_path), "--input", "other/",
         "--output", str(tmp_path / "out.jsonl"), "--tokenizer", "fake"],
        capture_output=True, text=True, cwd=str(tmp_path),
    )
    assert result.returncode == 1
    assert "mutually exclusive" in result.stdout


def test_corrupted_staged_companion_fails_closed(tmp_path):
    staging, manifest_path, config_path = _stage_companion(tmp_path)
    staged = os.path.join(staging, "references/components/map_component_functions.md")
    with open(staged, "a", encoding="utf-8") as f:
        f.write("tampered")
    with pytest.raises(ValueError, match="does not match the manifest"):
        build_s5_chunks(
            official_input_dir=str(tmp_path / "missing"),
            companion_manifest_path=manifest_path,
            companion_staging=staging,
            companion_config_path=config_path,
            tokenizer=_TOK,
        )
