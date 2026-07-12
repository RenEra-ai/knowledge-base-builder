"""Tests for the companion Markdown ingestion path in chunk_docs.py."""

import json

import pytest

from chunk_docs import (
    assign_chunk_indices,
    chunk_file,
    chunk_markdown_file,
    process_companion,
)
from companion import sha256_hex

UPSTREAM = {"repo": "OfficialBoomi/boomi-integration", "commit": "d" * 40}


def _entry(**overrides):
    entry = {
        "path": "references/components/map_component.md",
        "area": "Components",
        "title": "Map Component",
        # process_companion requires a sha256 on every entry (a manifest must not be
        # able to disable its own integrity check), and verifies it against the staged
        # file. MD is what the tests below stage.
        "sha256": sha256_hex(MD),
        "raw_url": "https://raw.githubusercontent.com/OfficialBoomi/boomi-integration/" + "d" * 40 + "/references/components/map_component.md",
        "blob_url": "https://github.com/OfficialBoomi/boomi-integration/blob/" + "d" * 40 + "/references/components/map_component.md",
        "latest_url": "https://github.com/OfficialBoomi/boomi-integration/blob/main/references/components/map_component.md",
        "sections": None,
        "drop_sections": None,
        "strip_xml_blocks": False,
    }
    entry.update(overrides)
    return entry


MD = """# Map Component

## Contents

- links

## Purpose

The map component transforms one profile into another with enough explanatory
text here to comfortably exceed the minimum token threshold for a standalone
chunk so it is not merged away during chunk assembly in the pipeline.

## Key Concepts

Maps depend on source and target profiles. This section also carries enough
descriptive prose to stand on its own as an independently indexable chunk in
the resulting knowledge base output for the test to assert against cleanly.
"""


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_chunk_markdown_file_page_key_and_provenance(tmp_path):
    md = _write(tmp_path, "map.md", MD)
    chunks = chunk_markdown_file(md, _entry(), UPSTREAM, min_tokens=0, max_tokens=100_000)

    assert chunks
    for c in chunks:
        assert c["page_key"] == "companion://OfficialBoomi/boomi-integration/references/components/map_component.md"
        assert c["source_type"] == "companion_reference"
        assert c["verification_status"] == "companion_unverified"
        assert c["upstream_repo"] == "OfficialBoomi/boomi-integration"
        assert c["upstream_commit"] == "d" * 40
        assert c["source_path"] == "references/components/map_component.md"
        assert c["source_url"].startswith("https://github.com/")
        assert c["raw_url"].startswith("https://raw.githubusercontent.com/")
        assert c["latest_url"].endswith("/blob/main/references/components/map_component.md")
        assert c["category"] == "Companion Reference"
        assert c["breadcrumb"] == "Companion Reference > Components > Map Component"
        assert c["content"].strip()  # never blank


def test_chunk_markdown_file_drops_contents_toc(tmp_path):
    md = _write(tmp_path, "map.md", MD)
    chunks = chunk_markdown_file(md, _entry(), UPSTREAM, min_tokens=0, max_tokens=100_000)
    headings = {c["section_heading"] for c in chunks}
    assert "Contents" not in headings
    assert "Purpose" in headings and "Key Concepts" in headings


def test_chunk_markdown_file_section_allowlist_slices(tmp_path):
    md = _write(tmp_path, "map.md", MD)
    chunks = chunk_markdown_file(
        md, _entry(sections=["key concepts"]), UPSTREAM, min_tokens=0, max_tokens=100_000
    )
    headings = {c["section_heading"] for c in chunks}
    assert headings == {"Key Concepts"}


def test_chunk_markdown_file_strips_large_xml(tmp_path):
    big = "<bns:Component>\n" + ("  <field/>\n" * 400) + "</bns:Component>"
    md_text = f"# Doc\n\n## Structure\n\nLead-in prose.\n\n```xml\n{big}\n```\n"
    md = _write(tmp_path, "op.md", md_text)
    chunks = chunk_markdown_file(
        md, _entry(path="references/components/x.md", strip_xml_blocks=True),
        UPSTREAM, min_tokens=0, max_tokens=100_000,
    )
    combined = "\n".join(c["content"] for c in chunks)
    assert "bns:Component" not in combined
    assert "Lead-in prose" in combined


def test_chunk_markdown_file_mixed_with_official_contiguous_index(tmp_path):
    md = _write(tmp_path, "map.md", MD)
    companion = chunk_markdown_file(md, _entry(), UPSTREAM, min_tokens=0, max_tokens=100_000)

    html = (
        "<h1>Official</h1>"
        "<p><strong>Path:</strong> <a href='https://help.boomi.com/docs/x'>X</a></p>"
        "<h2>Sec</h2><p>Official body text long enough to be its own chunk here.</p>"
    )
    hf = tmp_path / "off.html"
    hf.write_text(html, encoding="utf-8")
    official = chunk_file(str(hf), hf.name, min_tokens=0, max_tokens=100_000, verbose=False)

    allc = official + companion
    assign_chunk_indices(allc)

    # Every page_key must be contiguous 0..n-1, and the two key spaces disjoint.
    by_key = {}
    for c in allc:
        by_key.setdefault(c["page_key"], []).append(c["chunk_index"])
    for key, idxs in by_key.items():
        assert sorted(idxs) == list(range(len(idxs))), key
    assert any(k.startswith("companion://") for k in by_key)
    assert any(k.startswith("https://") for k in by_key)


def test_chunk_markdown_file_oversize_does_not_duplicate_heading(tmp_path):
    # A section larger than max_tokens is paragraph-split; the heading must be
    # prepended exactly once per sub-chunk, never doubled ("Purpose\nPurpose").
    body = "\n\n".join(f"Paragraph {i} carries enough words to matter." for i in range(8))
    md = _write(tmp_path, "big.md", f"# Doc\n\n## Purpose\n\n{body}\n")
    chunks = chunk_markdown_file(
        md, _entry(path="references/components/x.md"), UPSTREAM,
        min_tokens=0, max_tokens=20,
    )
    assert len(chunks) > 1  # oversize split actually happened
    for c in chunks:
        assert not c["content"].startswith("Purpose\nPurpose")
        assert c["content"].count("Purpose") <= 1


def test_process_companion_rejects_denied_manifest_path(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    manifest = {"repo": "R", "commit": "d" * 40, "files": [_entry(path="/etc/passwd")]}
    mpath = tmp_path / "companion_manifest.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="disallowed path"):
        process_companion(str(mpath), str(staging), 0, 100_000)


def test_process_companion_reads_staging_and_fails_on_missing(tmp_path):
    staging = tmp_path / "staging"
    (staging / "references/components").mkdir(parents=True)
    (staging / "references/components/map_component.md").write_text(MD, encoding="utf-8")

    manifest = {
        "repo": "OfficialBoomi/boomi-integration",
        "commit": "d" * 40,
        "files": [_entry()],
    }
    mpath = tmp_path / "companion_manifest.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")

    chunks = process_companion(str(mpath), str(staging), 0, 100_000)
    assert chunks and all(c["source_type"] == "companion_reference" for c in chunks)

    # A manifest naming a file that was never staged is a hard error.
    manifest["files"].append(_entry(path="references/components/missing.md"))
    mpath.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        process_companion(str(mpath), str(staging), 0, 100_000)


def test_process_companion_fails_when_a_staged_file_does_not_match_its_sha256(tmp_path):
    """Content that is not what was fetched must never be stamped with the pinned commit.

    Every companion chunk carries upstream_commit and the raw_url of the source file, so
    chunking a modified or partially-fetched staged copy would ship content asserting a
    provenance it does not have.
    """
    staging = tmp_path / "staging"
    (staging / "references/components").mkdir(parents=True)
    md = staging / "references/components/map_component.md"
    md.write_text(MD, encoding="utf-8")

    manifest = {"repo": "R", "commit": "d" * 40, "files": [_entry()]}
    mpath = tmp_path / "companion_manifest.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")

    # Untampered: the hash agrees and the build proceeds.
    assert process_companion(str(mpath), str(staging), 0, 100_000)

    # A hand-edit, or a truncated file from an interrupted re-fetch.
    md.write_text(MD + "\n## Purpose\nnever fetched from upstream\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the manifest"):
        process_companion(str(mpath), str(staging), 0, 100_000)


def test_process_companion_rejects_a_manifest_entry_with_no_sha256(tmp_path):
    """A manifest must not be able to switch off its own integrity check."""
    staging = tmp_path / "staging"
    (staging / "references/components").mkdir(parents=True)
    (staging / "references/components/map_component.md").write_text(MD, encoding="utf-8")

    entry = _entry()
    del entry["sha256"]
    mpath = tmp_path / "companion_manifest.json"
    mpath.write_text(
        json.dumps({"repo": "R", "commit": "d" * 40, "files": [entry]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="no sha256"):
        process_companion(str(mpath), str(staging), 0, 100_000)


def test_process_companion_fails_on_stale_curation_policy(tmp_path):
    """Editing companion_sources.json without re-fetching must not build silently.

    The manifest snapshots sections/drop_sections at fetch time and chunking reads
    them from there, so a stale manifest fails OPEN: the sections a curator just
    excluded stay in the corpus and the build still reports success.
    """
    staging = tmp_path / "staging"
    (staging / "references/components").mkdir(parents=True)
    (staging / "references/components/map_component.md").write_text(MD, encoding="utf-8")

    entry = _entry()
    entry["sections"] = ["purpose"]
    manifest = {"repo": "R", "commit": "d" * 40, "files": [entry]}
    mpath = tmp_path / "companion_manifest.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")

    cpath = tmp_path / "companion_sources.json"
    # Same file, but the curator has since tightened the allowlist.
    config = {"files": [dict(entry, sections=["purpose", "key management"])]}
    cpath.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="drifted"):
        process_companion(str(mpath), str(staging), 0, 100_000, config_path=str(cpath))

    # Re-fetching (manifest now agrees with the config) builds again.
    mpath.write_text(json.dumps({**manifest, "files": config["files"]}), encoding="utf-8")
    assert process_companion(str(mpath), str(staging), 0, 100_000, config_path=str(cpath))
