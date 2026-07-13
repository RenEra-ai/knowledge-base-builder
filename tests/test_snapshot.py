"""Tests for the frozen source snapshot (freeze / verify / input resolution)."""
import json
import os

import pytest

from companion import sha256_hex
from snapshot import (
    SNAPSHOT_MANIFEST_NAME,
    freeze_snapshot,
    resolve_snapshot_inputs,
    verify_snapshot,
)


def _make_sources(tmp_path):
    """Lay out a minimal official scrape + verified companion staging."""
    official = tmp_path / "knowledge_base"
    official.mkdir()
    (official / "aaa_page.html").write_text(
        "<h1>Page A</h1>\n<p>alpha</p>", encoding="utf-8"
    )
    (official / "bbb_page.html").write_text(
        "<h1>Page B</h1>\n<p>beta</p>", encoding="utf-8"
    )

    staging = tmp_path / "companion_sources"
    md_rel = "references/components/map_component.md"
    md_abs = staging / md_rel
    md_abs.parent.mkdir(parents=True)
    md_text = "# Map Component\n\n## Purpose\n\nMaps things.\n"
    md_abs.write_text(md_text, encoding="utf-8")

    manifest = {
        "repo": "OfficialBoomi/boomi-integration",
        "commit": "19aacdd0aa4c9c83f5d87e9b89fb213044447f52",
        "files": [{"path": md_rel, "sha256": sha256_hex(md_text)}],
    }
    manifest_path = tmp_path / "companion_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    config_path = tmp_path / "companion_sources.json"
    config_path.write_text(json.dumps({"repo": manifest["repo"]}), encoding="utf-8")

    return {
        "official_dir": str(official),
        "companion_staging": str(staging),
        "companion_manifest": str(manifest_path),
        "companion_config": str(config_path),
    }


def _freeze(tmp_path, **overrides):
    sources = _make_sources(tmp_path)
    sources.update(overrides)
    out_dir = str(tmp_path / "snapshot")
    manifest = freeze_snapshot(out_dir=out_dir, **sources)
    return out_dir, manifest


def test_freeze_then_verify_round_trips(tmp_path):
    out_dir, manifest = _freeze(tmp_path)
    combined = verify_snapshot(out_dir)
    assert combined == manifest["source_snapshot_sha256"]
    assert len(combined) == 64


def test_snapshot_manifest_records_upstream_and_sorted_files(tmp_path):
    out_dir, manifest = _freeze(tmp_path)
    assert manifest["upstream"] == {
        "repo": "OfficialBoomi/boomi-integration",
        "commit": "19aacdd0aa4c9c83f5d87e9b89fb213044447f52",
    }
    paths = [f["path"] for f in manifest["files"]]
    assert paths == sorted(paths)
    assert "official/aaa_page.html" in paths
    assert "companion/references/components/map_component.md" in paths
    assert "companion_manifest.json" in paths
    assert "companion_sources.json" in paths
    # The manifest file itself is not part of its own hash inputs.
    assert SNAPSHOT_MANIFEST_NAME not in paths
    on_disk = json.load(open(os.path.join(out_dir, SNAPSHOT_MANIFEST_NAME)))
    assert on_disk == manifest


def test_mutated_file_fails_verify_naming_the_file(tmp_path):
    out_dir, _ = _freeze(tmp_path)
    target = os.path.join(out_dir, "official", "aaa_page.html")
    with open(target, "a", encoding="utf-8") as f:
        f.write("<!-- drift -->")
    with pytest.raises(ValueError, match="official/aaa_page.html"):
        verify_snapshot(out_dir)


def test_missing_file_fails_verify(tmp_path):
    out_dir, _ = _freeze(tmp_path)
    os.remove(os.path.join(out_dir, "official", "bbb_page.html"))
    with pytest.raises(ValueError, match="official/bbb_page.html"):
        verify_snapshot(out_dir)


def test_unrecorded_extra_file_fails_verify(tmp_path):
    """A stray file inside the snapshot would be chunked while unhashed —
    fail closed rather than build from unrecorded bytes."""
    out_dir, _ = _freeze(tmp_path)
    with open(os.path.join(out_dir, "official", "zzz_stray.html"), "w") as f:
        f.write("<h1>stray</h1>")
    with pytest.raises(ValueError, match="zzz_stray.html"):
        verify_snapshot(out_dir)


def test_freeze_fails_on_missing_staged_companion_file(tmp_path):
    sources = _make_sources(tmp_path)
    os.remove(os.path.join(
        sources["companion_staging"], "references/components/map_component.md"
    ))
    with pytest.raises(FileNotFoundError, match="map_component.md"):
        freeze_snapshot(out_dir=str(tmp_path / "snapshot"), **sources)


def test_freeze_fails_on_staged_sha_mismatch(tmp_path):
    """Freezing corrupt staging would stamp the snapshot hash onto bytes that
    were never fetched — verify staged content against the companion manifest."""
    sources = _make_sources(tmp_path)
    staged = os.path.join(
        sources["companion_staging"], "references/components/map_component.md"
    )
    with open(staged, "a", encoding="utf-8") as f:
        f.write("tampered")
    with pytest.raises(ValueError, match="map_component.md"):
        freeze_snapshot(out_dir=str(tmp_path / "snapshot"), **sources)


def test_combined_hash_is_deterministic_across_freezes(tmp_path):
    out_a, manifest_a = _freeze(tmp_path)
    out_b = str(tmp_path / "snapshot_b")
    # Re-freeze the SAME sources into a second directory: identical content
    # must produce the identical source_snapshot_sha256 (created_at differs).
    manifest_b = freeze_snapshot(out_dir=out_b, **{
        "official_dir": str(tmp_path / "knowledge_base"),
        "companion_staging": str(tmp_path / "companion_sources"),
        "companion_manifest": str(tmp_path / "companion_manifest.json"),
        "companion_config": str(tmp_path / "companion_sources.json"),
    })
    assert manifest_a["source_snapshot_sha256"] == manifest_b["source_snapshot_sha256"]


def test_resolve_snapshot_inputs_verifies_then_returns_paths(tmp_path):
    out_dir, _ = _freeze(tmp_path)
    inputs = resolve_snapshot_inputs(out_dir)
    assert sorted(inputs) == [
        "companion_config", "companion_input", "companion_manifest", "input",
    ]
    assert os.path.isdir(inputs["input"])
    assert os.path.isdir(inputs["companion_input"])
    assert os.path.isfile(inputs["companion_manifest"])
    assert os.path.isfile(inputs["companion_config"])


def test_resolve_snapshot_inputs_fails_on_drifted_snapshot(tmp_path):
    out_dir, _ = _freeze(tmp_path)
    with open(os.path.join(out_dir, "official", "aaa_page.html"), "a") as f:
        f.write("x")
    with pytest.raises(ValueError, match="aaa_page.html"):
        resolve_snapshot_inputs(out_dir)
