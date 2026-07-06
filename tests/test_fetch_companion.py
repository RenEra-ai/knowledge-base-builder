"""Tests for fetch_companion.py: config validation, pinned-commit enforcement,
allowlist-only fetching, failure conditions, and manifest shape."""

import json
from datetime import datetime, timezone

import pytest

import fetch_companion
from fetch_companion import CompanionFetchError

PINNED = "19aacdd0aa4c9c83f5d87e9b89fb213044447f52"
FIXED_NOW = datetime(2026, 7, 6, tzinfo=timezone.utc)


def _config(**overrides):
    config = {
        "repo": "OfficialBoomi/boomi-integration",
        "repo_url": "https://github.com/OfficialBoomi/boomi-integration",
        "commit": PINNED,
        "license": "BSD-2-Clause",
        "source_type": "companion_reference",
        "verification_status": "companion_unverified",
        "raw_url_template": "https://raw.githubusercontent.com/OfficialBoomi/boomi-integration/{commit}/{path}",
        "blob_url_template": "https://github.com/OfficialBoomi/boomi-integration/blob/{commit}/{path}",
        "latest_blob_url_template": "https://github.com/OfficialBoomi/boomi-integration/blob/main/{path}",
        "files": [
            {"path": "references/components/map_component.md", "area": "Components", "title": "Map"},
            {"path": "references/guides/boomi_patterns.md", "area": "Guides", "title": "Patterns"},
        ],
    }
    config.update(overrides)
    return config


def _write_config(tmp_path, config):
    p = tmp_path / "companion_sources.json"
    p.write_text(json.dumps(config), encoding="utf-8")
    return str(p)


class _Resp:
    def __init__(self, status_code=200, text="# Doc\n\nbody", headers=None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {"content-type": "text/plain; charset=utf-8"}


class _Getter:
    """Records requested URLs and returns a configured response per URL."""

    def __init__(self, default=None, by_url=None):
        self.default = default or _Resp()
        self.by_url = by_url or {}
        self.calls = []

    def __call__(self, url):
        self.calls.append(url)
        return self.by_url.get(url, self.default)


# --- config validation --------------------------------------------------------

def test_load_and_validate_config_accepts_valid(tmp_path):
    path = _write_config(tmp_path, _config())
    config = fetch_companion.load_and_validate_config(path)
    assert config["commit"] == PINNED


def test_load_and_validate_config_rejects_moving_ref(tmp_path):
    path = _write_config(tmp_path, _config(commit="main"))
    with pytest.raises(CompanionFetchError, match="40-char"):
        fetch_companion.load_and_validate_config(path)


def test_load_and_validate_config_rejects_template_without_commit(tmp_path):
    bad = _config(raw_url_template="https://raw.githubusercontent.com/o/r/main/{path}")
    path = _write_config(tmp_path, bad)
    with pytest.raises(CompanionFetchError, match="commit"):
        fetch_companion.load_and_validate_config(path)


def test_load_and_validate_config_rejects_commit_not_a_path_segment(tmp_path):
    # {commit} present but as a query param, not the ref path segment.
    bad = _config(raw_url_template="https://raw.githubusercontent.com/o/r/x?ref={commit}&p={path}")
    path = _write_config(tmp_path, bad)
    with pytest.raises(CompanionFetchError, match="path segment"):
        fetch_companion.load_and_validate_config(path)


def test_load_and_validate_config_rejects_empty_files(tmp_path):
    path = _write_config(tmp_path, _config(files=[]))
    with pytest.raises(CompanionFetchError, match="non-empty list"):
        fetch_companion.load_and_validate_config(path)


def test_load_and_validate_config_rejects_denied_path(tmp_path):
    path = _write_config(tmp_path, _config(files=[{"path": "scripts/x.sh"}]))
    with pytest.raises(CompanionFetchError, match="not allowed"):
        fetch_companion.load_and_validate_config(path)


def test_load_and_validate_config_rejects_duplicate_path(tmp_path):
    dup = [{"path": "references/a.md"}, {"path": "references/a.md"}]
    path = _write_config(tmp_path, _config(files=dup))
    with pytest.raises(CompanionFetchError, match="Duplicate"):
        fetch_companion.load_and_validate_config(path)


# --- URL resolution -----------------------------------------------------------

def test_resolve_urls_pins_commit_in_raw_and_blob():
    config = _config()
    raw, blob, latest = fetch_companion.resolve_urls(config, config["files"][0])
    assert PINNED in raw and PINNED in blob
    assert "/blob/main/" in latest and PINNED not in latest


# --- fetch_text failure conditions -------------------------------------------

def test_fetch_text_ok_returns_body():
    getter = _Getter(_Resp(text="# Title\n\nhello"))
    assert fetch_companion.fetch_text(getter, "u") == "# Title\n\nhello"


def test_fetch_text_fails_on_non_200():
    getter = _Getter(_Resp(status_code=404, text="404: Not Found"))
    with pytest.raises(CompanionFetchError, match="404"):
        fetch_companion.fetch_text(getter, "u")


def test_fetch_text_fails_on_empty_body():
    getter = _Getter(_Resp(text="   \n  "))
    with pytest.raises(CompanionFetchError, match="empty"):
        fetch_companion.fetch_text(getter, "u")


def test_fetch_text_fails_on_html_body():
    getter = _Getter(_Resp(text="<!DOCTYPE html><html>nope</html>",
                           headers={"content-type": "text/html"}))
    with pytest.raises(CompanionFetchError, match="non-Markdown"):
        fetch_companion.fetch_text(getter, "u")


# --- run(): allowlist-only fetch + manifest ----------------------------------

def test_run_fetches_only_allowlisted_and_writes_manifest(tmp_path):
    config = _config()
    config_path = _write_config(tmp_path, config)
    staging = str(tmp_path / "staging")
    manifest_path = str(tmp_path / "companion_manifest.json")

    getter = _Getter(_Resp(text="# Doc\n\nbody content here"))
    manifest = fetch_companion.run(config_path, staging, manifest_path, getter, now=FIXED_NOW)

    # Exactly one GET per allowlisted file — no recursion, no tree/contents API.
    assert len(getter.calls) == 2
    expected_raw = {
        fetch_companion.resolve_urls(config, e)[0] for e in config["files"]
    }
    assert set(getter.calls) == expected_raw

    # Manifest shape + provenance.
    assert manifest["commit"] == PINNED
    assert manifest["fetched_at"] == FIXED_NOW.isoformat()
    assert len(manifest["files"]) == 2
    entry = manifest["files"][0]
    assert entry["path"] == "references/components/map_component.md"
    assert PINNED in entry["raw_url"] and PINNED in entry["blob_url"]
    assert "/blob/main/" in entry["latest_url"]
    assert entry["bytes"] > 0
    assert len(entry["sha256"]) == 64

    # Staged file written to disk under its repo-relative path.
    with open(f"{staging}/references/components/map_component.md") as f:
        assert "body content here" in f.read()

    # Manifest file persisted.
    assert json.load(open(manifest_path))["commit"] == PINNED


def test_run_aborts_when_any_source_missing(tmp_path):
    config = _config()
    config_path = _write_config(tmp_path, config)
    missing_url = fetch_companion.resolve_urls(config, config["files"][1])[0]
    getter = _Getter(
        default=_Resp(text="# ok\n\nbody"),
        by_url={missing_url: _Resp(status_code=404, text="404: Not Found")},
    )
    with pytest.raises(CompanionFetchError):
        fetch_companion.run(config_path, str(tmp_path / "s"),
                            str(tmp_path / "m.json"), getter, now=FIXED_NOW)
