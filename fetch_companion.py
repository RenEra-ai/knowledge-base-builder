#!/usr/bin/env python3
"""Fetch the curated Companion supplemental corpus from a pinned GitHub commit.

Reads ``companion_sources.json`` (one pinned repo + an explicit, non-recursive
allowlist) and downloads ONLY the allowlisted Markdown paths from their pinned
``raw.githubusercontent.com`` URLs into a staging directory, alongside a
``companion_manifest.json`` recording provenance + SHA-256 for each file.

Fail-closed by design: a non-pinned commit, a template that hardcodes a moving
ref, a disallowed path, or any allowlisted source that is missing / empty /
non-Markdown aborts the whole run with a non-zero exit. Unlike the official-docs
scraper (main.py), every configured source here is mandatory.

Usage:
    python fetch_companion.py [--config companion_sources.json]
                              [--staging companion_sources/]
                              [--manifest companion_manifest.json]
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

from companion import build_url, is_denied_path, is_pinned_commit, sha256_hex

DEFAULT_CONFIG = "companion_sources.json"
DEFAULT_STAGING = "companion_sources/"
DEFAULT_MANIFEST = "companion_manifest.json"

_REQUIRED_TOP_KEYS = (
    "repo", "commit", "raw_url_template", "blob_url_template",
    "latest_blob_url_template", "files",
)


class CompanionFetchError(Exception):
    """Raised when the Companion source config or a fetch is invalid."""


def load_and_validate_config(config_path):
    """Load companion_sources.json and enforce the pinned-source invariants.

    Raises CompanionFetchError on: missing file/keys, a non-40-hex commit, a
    raw/blob template that omits ``{commit}`` (i.e. hardcodes a moving ref) or
    ``{path}``, an empty allowlist, or any denied/duplicate file path.
    """
    if not os.path.isfile(config_path):
        raise CompanionFetchError(f"Companion config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise CompanionFetchError("Companion config must be a JSON object")

    missing = [k for k in _REQUIRED_TOP_KEYS if k not in config]
    if missing:
        raise CompanionFetchError(f"Companion config missing key(s): {missing}")

    commit = config["commit"]
    if not is_pinned_commit(commit):
        raise CompanionFetchError(
            f"Companion commit must be a full 40-char hex sha (got {commit!r}); "
            "moving refs like main/master/HEAD/tags are rejected"
        )

    for key in ("raw_url_template", "blob_url_template"):
        template = config[key]
        if "{commit}" not in template or "{path}" not in template:
            raise CompanionFetchError(
                f"{key} must embed both {{commit}} and {{path}} so it pins the "
                f"commit; got {template!r}"
            )
        # {commit} must be its own URL path segment (…/{commit}/…), i.e. the git
        # ref, not a query param or a fragment of some other segment. A template
        # that additionally hardcodes a moving ref elsewhere would resolve to a
        # 404 and be caught by the mandatory-source fetch check below.
        if "/{commit}/" not in template:
            raise CompanionFetchError(
                f"{key} must place {{commit}} as its own path segment "
                f"(.../{{commit}}/...) so the pinned sha is the ref; got {template!r}"
            )
    if "{path}" not in config["latest_blob_url_template"]:
        raise CompanionFetchError("latest_blob_url_template must embed {path}")

    files = config["files"]
    if not isinstance(files, list) or not files:
        raise CompanionFetchError("Companion config 'files' must be a non-empty list")

    seen = set()
    for entry in files:
        path = entry.get("path") if isinstance(entry, dict) else None
        if not path:
            raise CompanionFetchError(f"Companion file entry missing 'path': {entry!r}")
        if is_denied_path(path):
            raise CompanionFetchError(f"Companion file path is not allowed: {path!r}")
        if path in seen:
            raise CompanionFetchError(f"Duplicate Companion file path: {path!r}")
        seen.add(path)

    return config


def resolve_urls(config, entry):
    """Return (raw_url, blob_url, latest_url) for one allowlist entry."""
    commit, path = config["commit"], entry["path"]
    return (
        build_url(config["raw_url_template"], commit, path),
        build_url(config["blob_url_template"], commit, path),
        build_url(config["latest_blob_url_template"], commit, path),
    )


def _looks_like_markdown(text, headers):
    """Reject HTML error bodies masquerading as a 200 Markdown response."""
    content_type = (headers or {}).get("content-type", "").lower()
    if "text/html" in content_type:
        return False
    head = text.lstrip()[:512].lower()
    if head.startswith("<!doctype html") or head.startswith("<html"):
        return False
    return True


def fetch_text(http_get, raw_url):
    """GET a raw URL and return its text, failing on any non-Markdown result.

    ``http_get(url)`` must return an object with ``.status_code``, ``.text`` and
    ``.headers`` (an httpx.Response satisfies this). Raises CompanionFetchError
    on non-200, empty/whitespace body, or an HTML error page.
    """
    resp = http_get(raw_url)
    status = getattr(resp, "status_code", None)
    if status != 200:
        raise CompanionFetchError(f"Fetch failed [{status}] for {raw_url}")
    text = resp.text
    if not text or not text.strip():
        raise CompanionFetchError(f"Fetched empty body for {raw_url}")
    if not _looks_like_markdown(text, getattr(resp, "headers", {})):
        raise CompanionFetchError(f"Fetched non-Markdown (HTML?) body for {raw_url}")
    return text


def build_manifest_entry(config, entry, raw_url, blob_url, latest_url, text):
    """Assemble one companion_manifest.json file record (pure)."""
    return {
        "path": entry["path"],
        "area": entry.get("area", ""),
        "title": entry.get("title", ""),
        "priority": entry.get("priority"),
        "sections": entry.get("sections"),
        "drop_sections": entry.get("drop_sections"),
        "strip_xml_blocks": bool(entry.get("strip_xml_blocks", False)),
        "raw_url": raw_url,
        "blob_url": blob_url,
        "latest_url": latest_url,
        "sha256": sha256_hex(text),
        "bytes": len(text.encode("utf-8")),
    }


def build_manifest(config, file_entries, fetched_at):
    """Assemble the top-level companion_manifest.json (pure)."""
    return {
        "repo": config["repo"],
        "repo_url": config.get("repo_url", ""),
        "commit": config["commit"],
        "license": config.get("license", ""),
        "source_type": config.get("source_type", "companion_reference"),
        "verification_status": config.get("verification_status", "companion_unverified"),
        "raw_url_template": config["raw_url_template"],
        "blob_url_template": config["blob_url_template"],
        "latest_blob_url_template": config["latest_blob_url_template"],
        "fetched_at": fetched_at,
        "files": file_entries,
    }


def run(config_path, staging_dir, manifest_path, http_get, now=None):
    """Fetch all allowlisted files and write staging + manifest. Returns manifest."""
    config = load_and_validate_config(config_path)
    fetched_at = (now or datetime.now(timezone.utc)).isoformat()

    file_entries = []
    for entry in config["files"]:
        raw_url, blob_url, latest_url = resolve_urls(config, entry)
        text = fetch_text(http_get, raw_url)

        dest = os.path.join(staging_dir, entry["path"])
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(text)

        file_entries.append(
            build_manifest_entry(config, entry, raw_url, blob_url, latest_url, text)
        )
        print(f"  fetched {entry['path']} ({len(text.encode('utf-8')):,} bytes)")

    manifest = build_manifest(config, file_entries, fetched_at)
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _httpx_getter():
    """Build an httpx-backed getter matching main.py's client conventions."""
    import httpx

    client = httpx.Client(
        follow_redirects=True,
        timeout=30.0,
        headers={"User-Agent": "knowledge-base-builder/companion-fetch"},
    )
    return client.get


def main():
    parser = argparse.ArgumentParser(description="Fetch the Companion supplemental corpus")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--staging", default=DEFAULT_STAGING)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    try:
        manifest = run(args.config, args.staging, args.manifest, _httpx_getter())
    except CompanionFetchError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)

    print(f"\nFetched {len(manifest['files'])} companion file(s) at commit "
          f"{manifest['commit'][:10]} -> {args.staging}")
    print(f"Wrote manifest: {args.manifest}")


if __name__ == "__main__":
    main()
