#!/usr/bin/env python3
"""Frozen source snapshot for candidate builds.

Every kb-25 candidate (C0, B5, B56, B567) must be built from ONE immutable
source snapshot so feature-attribution comparisons are never confounded by
live official-document drift. A snapshot directory holds the official scrape
artifacts, the verified Companion Markdown, and both companion manifests, plus
a ``snapshot_manifest.json`` recording every file's sha256 and the combined
``source_snapshot_sha256``. Chunking CLIs accept ``--snapshot <dir>`` and read
inputs ONLY from a snapshot that verifies.

Usage:
    python snapshot.py freeze --official knowledge_base/ \
        --companion-staging companion_sources/ \
        --companion-manifest companion_manifest.json \
        --companion-config companion_sources.json --out snapshot/
    python snapshot.py verify --snapshot snapshot/
"""
import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone

from companion import sha256_hex

SNAPSHOT_MANIFEST_NAME = "snapshot_manifest.json"
SNAPSHOT_SCHEMA = 1


def _file_sha256(path):
    with open(path, "rb") as f:
        return sha256_hex(f.read())


def _combined_sha256(file_records):
    """source_snapshot_sha256 over the sorted (path, sha256) pairs.

    Content-derived only — created_at and layout order do not affect it, so
    re-freezing identical sources always reproduces the same hash.
    """
    lines = "\n".join(
        f"{rec['path']}\t{rec['sha256']}"
        for rec in sorted(file_records, key=lambda r: r["path"])
    )
    return sha256_hex(lines)


def freeze_snapshot(*, official_dir, companion_staging, companion_manifest,
                    companion_config, out_dir):
    """Copy all build inputs into ``out_dir`` and record their hashes.

    Fails closed: a companion file missing from staging, or staged content that
    does not hash to the sha256 its manifest entry recorded, aborts the freeze —
    a snapshot must never bless bytes that were not verifiably fetched.
    Returns the written snapshot manifest dict.
    """
    with open(companion_manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    records = []

    def _copy(src, rel_dest):
        dest = os.path.join(out_dir, rel_dest)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(src, dest)
        records.append({
            "path": rel_dest,
            "sha256": _file_sha256(dest),
            "bytes": os.path.getsize(dest),
        })

    for filename in sorted(os.listdir(official_dir)):
        src = os.path.join(official_dir, filename)
        if os.path.isfile(src):
            _copy(src, f"official/{filename}")

    for entry in manifest.get("files", []):
        rel = entry["path"]
        src = os.path.join(companion_staging, rel)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"Staged companion file missing: {src}")
        expected = entry.get("sha256")
        with open(src, "r", encoding="utf-8") as f:
            staged_text = f.read()
        if expected and sha256_hex(staged_text) != expected:
            raise ValueError(
                f"Staged companion file does not match its manifest sha256: {rel}. "
                "Re-run fetch_companion.py before freezing a snapshot."
            )
        _copy(src, f"companion/{rel}")

    _copy(companion_manifest, "companion_manifest.json")
    _copy(companion_config, "companion_sources.json")

    snapshot_manifest = {
        "schema": SNAPSHOT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "upstream": {
            "repo": manifest.get("repo", ""),
            "commit": manifest.get("commit", ""),
        },
        "files": sorted(records, key=lambda r: r["path"]),
        "source_snapshot_sha256": _combined_sha256(records),
    }
    with open(os.path.join(out_dir, SNAPSHOT_MANIFEST_NAME), "w",
              encoding="utf-8") as f:
        json.dump(snapshot_manifest, f, indent=2)
    return snapshot_manifest


def verify_snapshot(snapshot_dir):
    """Re-hash every recorded file and the combined hash; return the latter.

    Raises ValueError naming the offending file on any mismatch, on a missing
    file, and on any file present in the snapshot that the manifest does not
    record (an unrecorded file would be chunked while unhashed — fail closed).
    """
    manifest_path = os.path.join(snapshot_dir, SNAPSHOT_MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        raise ValueError(f"Snapshot manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    recorded = {rec["path"]: rec for rec in manifest.get("files", [])}
    for rel, rec in sorted(recorded.items()):
        path = os.path.join(snapshot_dir, rel)
        if not os.path.isfile(path):
            raise ValueError(f"Snapshot file missing: {rel}")
        actual = _file_sha256(path)
        if actual != rec["sha256"]:
            raise ValueError(
                f"Snapshot file drifted: {rel}\n"
                f"  recorded sha256: {rec['sha256']}\n"
                f"  actual sha256:   {actual}"
            )

    on_disk = set()
    for dirpath, _dirnames, filenames in os.walk(snapshot_dir):
        for filename in filenames:
            rel = os.path.relpath(os.path.join(dirpath, filename), snapshot_dir)
            rel = rel.replace(os.sep, "/")
            if rel != SNAPSHOT_MANIFEST_NAME:
                on_disk.add(rel)
    extra = sorted(on_disk - set(recorded))
    if extra:
        raise ValueError(
            f"Snapshot contains unrecorded file(s): {extra}. A frozen snapshot "
            "must record every input byte it serves."
        )

    combined = _combined_sha256(list(recorded.values()))
    if combined != manifest.get("source_snapshot_sha256"):
        raise ValueError(
            "Snapshot combined hash mismatch: recorded "
            f"{manifest.get('source_snapshot_sha256')!r}, recomputed {combined!r}"
        )
    return combined


def resolve_snapshot_inputs(snapshot_dir):
    """Verify the snapshot, then return the four chunking input paths.

    The ONLY sanctioned way a chunking CLI consumes a snapshot: verification
    runs first, so a drifted snapshot aborts before any chunking happens.
    """
    verify_snapshot(snapshot_dir)
    return {
        "input": os.path.join(snapshot_dir, "official"),
        "companion_input": os.path.join(snapshot_dir, "companion"),
        "companion_manifest": os.path.join(snapshot_dir, "companion_manifest.json"),
        "companion_config": os.path.join(snapshot_dir, "companion_sources.json"),
    }


def main():
    parser = argparse.ArgumentParser(description="Freeze/verify a source snapshot")
    sub = parser.add_subparsers(dest="command", required=True)

    freeze = sub.add_parser("freeze", help="Copy build inputs into a snapshot")
    freeze.add_argument("--official", default="knowledge_base/")
    freeze.add_argument("--companion-staging", default="companion_sources/")
    freeze.add_argument("--companion-manifest", default="companion_manifest.json")
    freeze.add_argument("--companion-config", default="companion_sources.json")
    freeze.add_argument("--out", required=True)

    verify = sub.add_parser("verify", help="Verify a frozen snapshot")
    verify.add_argument("--snapshot", required=True)

    args = parser.parse_args()
    if args.command == "freeze":
        manifest = freeze_snapshot(
            official_dir=args.official,
            companion_staging=args.companion_staging,
            companion_manifest=args.companion_manifest,
            companion_config=args.companion_config,
            out_dir=args.out,
        )
        print(f"source_snapshot_sha256: {manifest['source_snapshot_sha256']}")
        print(f"files: {len(manifest['files'])}")
    else:
        combined = verify_snapshot(args.snapshot)
        print(f"source_snapshot_sha256: {combined}")
        print("snapshot OK")


if __name__ == "__main__":
    sys.exit(main() or 0)
