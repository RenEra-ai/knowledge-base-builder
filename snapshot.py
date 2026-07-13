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

    Fails closed via the SAME companion gate the chunker uses
    (chunk_docs.verified_companion_texts): a missing staged file, a manifest
    entry lacking a sha256, staged content that does not hash to it, a
    traversal/absolute path, or curation drift from the config all abort the
    freeze — a snapshot must never bless bytes that were not verifiably fetched.
    Returns the written snapshot manifest dict.
    """
    # verified_companion_texts applies is_denied_path, requires sha256, verifies
    # every staged file's hash, and checks curation drift — the fail-closed
    # contract we must not weaken by re-implementing it here.
    from chunk_docs import verified_companion_texts

    upstream, entries = verified_companion_texts(
        companion_manifest, companion_staging, config_path=companion_config
    )

    # Rebuild out_dir from scratch: a re-freeze that left stale files behind
    # would fail verify_snapshot's own unrecorded-file check.
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
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

    def _write(text, rel_dest):
        dest = os.path.join(out_dir, rel_dest)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(text)
        records.append({
            "path": rel_dest,
            "sha256": sha256_hex(text),
            "bytes": os.path.getsize(dest),
        })

    for filename in sorted(os.listdir(official_dir)):
        src = os.path.join(official_dir, filename)
        if os.path.isfile(src):
            _copy(src, f"official/{filename}")

    # Write the exact verified bytes (not a re-read of staging) so the snapshot
    # captures what the gate blessed, with no window for the file to change.
    for entry, staged_text in entries:
        _write(staged_text, f"companion/{entry['path']}")

    _copy(companion_manifest, "companion_manifest.json")
    _copy(companion_config, "companion_sources.json")

    snapshot_manifest = {
        "schema": SNAPSHOT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "upstream": {
            "repo": upstream.get("repo", ""),
            "commit": upstream.get("commit", ""),
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


def apply_snapshot_to_args(args, override_specs):
    """Resolve ``args.snapshot`` into the four input args in place (shared by
    both chunking CLIs so the mutual-exclusion, verify, and assignment logic
    lives once).

    ``override_specs`` is an iterable of ``(flag_name, current_value,
    default_value)``; a value differing from its default means the operator
    also passed an explicit input flag, which is mutually exclusive with
    ``--snapshot``. On a conflict or snapshot-verification failure this prints a
    ``FAILED:`` line and exits non-zero.
    """
    overridden = [flag for flag, value, default in override_specs
                  if value != default]
    if overridden:
        print(f"FAILED: --snapshot is mutually exclusive with {overridden}")
        sys.exit(1)
    try:
        inputs = resolve_snapshot_inputs(args.snapshot)
    except ValueError as e:
        print(f"FAILED: snapshot verification: {e}")
        sys.exit(1)
    args.input = inputs["input"]
    args.companion_input = inputs["companion_input"]
    args.companion_manifest = inputs["companion_manifest"]
    args.companion_config = inputs["companion_config"]


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
