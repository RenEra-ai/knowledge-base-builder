#!/usr/bin/env python3
"""
Build a ChromaDB semantic search index from chunked Boomi documentation.

Usage:
    python build_index.py [--input chunks.jsonl] [--output boomi_knowledge_db/]
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import urlparse

# Provenance label constants live in companion.py (pure stdlib) — the single
# source of truth shared with chunk_docs.py, so the official/companion labels and
# the where-filter never drift between producer and index.
from companion import (
    COMPANION_SOURCE_TYPE,
    OFFICIAL_SOURCE_TYPE,
    OFFICIAL_VERIFICATION_STATUS,
    contains_raw_component_xml,
)

# chromadb / sentence-transformers are imported lazily inside main() — see note
# there — so the pure helpers in this module stay importable without the heavy
# ML stack installed.

DEFAULT_INPUT = "chunks.jsonl"
DEFAULT_OUTPUT = "boomi_knowledge_db/"
DEFAULT_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 200
COLLECTION_NAME = "boomi_docs"
MANIFEST_SCHEMA_VERSION = "1"
MANIFEST_SCHEMA_VERSION_V2 = "2"
BUILDER_VERSION = "0.1.0"
VERIFY_MAX_DISTANCE = 0.45

# kb-25 candidate builds. c0 is the legacy path (implicit Chroma embeddings,
# schema-v1 manifest); b5/b56/b567 consume S5 chunk records and pass EXPLICIT
# pinned-revision embeddings. Only the release candidates (b56/b567) emit a
# schema-v2 manifest with the embedding contract; b5 is eval-only and stays
# legacy-shaped (its headerless raw-fragment text IS the raw-v1 contract the
# serving side maps schema-v1 manifests to).
CANDIDATES = ("c0", "b5", "b56", "b567")
EMBEDDING_TEXT_VERSIONS = {"b56": "s5-s6-v1", "b567": "s5-s6-s7-v1"}

# The S5 chunk fields (beyond the legacy 15-field contract). Presence of
# source_record_id marks an S5 chunk; then the whole set is required.
S5_REQUIRED_FIELDS = (
    "source_token_count", "content_token_count", "embedding_token_count",
    "source_record_id", "source_start_byte", "source_end_byte",
    "synthetic_wrapper_metadata", "s6_header",
)

# Fields every chunk must carry once chunk_docs.py has run. source_type and
# verification_status are the load-bearing provenance labels (always populated,
# "official" for docs and "companion_reference"/"companion_unverified" for the
# supplemental corpus). The repo/path/url provenance fields are optional and
# read with .get(..., "") because they are only populated for companion chunks.
REQUIRED_CHUNK_FIELDS = {
    "id", "content", "title", "section_heading", "breadcrumb",
    "source_url", "page_key", "chunk_index", "category", "token_estimate",
    "source_type", "verification_status",
}

# Optional per-chunk provenance metadata persisted to Chroma (empty for official
# docs). Kept in sync with companion.PROVENANCE_FIELDS.
PROVENANCE_METADATA_FIELDS = (
    "source_type", "verification_status", "upstream_repo", "upstream_commit",
    "source_path", "raw_url", "latest_url",
)

# The five extended provenance fields that companion chunks must populate with
# real values and official chunks must leave blank.
PROVENANCE_REAL_VALUE_FIELDS = (
    "upstream_repo", "upstream_commit", "source_path", "raw_url", "latest_url",
)

VERIFY_QUERIES = [
    "How do environment extensions work?",
    "Configure a database connector",
    "Deploy a process to production",
    "Trading partner EDI setup",
    "Groovy scripting in data process shape",
]

# Companion smoke queries — each must retrieve a companion_reference-typed hit
# within the distance threshold. Only run when the corpus actually contains
# companion chunks, so a companion-less build never fails the gate.
COMPANION_VERIFY_QUERIES = [
    "REST connector operation dynamic properties runtime binding",
    "Groovy data process step dataContext storeStream DDP",
    "MCP Server connector SSE transport tool schema",
    "map isMappable tagList instance identifier",
]


def validate_chunks(chunks):
    """Validate every chunk against the producer contract.

    Checks the plan's required invariants for *all* chunks, not just the first:
    required fields present; non-empty string id/page_key/content; integer
    chunk_index; and contiguous 0-based chunk_index within each page_key.
    Returns a list of human-readable error strings (empty == release-quality).
    """
    errors = []
    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            errors.append(f"chunk {i} is not a JSON object: {type(chunk).__name__}")
            continue
        missing = REQUIRED_CHUNK_FIELDS - set(chunk)
        if missing:
            errors.append(f"chunk {i} missing required field(s): {sorted(missing)}")
            continue
        for field in ("id", "page_key", "content"):
            value = chunk[field]
            if not isinstance(value, str) or not value.strip():
                errors.append(f"chunk {i} ({chunk.get('id')!r}) has empty/non-string {field!r}")
        idx = chunk["chunk_index"]
        if not isinstance(idx, int) or isinstance(idx, bool):
            errors.append(f"chunk {i} ({chunk.get('id')!r}) has non-integer chunk_index: {idx!r}")

        # Provenance contract: companion chunks must carry real attribution;
        # official chunks must leave the five extended fields blank. Only these
        # two source_types are validated, so official-only corpora and any future
        # type stay backward-compatible.
        stype = chunk.get("source_type")
        if stype == COMPANION_SOURCE_TYPE:
            for field in PROVENANCE_REAL_VALUE_FIELDS:
                value = chunk.get(field)
                if not isinstance(value, str) or not value.strip():
                    errors.append(
                        f"chunk {i} ({chunk.get('id')!r}) companion chunk has empty {field!r}"
                    )
            # The corpus never ingests canned <bns:Component> component
            # templates. A raw one here means an allowlist entry lost its section
            # drop, or a skeleton sits under the size-based strip threshold. Fail
            # the build rather than ship it. (This gate is scoped to component
            # serializations, not all low-level XML — small illustrative XML is
            # permitted short context; see companion.contains_raw_component_xml.)
            if contains_raw_component_xml(chunk.get("content")):
                errors.append(
                    f"chunk {i} ({chunk.get('id')!r}) companion chunk contains raw "
                    f"<bns:Component> XML (source_path={chunk.get('source_path')!r}, "
                    f"section={chunk.get('section_heading')!r})"
                )
        elif stype == OFFICIAL_SOURCE_TYPE:
            for field in PROVENANCE_REAL_VALUE_FIELDS:
                # Must be exactly the empty string (missing key -> "" is fine).
                # A present None/0 is falsey but would reach the Chroma metadata
                # write as a non-string, so reject it here instead of passing it.
                if chunk.get(field, "") != "":
                    errors.append(
                        f"chunk {i} ({chunk.get('id')!r}) official chunk must leave {field!r} blank"
                    )

        # S5 chunk records (marked by source_record_id) must carry the whole
        # S5 field set with valid types — a partial record would embed with
        # wrong budgets or break span-based holdout matching downstream.
        if "source_record_id" in chunk:
            errors.extend(
                f"chunk {i} ({chunk.get('id')!r}) S5 chunk: {problem}"
                for problem in _s5_field_errors(chunk)
            )

    by_page = {}
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        page_key = chunk.get("page_key")
        idx = chunk.get("chunk_index")
        if isinstance(page_key, str) and page_key and isinstance(idx, int) and not isinstance(idx, bool):
            by_page.setdefault(page_key, []).append(idx)
    for page_key, indices in sorted(by_page.items()):
        if sorted(indices) != list(range(len(indices))):
            errors.append(
                f"page_key {page_key!r} has non-contiguous chunk_index: {sorted(indices)}"
            )

    return errors


def _s5_field_errors(chunk):
    """Type/consistency problems in one S5 chunk's added fields."""
    problems = []
    missing = [f for f in S5_REQUIRED_FIELDS if f not in chunk]
    if missing:
        return [f"missing S5 field(s): {missing}"]

    def _is_count(value):
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    for field in ("source_token_count", "content_token_count",
                  "embedding_token_count", "source_start_byte", "source_end_byte"):
        if not _is_count(chunk[field]):
            problems.append(f"{field!r} must be a non-negative int, got {chunk[field]!r}")
    if (_is_count(chunk["source_start_byte"]) and _is_count(chunk["source_end_byte"])
            and chunk["source_end_byte"] < chunk["source_start_byte"]):
        problems.append(
            f"source_end_byte {chunk['source_end_byte']} precedes "
            f"source_start_byte {chunk['source_start_byte']}"
        )
    if not isinstance(chunk["source_record_id"], str) or not chunk["source_record_id"]:
        problems.append("'source_record_id' must be a non-empty string")
    if not isinstance(chunk["synthetic_wrapper_metadata"], dict):
        problems.append("'synthetic_wrapper_metadata' must be an object")
    if not isinstance(chunk["s6_header"], str):
        problems.append("'s6_header' must be a string (empty is legal)")
    return problems


def load_chunks(jsonl_path):
    """Load all chunks from a JSONL file."""
    chunks = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def chunk_metadata(chunk):
    """The per-chunk Chroma metadata (scalars only).

    Legacy chunks keep the frozen 15-key shape byte-for-byte. S5 chunks add
    their token counts, source-record span, JSON-encoded wrapper metadata, and
    header text (Chroma metadata must be scalars, hence the JSON string).
    """
    meta = {
        "title": chunk["title"],
        "section_heading": chunk["section_heading"],
        "breadcrumb": chunk["breadcrumb"],
        "source_url": chunk["source_url"],
        "page_key": chunk["page_key"],
        "chunk_index": chunk["chunk_index"],
        "category": chunk["category"],
        "token_estimate": chunk["token_estimate"],
        # Provenance (empty strings for official docs). Chroma metadata
        # must be scalars, so default missing values to "" not None.
        "source_type": chunk.get("source_type", OFFICIAL_SOURCE_TYPE),
        "verification_status": chunk.get("verification_status", OFFICIAL_VERIFICATION_STATUS),
        "upstream_repo": chunk.get("upstream_repo", ""),
        "upstream_commit": chunk.get("upstream_commit", ""),
        "source_path": chunk.get("source_path", ""),
        "raw_url": chunk.get("raw_url", ""),
        "latest_url": chunk.get("latest_url", ""),
    }
    if "source_record_id" in chunk:
        meta.update({
            "source_token_count": chunk["source_token_count"],
            "content_token_count": chunk["content_token_count"],
            "embedding_token_count": chunk["embedding_token_count"],
            "source_record_id": chunk["source_record_id"],
            "source_start_byte": chunk["source_start_byte"],
            "source_end_byte": chunk["source_end_byte"],
            "synthetic_wrapper_metadata": json.dumps(
                chunk["synthetic_wrapper_metadata"], ensure_ascii=False,
                sort_keys=True,
            ),
            "s6_header": chunk["s6_header"],
        })
    return meta


def build_index(chunks, collection, batch_size, verbose):
    """Add all chunks to the ChromaDB collection in batches."""
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["content"] for c in batch],
            metadatas=[chunk_metadata(c) for c in batch],
        )
        indexed = min(i + batch_size, total)
        print(f"  Indexed {indexed}/{total} chunks")

        if verbose:
            print(f"    Last batch: {batch[-1]['id']}")


# --- candidate embedding inputs (B5/B56/B567) ----------------------------------

def b56_embedding_input(chunk):
    """B56 embedding text: deduplicated S6 header + raw document (header may
    be legally empty after deduplication, then the raw document embeds alone)."""
    header = chunk.get("s6_header", "")
    return header + "\n" + chunk["content"] if header else chunk["content"]


def embedding_input_for(chunk, candidate, tokenizer=None, s7=None):
    """The (embedding_text, embedding_source) one chunk contributes.

    b567 semantics (intent plan): official chunks and non-eligible companion
    chunks always embed the B56 input; an eligible companion chunk embeds its
    validated description (status ok) or the EXACT B56 input (raw_fallback) —
    so any B567-vs-B56 regression is attributable to successful description
    vectors, never to fallback behavior. ``s7`` carries {"cache", "contract",
    "identifiers_fn"} for the cache-key lookup.
    """
    if candidate in ("c0", "b5"):
        return chunk["content"], "raw"
    if candidate == "b56":
        return b56_embedding_input(chunk), "raw"
    if candidate != "b567":
        raise ValueError(f"unknown candidate: {candidate!r}")

    if chunk.get("source_type") != COMPANION_SOURCE_TYPE:
        return b56_embedding_input(chunk), "raw"

    from s7_eligibility import is_eligible

    if not is_eligible(chunk, tokenizer):
        return b56_embedding_input(chunk), "raw"

    from s7_cache import chunk_cache_key

    key = chunk_cache_key(chunk, s7["contract"], s7["identifiers_fn"])
    record = s7["cache"].get(key)
    if record is None:
        raise ValueError(
            f"S7 cache entry missing for eligible chunk {chunk['id']!r} "
            f"(key {key}) — release builds never call an LLM; generate the "
            "cache first"
        )
    if record["status"] == "ok":
        return record["description"], "description"
    return b56_embedding_input(chunk), "raw_fallback"


def assert_embedding_inputs_within_budget(plan, tokenizer):
    """Hard rule: every final embedding input fits the model window
    (model-side truncation is forbidden). Raises naming every offender."""
    from kb_tokenizer import EFFECTIVE_BUDGET

    offenders = []
    for item in plan:
        count = tokenizer.count(item["embedding_input"])
        if count > EFFECTIVE_BUDGET:
            offenders.append(f"{item['id']} ({count} WordPieces)")
    if offenders:
        raise ValueError(
            f"{len(offenders)} embedding input(s) exceed the "
            f"{EFFECTIVE_BUDGET}-WordPiece budget: {offenders[:5]}"
        )


def compute_embedding_plan(chunks, candidate, tokenizer, s7=None):
    """Per-chunk embedding plan: id, stored document (ALWAYS the raw document),
    embedding input, embedding source, and the Chroma metadata (annotated with
    the source, plus description/contract hash where a description is used).
    Asserts every input fits the token budget."""
    plan = []
    contract_sha = None
    if s7 is not None:
        from s7_cache import generator_contract_sha256

        contract_sha = generator_contract_sha256(s7["contract"])
    for chunk in chunks:
        text, source = embedding_input_for(chunk, candidate, tokenizer=tokenizer,
                                           s7=s7)
        meta = chunk_metadata(chunk)
        meta["embedding_source"] = source
        if source == "description":
            meta["description"] = text
        if contract_sha and chunk.get("source_type") == COMPANION_SOURCE_TYPE:
            meta["generator_contract_sha256"] = contract_sha
        plan.append({
            "id": chunk["id"],
            "document": chunk["content"],
            "embedding_input": text,
            "embedding_source": source,
            "metadata": meta,
        })
    assert_embedding_inputs_within_budget(plan, tokenizer)
    return plan


def verify_index(collection, max_distance=VERIFY_MAX_DISTANCE, queries=None):
    """Run smoke queries and return True only if the index is release-quality.

    ``queries`` is a list of ``(query_text, expected_source_type)`` pairs. When
    ``expected_source_type`` is set the query is metadata-filtered to that
    source_type, so each gate verifies *its own* corpus independently: the
    official gate cannot be satisfied by a companion hit, and the companion gate
    cannot false-fail because official docs out-rank companion content in the
    top-k. ``None`` accepts any hit. Defaults to the official VERIFY_QUERIES. The
    gate: every query must return a hit within ``max_distance``. Returns False
    (and prints the offending queries) otherwise so callers can fail the build.
    """
    if queries is None:
        queries = [(q, None) for q in VERIFY_QUERIES]

    print("\nVerification — test queries:")
    failures = []
    for query, expected_type in queries:
        where = {"source_type": expected_type} if expected_type else None
        results = collection.query(query_texts=[query], n_results=3, where=where)
        ids = results["ids"][0]
        distances = results["distances"][0]
        metas = results["metadatas"][0]
        label = f"{query!r} [{expected_type}]" if expected_type else repr(query)
        print(f"\n  Q: {label}")
        if not ids:
            print("    (no results)")
            failures.append((query, "no results"))
            continue
        for rank, (distance, meta) in enumerate(zip(distances, metas), start=1):
            stype = meta.get("source_type", OFFICIAL_SOURCE_TYPE)
            print(f"    {rank}. [{distance:.3f}] ({stype}) {meta['title']} > {meta['section_heading']}")

        # The where-filter already restricts hits to the expected source_type, so
        # a within-threshold hit is sufficient.
        best = min(distances)
        if best > max_distance:
            failures.append((query, f"best distance {best:.3f}"))

    if failures:
        print(f"\nFAILED: {len(failures)} verify query(ies) below quality bar:")
        for query, reason in failures:
            print(f"  - {query!r}: {reason}")
        return False

    print(f"\nAll {len(queries)} verify queries passed (distance <= {max_distance}).")
    return True


def get_dir_size(path):
    """Calculate total size of a directory in bytes."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total


def print_stats(chunk_count, model_name, output_dir, elapsed, embedding_dim=None):
    """Print summary statistics for the index build."""
    size_bytes = get_dir_size(output_dir)
    if size_bytes >= 1024 * 1024:
        size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        size_str = f"{size_bytes / 1024:.1f} KB"

    print(f"\n{'=' * 60}")
    print("Index Build Complete")
    print(f"{'=' * 60}")
    print(f"Chunks indexed:   {chunk_count:,}")
    print(f"Embedding model:  {model_name}")
    if embedding_dim is not None:
        print(f"Embedding dim:    {embedding_dim}")
    print(f"Index size:       {size_str}")
    print(f"Build time:       {elapsed:.1f} seconds")
    print(f"Output:           {output_dir}")
    print(f"{'=' * 60}")


def get_embedding_dim(embedding_function):
    """Best-effort embedding dimension probe that does not depend on Chroma APIs."""
    try:
        sample_embeddings = embedding_function(["dimension probe"])
    except Exception as exc:
        print(f"WARNING: Could not determine embedding dimension: {exc}")
        return None

    if not sample_embeddings:
        return None

    return len(sample_embeddings[0])


def build_manifest(chunks, args, embedding_dim):
    """Assemble the manifest.json contract (schema v1) from the indexed chunks.

    Pure function (no I/O) so it is cheap to unit-test. main() writes the
    returned dict into the Chroma output directory.
    """
    category_counts = dict(Counter(c["category"] for c in chunks))
    page_keys = {c["page_key"] for c in chunks}

    source_roots = sorted({
        f"{parsed.scheme}://{parsed.netloc}"
        for parsed in (
            urlparse(c["source_url"]) for c in chunks if c.get("source_url")
        )
        if parsed.scheme and parsed.netloc
    })
    if not source_roots:
        source_roots = ["https://help.boomi.com"]

    source_type_counts = dict(Counter(c.get("source_type", OFFICIAL_SOURCE_TYPE) for c in chunks))

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "collection_name": COLLECTION_NAME,
        "embedding_model": args.model,
        "embedding_dim": embedding_dim,
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "chunk_count": len(chunks),
        "page_count": len(page_keys),
        "category_counts": category_counts,
        "source_type_counts": source_type_counts,
        "source_roots": source_roots,
        "artifact_tag": args.artifact_tag,
        "builder_commit": args.builder_commit,
        "builder_version": BUILDER_VERSION,
    }

    companion = build_companion_summary(chunks)
    if companion:
        manifest["companion"] = companion

    return manifest


def build_manifest_v2(chunks, args, embedding_dim, candidate,
                      source_snapshot_sha256, s7_hashes=None):
    """Schema-v2 manifest for the release candidates (b56/b567): the v1 base
    plus the explicit embedding contract, the frozen-source hash, and (b567)
    the S7 generation hashes."""
    if candidate not in EMBEDDING_TEXT_VERSIONS:
        raise ValueError(
            f"schema-v2 manifests exist only for {sorted(EMBEDDING_TEXT_VERSIONS)}, "
            f"not {candidate!r}"
        )
    if not source_snapshot_sha256:
        raise ValueError(
            "a v2 manifest requires source_snapshot_sha256 — candidate builds "
            "read inputs only from a verified frozen snapshot"
        )
    from kb_tokenizer import MAX_SEQ_TOKENS, MODEL_ID, MODEL_REVISION

    manifest = build_manifest(chunks, args, embedding_dim)
    manifest["schema_version"] = MANIFEST_SCHEMA_VERSION_V2
    # Candidate builds embed with the pinned MODEL_ID regardless of --model, so
    # the top-level embedding_model must report the model that actually ran —
    # the serving resolver requires embedding_model == embedding_contract.model_id.
    manifest["embedding_model"] = MODEL_ID
    manifest["embedding_contract"] = {
        "version": 1,
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "max_seq_length": MAX_SEQ_TOKENS,
        "distance_metric": "cosine",
        "normalize_embeddings": False,
        "embedding_text_version": EMBEDDING_TEXT_VERSIONS[candidate],
        "s7_enabled": candidate == "b567",
    }
    manifest["source_snapshot_sha256"] = source_snapshot_sha256
    if s7_hashes:
        manifest.update(s7_hashes)
    return manifest


def write_semantic_inputs(path, plan, combined_hash_inputs):
    """Write the reproducibility sidecar and return its summary record.

    One row per chunk ({id, raw_document, canonical_metadata,
    embedding_input_sha256}) plus a trailing summary carrying
    ``semantic_inputs_sha256`` — the hash of the sorted (id, input-hash) pairs
    — and the combined-hash inputs (snapshot hash, embedding contract,
    generator-contract/cache/usage-map hashes) release comparisons gate on.
    """
    from companion import sha256_hex
    from s7_cache import canonical_json

    pairs = sorted([item["id"], sha256_hex(item["embedding_input"])]
                   for item in plan)
    summary = {
        "semantic_inputs_sha256": sha256_hex(canonical_json(pairs)),
        "combined_hash_inputs": combined_hash_inputs,
        "row_count": len(plan),
    }
    with open(path, "w", encoding="utf-8") as f:
        for item in plan:
            f.write(json.dumps({
                "id": item["id"],
                "raw_document": item["document"],
                "canonical_metadata": item["metadata"],
                "embedding_input_sha256": sha256_hex(item["embedding_input"]),
            }, ensure_ascii=False, sort_keys=True) + "\n")
        f.write(json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n")
    return summary


def build_companion_summary(chunks):
    """Summarize the companion (supplemental) chunks, or None if there are none.

    Kept as its own object so the corpus resource can render the supplemental
    source (repo + commit) distinctly from the official docs. schema_version
    stays "1" — this is an additive, optional field.
    """
    companion_chunks = [
        c for c in chunks if c.get("source_type") == COMPANION_SOURCE_TYPE
    ]
    if not companion_chunks:
        return None

    repos = sorted({c.get("upstream_repo", "") for c in companion_chunks if c.get("upstream_repo")})
    commits = sorted({c.get("upstream_commit", "") for c in companion_chunks if c.get("upstream_commit")})
    file_paths = {c.get("source_path", "") for c in companion_chunks if c.get("source_path")}
    # area = middle breadcrumb segment of "Companion Reference > <area> > <title>".
    # Require all three segments: a 2-part breadcrumb (empty area) would otherwise
    # count the document title as an area.
    area_counts = Counter(
        parts[1].strip()
        for parts in (c.get("breadcrumb", "").split(" > ") for c in companion_chunks)
        if len(parts) >= 3 and parts[1].strip()
    )

    return {
        "repo": ", ".join(repos),
        "commit": ", ".join(commits),
        "file_count": len(file_paths),
        "chunk_count": len(companion_chunks),
        "area_counts": dict(area_counts),
    }


def _load_eval_fixtures(path):
    """Load a fixtures-override JSON file into EvalQuery objects (test hook)."""
    from eval_fixtures import EvalQuery, TargetGroup

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        EvalQuery(
            query_id=q["query_id"],
            query=q.get("query", ""),
            target_groups=tuple(
                TargetGroup(g["page_match"], tuple(g.get("markers", ())))
                for g in q["groups"]
            ),
        )
        for q in data
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Build ChromaDB index from chunked docs"
    )
    parser.add_argument(
        "--input", default=DEFAULT_INPUT, help="Input JSONL chunks file"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT, help="Output ChromaDB directory"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Sentence transformer model name"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Batch size for indexing"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Run test queries after building"
    )
    parser.add_argument(
        "--verify-threshold", type=float, default=VERIFY_MAX_DISTANCE,
        help="Maximum accepted best-hit distance for --verify smoke queries",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print progress details"
    )
    parser.add_argument(
        "--artifact-tag", default="",
        help="GitHub release tag for this corpus (e.g. kb-42); recorded in manifest.json",
    )
    parser.add_argument(
        "--builder-commit", default="",
        help="Builder git commit sha; recorded in manifest.json",
    )
    parser.add_argument(
        "--candidate", choices=CANDIDATES, default="c0",
        help="kb-25 candidate: c0 = legacy implicit embeddings (default, "
             "byte-compatible); b5/b56/b567 = S5 chunks with explicit "
             "pinned-revision embeddings",
    )
    parser.add_argument(
        "--source-snapshot-sha256", default="",
        help="source_snapshot_sha256 from snapshot.py verify; required for "
             "b56/b567 (recorded in the schema-v2 manifest)",
    )
    parser.add_argument(
        "--s7-cache", default=None,
        help="S7 description cache JSONL (required for b567)",
    )
    parser.add_argument(
        "--s7-contract", default=None,
        help="Generator-contract JSON file (required for b567)",
    )
    parser.add_argument(
        "--semantic-inputs-out", default=None,
        help="Write the reproducibility semantic-inputs sidecar JSONL here",
    )
    parser.add_argument(
        "--eval-fixtures", default=None,
        help="JSON file overriding the built-in eval fixtures for the "
             "pre-embedding marker-integrity gate (testing hook; the default "
             "is the full visible calibration + exposed regression suites)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"FAILED: Input file not found: {args.input}")
        sys.exit(1)

    print(f"Loading chunks from {args.input}...")
    chunks = load_chunks(args.input)
    if not chunks:
        print("FAILED: No chunks found in input file")
        sys.exit(1)
    print(f"  Loaded {len(chunks):,} chunks")

    # Duplicate ids are the most fundamental corpus error — check them first so
    # the clearest message wins. Only count well-formed string ids: missing-id
    # and non-object rows are left for validate_chunks() to report cleanly
    # rather than raising a traceback here.
    string_ids = [
        c["id"] for c in chunks
        if isinstance(c, dict) and isinstance(c.get("id"), str)
    ]
    duplicate_ids = sorted(
        chunk_id for chunk_id, count in Counter(string_ids).items()
        if count > 1
    )
    if duplicate_ids:
        print(f"FAILED: {len(duplicate_ids)} duplicate chunk id(s) in {args.input}, "
              f"e.g. {duplicate_ids[:5]}")
        sys.exit(1)

    chunk_errors = validate_chunks(chunks)
    if chunk_errors:
        print(f"FAILED: {len(chunk_errors)} chunk validation error(s) in {args.input}:")
        for err in chunk_errors[:10]:
            print(f"  - {err}")
        if len(chunk_errors) > 10:
            print(f"  ... and {len(chunk_errors) - 10} more")
        print("Re-run chunk_docs.py.")
        sys.exit(1)

    candidate = args.candidate
    s7 = None
    if candidate != "c0":
        # Candidate builds require S5 chunk records throughout.
        non_s5 = [c["id"] for c in chunks if "source_record_id" not in c]
        if non_s5:
            print(f"FAILED: candidate {candidate!r} requires S5 chunk records; "
                  f"{len(non_s5)} legacy chunk(s), e.g. {non_s5[:3]}. "
                  "Re-run s5_pipeline.py.")
            sys.exit(1)
        if candidate in EMBEDDING_TEXT_VERSIONS and not args.source_snapshot_sha256:
            print("FAILED: --source-snapshot-sha256 is required for "
                  f"candidate {candidate!r} (snapshot.py verify prints it).")
            sys.exit(1)
        if candidate == "b567":
            if not args.s7_cache or not args.s7_contract:
                print("FAILED: candidate b567 requires --s7-cache and "
                      "--s7-contract (release builds never call an LLM).")
                sys.exit(1)

        # Visible fixture integrity: after S5, before embedding. Report-only —
        # never changes split placement; a failure stops the build.
        from eval_fixtures import check_fixture_integrity

        queries = _load_eval_fixtures(args.eval_fixtures) if args.eval_fixtures else None
        integrity_errors = check_fixture_integrity(chunks, queries)
        if integrity_errors:
            print(f"FAILED: {len(integrity_errors)} fixture integrity error(s):")
            for err in integrity_errors[:10]:
                print(f"  - {err}")
            sys.exit(1)

        if candidate == "b567":
            from s7_cache import S7Cache, enforce_release
            from s7_eligibility import chunk_identifiers, is_eligible
            from kb_tokenizer import HFWordPieceTokenizer

            with open(args.s7_contract, "r", encoding="utf-8") as f:
                contract = json.load(f)
            cache = S7Cache.load(args.s7_cache)
            s7 = {"cache": cache, "contract": contract,
                  "identifiers_fn": chunk_identifiers}
            tokenizer = HFWordPieceTokenizer()
            eligible = [c for c in chunks
                        if c.get("source_type") == COMPANION_SOURCE_TYPE
                        and is_eligible(c, tokenizer)]
            release_errors = enforce_release(
                cache, eligible, contract,
                identifiers_fn=chunk_identifiers,
            )
            if release_errors:
                print(f"FAILED: {len(release_errors)} S7 release error(s):")
                for err in release_errors[:10]:
                    print(f"  - {err}")
                sys.exit(1)

    # Heavy ML deps are imported only after the cheap input-validation gates
    # above have passed, so a bad JSONL fails fast without loading chromadb.
    import chromadb
    from chromadb.utils import embedding_functions

    if candidate == "c0":
        print(f"\nInitializing embedding model: {args.model}")
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=args.model
        )
        embedding_dim = get_embedding_dim(ef)
    else:
        # Explicit pinned embeddings: the exact model revision the manifest
        # contract declares, raw documents stored, vectors supplied by us. The
        # embedding function is constructed IDENTICALLY to the serving side
        # (model_name + revision + normalize) so the config Chroma persists in
        # the collection matches what get_collection is later handed — without
        # it Chroma would persist its default EF and serving would raise an
        # embedding-function conflict, and --verify would embed its smoke
        # queries with the wrong model. Reusing ef._model for encode also
        # avoids loading the pinned weights twice.
        from kb_tokenizer import (
            MAX_SEQ_TOKENS,
            MODEL_ID,
            MODEL_REVISION,
            HFWordPieceTokenizer,
        )

        print(f"\nInitializing pinned embedding model: {MODEL_ID}@{MODEL_REVISION[:7]}")
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=MODEL_ID, revision=MODEL_REVISION, normalize_embeddings=False,
        )
        model = ef._model
        if model.max_seq_length != MAX_SEQ_TOKENS:
            print(f"FAILED: loaded model max_seq_length {model.max_seq_length} "
                  f"!= contract {MAX_SEQ_TOKENS}")
            sys.exit(1)
        tokenizer = HFWordPieceTokenizer()
        plan = compute_embedding_plan(chunks, candidate, tokenizer, s7=s7)
        embedding_dim = model.get_sentence_embedding_dimension()

    print(f"Creating ChromaDB at {args.output}...")
    client = chromadb.PersistentClient(path=args.output)

    # Always rebuild from scratch
    try:
        client.delete_collection(COLLECTION_NAME)
        print("  Deleted existing collection")
    except Exception:
        pass

    # Both paths persist a SentenceTransformer EF so a reopened collection
    # binds the same embedder identity; candidate builds additionally supply
    # explicit pinned vectors on add().
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"\nBuilding index (batch size {args.batch_size})...")
    start = time.time()
    if candidate == "c0":
        build_index(chunks, collection, args.batch_size, args.verbose)
    else:
        total = len(plan)
        for i in range(0, total, args.batch_size):
            batch = plan[i:i + args.batch_size]
            vectors = model.encode(
                [item["embedding_input"] for item in batch],
                normalize_embeddings=False,
                batch_size=min(64, len(batch)),
            )
            collection.add(
                ids=[item["id"] for item in batch],
                documents=[item["document"] for item in batch],
                embeddings=[vector.tolist() for vector in vectors],
                metadatas=[item["metadata"] for item in batch],
            )
            print(f"  Indexed {min(i + args.batch_size, total)}/{total} chunks")
    elapsed = time.time() - start

    print_stats(len(chunks), args.model, args.output, elapsed, embedding_dim)

    s7_hashes = None
    if candidate == "b567":
        from companion import sha256_hex as _sha256_hex
        from s7_cache import (
            canonical_json,
            generator_contract_sha256,
            usage_map,
        )

        with open(args.s7_cache, "rb") as f:
            cache_bytes = f.read()
        mapping = usage_map(s7["contract"], chunks,
                            identifiers_fn=s7["identifiers_fn"])
        s7_hashes = {
            "generator_contract_sha256": generator_contract_sha256(s7["contract"]),
            "s7_cache_sha256": _sha256_hex(cache_bytes),
            "usage_map_sha256": _sha256_hex(canonical_json(mapping)),
        }

    if candidate in EMBEDDING_TEXT_VERSIONS:
        manifest = build_manifest_v2(
            chunks, args, embedding_dim, candidate,
            source_snapshot_sha256=args.source_snapshot_sha256,
            s7_hashes=s7_hashes,
        )
    else:
        manifest = build_manifest(chunks, args, embedding_dim)
    manifest_path = os.path.join(args.output, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest: {manifest_path}")

    if args.semantic_inputs_out and candidate != "c0":
        combined = {
            "source_snapshot_sha256": args.source_snapshot_sha256,
            "embedding_contract": manifest.get("embedding_contract"),
        }
        if s7_hashes:
            combined.update({
                "generator_contract_sha256": s7_hashes["generator_contract_sha256"],
                "s7_cache_sha256": s7_hashes["s7_cache_sha256"],
                "usage_map_sha256": s7_hashes["usage_map_sha256"],
            })
        summary = write_semantic_inputs(args.semantic_inputs_out, plan, combined)
        print(f"Wrote semantic inputs: {args.semantic_inputs_out} "
              f"(semantic_inputs_sha256 {summary['semantic_inputs_sha256'][:12]}…)")

    if args.verify:
        # Official queries verify official docs; companion queries verify
        # companion content — each metadata-filtered to its own source_type so
        # neither gate can be satisfied (or false-failed) by the other corpus.
        queries = [(q, OFFICIAL_SOURCE_TYPE) for q in VERIFY_QUERIES]
        if manifest.get("companion"):  # reuse the summary build_manifest computed
            queries += [(q, COMPANION_SOURCE_TYPE) for q in COMPANION_VERIFY_QUERIES]
        if not verify_index(collection, args.verify_threshold, queries):
            sys.exit(1)


if __name__ == "__main__":
    main()
