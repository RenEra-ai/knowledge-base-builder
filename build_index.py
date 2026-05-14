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

# chromadb / sentence-transformers are imported lazily inside main() — see note
# there — so the pure helpers in this module stay importable without the heavy
# ML stack installed.

DEFAULT_INPUT = "chunks.jsonl"
DEFAULT_OUTPUT = "boomi_knowledge_db/"
DEFAULT_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 200
COLLECTION_NAME = "boomi_docs"
MANIFEST_SCHEMA_VERSION = "1"
BUILDER_VERSION = "0.1.0"
VERIFY_MAX_DISTANCE = 0.45

# Fields every chunk must carry once chunk_docs.py has run.
REQUIRED_CHUNK_FIELDS = {
    "id", "content", "title", "section_heading", "breadcrumb",
    "source_url", "page_key", "chunk_index", "category", "token_estimate",
}

VERIFY_QUERIES = [
    "How do environment extensions work?",
    "Configure a database connector",
    "Deploy a process to production",
    "Trading partner EDI setup",
    "Groovy scripting in data process shape",
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

    by_page = {}
    for chunk in chunks:
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


def load_chunks(jsonl_path):
    """Load all chunks from a JSONL file."""
    chunks = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def build_index(chunks, collection, batch_size, verbose):
    """Add all chunks to the ChromaDB collection in batches."""
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["content"] for c in batch],
            metadatas=[{
                "title": c["title"],
                "section_heading": c["section_heading"],
                "breadcrumb": c["breadcrumb"],
                "source_url": c["source_url"],
                "page_key": c["page_key"],
                "chunk_index": c["chunk_index"],
                "category": c["category"],
                "token_estimate": c["token_estimate"],
            } for c in batch],
        )
        indexed = min(i + batch_size, total)
        print(f"  Indexed {indexed}/{total} chunks")

        if verbose:
            print(f"    Last batch: {batch[-1]['id']}")


def verify_index(collection, max_distance=VERIFY_MAX_DISTANCE):
    """Run smoke queries and return True only if the index is release-quality.

    The release gate: every VERIFY_QUERIES entry must return at least one hit
    with cosine distance at or below max_distance. Returns False (and prints the
    offending queries) otherwise so callers can fail the build.
    """
    print("\nVerification — test queries:")
    failures = []
    for query in VERIFY_QUERIES:
        results = collection.query(query_texts=[query], n_results=3)
        ids = results["ids"][0]
        distances = results["distances"][0]
        print(f"\n  Q: {query}")
        if not ids:
            print("    (no results)")
            failures.append((query, None))
            continue
        for i, (_doc_id, distance) in enumerate(zip(ids, distances)):
            meta = results["metadatas"][0][i]
            print(f"    {i+1}. [{distance:.3f}] {meta['title']} > {meta['section_heading']}")
        if min(distances) > max_distance:
            failures.append((query, min(distances)))

    if failures:
        print(f"\nFAILED: {len(failures)} verify query(ies) above distance {max_distance}:")
        for query, best in failures:
            shown = "no results" if best is None else f"best distance {best:.3f}"
            print(f"  - {query!r}: {shown}")
        return False

    print(f"\nAll {len(VERIFY_QUERIES)} verify queries passed (distance <= {max_distance}).")
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

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "collection_name": COLLECTION_NAME,
        "embedding_model": args.model,
        "embedding_dim": embedding_dim,
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "chunk_count": len(chunks),
        "page_count": len(page_keys),
        "category_counts": category_counts,
        "source_roots": source_roots,
        "artifact_tag": args.artifact_tag,
        "builder_commit": args.builder_commit,
        "builder_version": BUILDER_VERSION,
    }


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

    # Duplicate ids are the most fundamental corpus error — check them before
    # the per-chunk field/contiguity validation so the clearest message wins.
    duplicate_ids = sorted(
        chunk_id for chunk_id, count in Counter(c["id"] for c in chunks).items()
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

    # Heavy ML deps are imported only after the cheap input-validation gates
    # above have passed, so a bad JSONL fails fast without loading chromadb.
    import chromadb
    from chromadb.utils import embedding_functions

    print(f"\nInitializing embedding model: {args.model}")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=args.model
    )
    embedding_dim = get_embedding_dim(ef)

    print(f"Creating ChromaDB at {args.output}...")
    client = chromadb.PersistentClient(path=args.output)

    # Always rebuild from scratch
    try:
        client.delete_collection(COLLECTION_NAME)
        print("  Deleted existing collection")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"\nBuilding index (batch size {args.batch_size})...")
    start = time.time()
    build_index(chunks, collection, args.batch_size, args.verbose)
    elapsed = time.time() - start

    print_stats(len(chunks), args.model, args.output, elapsed, embedding_dim)

    manifest = build_manifest(chunks, args, embedding_dim)
    manifest_path = os.path.join(args.output, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest: {manifest_path}")

    if args.verify:
        if not verify_index(collection, args.verify_threshold):
            sys.exit(1)


if __name__ == "__main__":
    main()
