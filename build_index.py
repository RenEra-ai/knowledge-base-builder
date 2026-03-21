#!/usr/bin/env python3
"""
Build a ChromaDB semantic search index from chunked Boomi documentation.

Usage:
    python build_index.py [--input chunks.jsonl] [--output boomi_knowledge_db/]
"""

import argparse
import json
import os
import time

import chromadb
from chromadb.utils import embedding_functions

DEFAULT_INPUT = "chunks.jsonl"
DEFAULT_OUTPUT = "boomi_knowledge_db/"
DEFAULT_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 200

VERIFY_QUERIES = [
    "How do environment extensions work?",
    "Configure a database connector",
    "Deploy a process to production",
    "Trading partner EDI setup",
    "Groovy scripting in data process shape",
]


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
                "category": c["category"],
                "token_estimate": c["token_estimate"],
            } for c in batch],
        )
        indexed = min(i + batch_size, total)
        print(f"  Indexed {indexed}/{total} chunks")

        if verbose:
            print(f"    Last batch: {batch[-1]['id']}")


def verify_index(collection):
    """Run test queries to verify the index returns relevant results."""
    print("\nVerification — test queries:")
    for query in VERIFY_QUERIES:
        results = collection.query(query_texts=[query], n_results=3)
        print(f"\n  Q: {query}")
        for i, (_doc_id, distance) in enumerate(
            zip(results["ids"][0], results["distances"][0])
        ):
            meta = results["metadatas"][0][i]
            print(f"    {i+1}. [{distance:.3f}] {meta['title']} > {meta['section_heading']}")


def get_dir_size(path):
    """Calculate total size of a directory in bytes."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total


def print_stats(chunk_count, model_name, output_dir, elapsed):
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
    print(f"Index size:       {size_str}")
    print(f"Build time:       {elapsed:.1f} seconds")
    print(f"Output:           {output_dir}")
    print(f"{'=' * 60}")


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
        "--verbose", action="store_true", help="Print progress details"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"FAILED: Input file not found: {args.input}")
        return

    print(f"Loading chunks from {args.input}...")
    chunks = load_chunks(args.input)
    if not chunks:
        print("FAILED: No chunks found in input file")
        return
    print(f"  Loaded {len(chunks):,} chunks")

    print(f"\nInitializing embedding model: {args.model}")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=args.model
    )

    print(f"Creating ChromaDB at {args.output}...")
    client = chromadb.PersistentClient(path=args.output)

    # Always rebuild from scratch
    try:
        client.delete_collection("boomi_docs")
        print("  Deleted existing collection")
    except Exception:
        pass

    collection = client.create_collection(
        name="boomi_docs",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"\nBuilding index (batch size {args.batch_size})...")
    start = time.time()
    build_index(chunks, collection, args.batch_size, args.verbose)
    elapsed = time.time() - start

    print_stats(len(chunks), args.model, args.output, elapsed)

    if args.verify:
        verify_index(collection)


if __name__ == "__main__":
    main()
