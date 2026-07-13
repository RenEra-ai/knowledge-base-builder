#!/usr/bin/env python3
"""Retrieval evaluation harness for kb-25 candidates.

Runs every fixture query through the PINNED Track A ``KbService.search``
implementation (top_k=5, source_type unset) — the serving stack IS the
evaluation path, so gate numbers measure exactly what production retrieval
will do. The Track A checkout is located via ``BOOMI_MCP_SERVER_PATH``
(default: the sibling ``../boomi-mcp-server``), mirroring Track A's own
``KB_BUILDER_PATH`` convention.

Reproducibility: release comparisons run CPU-only with a single Torch/OMP/MKL
thread (``pin_determinism`` — call it before loading anything); two runs are
compared by ``compare_runs`` with HARD equality on ids, documents, metadata,
contracts, cache, semantic inputs, ranks, recall, and status, and a 0.0001
tolerance on numeric values (distances, MRR).
"""
import argparse
import json
import os
import sys

from eval_fixtures import (
    EXPOSED_REGRESSION,
    VISIBLE_CALIBRATION,
    group_hit,
)

DEFAULT_SERVER_PATH = os.path.join("..", "boomi-mcp-server")
DEFAULT_TOP_K = 5
NUMERIC_TOLERANCE = 0.0001


def pin_determinism():
    """Force CPU-only, single-thread inference for release comparisons.

    Must run BEFORE torch/sentence-transformers import anywhere in the
    process: the env vars only bind at library initialization.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


def load_kb_service(db_path, server_path=None):
    """Build a ready Track A KbService over ``db_path`` with default config.

    Uses the pinned serving implementation (cheap manifest validation,
    contract resolution, heavy build) — never a reimplementation — so ranks,
    diversification, and confidence thresholds are exactly production's.
    """
    server_path = (server_path or os.environ.get("BOOMI_MCP_SERVER_PATH")
                   or DEFAULT_SERVER_PATH)
    src = os.path.join(server_path, "src")
    if not os.path.isdir(src):
        raise FileNotFoundError(
            f"Track A checkout not found at {server_path!r} — set "
            "BOOMI_MCP_SERVER_PATH to the boomi-mcp-server repo"
        )
    if src not in sys.path:
        sys.path.insert(0, src)

    from boomi_mcp.kb.embedding_contract import resolve_embedding_contract
    from boomi_mcp.kb.manifest import load_manifest, validate_manifest
    from boomi_mcp.kb.service import KbBootstrap, build_kb_service_heavy

    config = {
        "db_path": db_path,
        "collection": "boomi_docs",
        "top_k_default": DEFAULT_TOP_K,
        "top_k_max": 10,
        "low_confidence_distance": 0.45,
    }
    manifest = load_manifest(db_path)
    validate_manifest(manifest, config["collection"])
    contract = resolve_embedding_contract(manifest)
    bootstrap = KbBootstrap(config=config, manifest=manifest, contract=contract)
    return build_kb_service_heavy(bootstrap)


def run_suite(service, queries=None, top_k=DEFAULT_TOP_K):
    """Run every query through ``service.search`` and score its target groups.

    Returns a JSON-serializable dict:
    ``{"top_k", "queries": {qid: {"query", "status", "rank1_source_type",
    "groups": [{"index", "identity", "rank", "distance", "hit"}]}}}``.
    A group's rank/distance come from the FIRST (best) hit satisfying it;
    ``rank`` is None on a miss. source_type stays unset on every call.
    """
    if queries is None:
        queries = VISIBLE_CALIBRATION + EXPOSED_REGRESSION

    results = {"top_k": top_k, "queries": {}}
    for query in queries:
        response = service.search(query.query, top_k)
        hits = response.get("hits", [])
        groups = []
        for index, group in enumerate(query.target_groups):
            rank = None
            distance = None
            for position, hit in enumerate(hits, start=1):
                if group_hit(group, hit.get("page_key", ""),
                             hit.get("content", "")):
                    rank = position
                    distance = hit.get("distance")
                    break
            page, markers = group.identity()
            groups.append({
                "index": index,
                "identity": [page, list(markers)],
                "rank": rank,
                "distance": distance,
                "hit": rank is not None,
            })
        results["queries"][query.query_id] = {
            "query": query.query,
            "status": response.get("status"),
            "rank1_source_type": (hits[0].get("source_type")
                                  if hits else None),
            "groups": groups,
        }
    return results


def evaluate(db_path, out_path, queries=None, server_path=None):
    """Load the corpus through Track A, run the suites, write results JSON."""
    service = load_kb_service(db_path, server_path=server_path)
    results = run_suite(service, queries)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    return results


# --- reproducibility comparison ---------------------------------------------------

def _load_bundle(directory):
    """A run bundle directory: results.json, manifest.json, semantic_inputs.jsonl
    (with trailing summary), optional s7_cache.jsonl."""
    bundle = {}
    with open(os.path.join(directory, "results.json"), encoding="utf-8") as f:
        bundle["results"] = json.load(f)
    with open(os.path.join(directory, "manifest.json"), encoding="utf-8") as f:
        bundle["manifest"] = json.load(f)
    rows = []
    with open(os.path.join(directory, "semantic_inputs.jsonl"),
              encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    bundle["semantic_summary"] = rows[-1]
    bundle["semantic_inputs"] = rows[:-1]
    cache_path = os.path.join(directory, "s7_cache.jsonl")
    if os.path.isfile(cache_path):
        from companion import sha256_hex

        with open(cache_path, "rb") as f:
            bundle["cache_sha256"] = sha256_hex(f.read())
    else:
        bundle["cache_sha256"] = None
    return bundle


def compare_runs(run_a, run_b, tolerance=NUMERIC_TOLERANCE):
    """Reproducibility hard gate between two run bundles.

    HARD equality on: chunk IDs, documents (raw text + embedding-input hash),
    canonical metadata, the embedding contract, the S7 cache hash,
    ``semantic_inputs_sha256``, per-group ranks, group recall (hit booleans),
    and response status. Distances compare within ``tolerance``. Returns a
    GateReport whose failures name the failing class.
    """
    from eval_gates import GateReport

    failures = []

    rows_a = {row["id"]: row for row in run_a["semantic_inputs"]}
    rows_b = {row["id"]: row for row in run_b["semantic_inputs"]}
    if sorted(rows_a) != sorted(rows_b):
        failures.append("ids: chunk id sets differ")
    else:
        for chunk_id, row in rows_a.items():
            other = rows_b[chunk_id]
            if row["raw_document"] != other["raw_document"]:
                failures.append(f"documents: raw_document differs for {chunk_id}")
                break
        for chunk_id, row in rows_a.items():
            if (row["embedding_input_sha256"]
                    != rows_b[chunk_id]["embedding_input_sha256"]):
                failures.append(
                    f"documents: embedding input differs for {chunk_id}"
                )
                break
        for chunk_id, row in rows_a.items():
            if row["canonical_metadata"] != rows_b[chunk_id]["canonical_metadata"]:
                failures.append(f"metadata: differs for {chunk_id}")
                break

    if (run_a["manifest"].get("embedding_contract")
            != run_b["manifest"].get("embedding_contract")):
        failures.append("contracts: embedding_contract differs")
    if run_a["cache_sha256"] != run_b["cache_sha256"]:
        failures.append("cache: S7 cache hash differs")
    if (run_a["semantic_summary"]["semantic_inputs_sha256"]
            != run_b["semantic_summary"]["semantic_inputs_sha256"]):
        failures.append("semantic_inputs: semantic_inputs_sha256 differs")

    queries_a = run_a["results"]["queries"]
    queries_b = run_b["results"]["queries"]
    if sorted(queries_a) != sorted(queries_b):
        failures.append("results: query id sets differ")
    else:
        for qid in sorted(queries_a):
            qa, qb = queries_a[qid], queries_b[qid]
            if qa["status"] != qb["status"]:
                failures.append(f"status: differs for {qid}")
            if len(qa["groups"]) != len(qb["groups"]):
                # zip() would silently ignore the extra groups otherwise, so a
                # run bundle with fewer target groups must fail the hard gate.
                failures.append(
                    f"results: {qid} has {len(qa['groups'])} vs "
                    f"{len(qb['groups'])} target groups"
                )
                continue
            for ga, gb in zip(qa["groups"], qb["groups"]):
                if ga["rank"] != gb["rank"]:
                    failures.append(f"ranks: differ for {qid} group {ga['index']}")
                if ga["hit"] != gb["hit"]:
                    failures.append(f"recall: differs for {qid} group {ga['index']}")
                da, db = ga["distance"], gb["distance"]
                if (da is None) != (db is None):
                    failures.append(
                        f"distances: presence differs for {qid} group {ga['index']}"
                    )
                elif da is not None and abs(da - db) > tolerance:
                    failures.append(
                        f"distances: |{da} - {db}| > {tolerance} for {qid} "
                        f"group {ga['index']}"
                    )

    return GateReport(passed=not failures, failures=failures, advisories=[])


def main():
    parser = argparse.ArgumentParser(description="kb-25 retrieval evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Evaluate a candidate corpus")
    run.add_argument("--db-path", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--server-path", default=None)

    compare = sub.add_parser("compare", help="Compare two run bundles")
    compare.add_argument("--a", required=True)
    compare.add_argument("--b", required=True)

    args = parser.parse_args()
    if args.command == "run":
        pin_determinism()
        results = evaluate(args.db_path, args.out, server_path=args.server_path)
        hit_counts = sum(
            g["hit"] for q in results["queries"].values() for g in q["groups"]
        )
        print(f"evaluated {len(results['queries'])} queries; "
              f"{hit_counts} group hits -> {args.out}")
        return 0

    report = compare_runs(_load_bundle(args.a), _load_bundle(args.b))
    if report.passed:
        print("runs match (within tolerance)")
        return 0
    print(f"FAILED: {len(report.failures)} reproducibility failure(s):")
    for failure in report.failures:
        print(f"  - {failure}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
