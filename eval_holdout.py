#!/usr/bin/env python3
"""Sealed release holdout machinery (intent plan: evaluation.sealed_holdout).

The sealed holdout is a small set of natural-language queries whose targets are
anchored to FROZEN source records — never to build-time marker text — so it can
score retrieval without constraining how the corpus is split. It is stored,
base64-encoded, as ``KB_RELEASE_HOLDOUT_B64`` in the protected release
environment; only its SHA-256 commitment and row count may appear in logs or
commits BEFORE the one-shot evaluation, and the query text is never logged.

Target matching resolves a target to the CHUNKS that carry its content by frozen
``source_record_id`` plus either an anchor byte-range overlap with the chunk's
``[source_start_byte, source_end_byte)`` span, or section identity (normalized
``section_heading``). ``run_holdout`` then scores a target as a hit when a top-5
``KbService.search`` hit's ``chunk_id`` is one of those chunks.
"""
import base64
import binascii
import json
import os

from companion import sha256_hex
from eval_gates import DISTANCE_GATE, GateReport

DEFAULT_TOP_K = 5

# Family taxonomy (intent plan: evaluation.sealed_holdout.target_taxonomies).
SDP_FAMILY = "set_document_property"
OFFICIAL_FAMILY = "official"
# The five Companion families (SDP first); official is scored separately.
COMPANION_FAMILIES = (
    SDP_FAMILY,
    "rest_operation_step",
    "groovy_stream",
    "edi_qualifier_map",
    "mcp_schema_cookie",
)
NON_SDP_COMPANION_FAMILIES = COMPANION_FAMILIES[1:]
HOLDOUT_FAMILIES = COMPANION_FAMILIES + (OFFICIAL_FAMILY,)

_ENV_VAR = "KB_RELEASE_HOLDOUT_B64"
_REQUIRED_ROW_FIELDS = ("query_id", "query", "family", "targets")

# Status ordering for the B56 no-regression gate (higher is better).
_STATUS_ORDER = {"no_match": 0, "low_confidence": 1, "ok": 2}


# --- decode / commitment --------------------------------------------------------

def _decode_payload(raw_b64):
    """Base64-decode the raw payload string to its UTF-8 JSONL text."""
    try:
        payload = base64.b64decode(raw_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{_ENV_VAR} is not valid base64: {exc}") from exc
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{_ENV_VAR} payload is not valid UTF-8") from exc


def _parse_rows(raw_b64):
    """Decode and parse the JSONL payload into a list of row dicts."""
    rows = []
    for lineno, line in enumerate(_decode_payload(raw_b64).splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rows.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise ValueError(f"holdout row {lineno} is not valid JSON: {exc}") \
                from exc
    return rows


def load_holdout(env=os.environ):
    """Decode ``KB_RELEASE_HOLDOUT_B64`` into validated holdout rows.

    Each row is ``{query_id, query, family, targets: [{source_record_id,
    anchor}]}`` where ``anchor`` is either ``{start_byte, end_byte}`` or
    ``{section_heading}``. Raises ``KeyError`` when the variable is unset and
    ``ValueError`` on a malformed payload or a row missing a required field.
    """
    if _ENV_VAR not in env:
        raise KeyError(f"{_ENV_VAR} is not set in the environment")
    rows = _parse_rows(env[_ENV_VAR])
    for index, row in enumerate(rows):
        _validate_row(row, index)
    return rows


def holdout_commitment(raw_b64):
    """The pre-evaluation commitment: the ONLY values that may be logged or
    committed before the holdout is run.

    ``sha256`` is over the RAW base64 payload string (UTF-8); ``count`` is the
    row count. Neither exposes any query text.
    """
    rows = _parse_rows(raw_b64)
    return {
        "sha256": sha256_hex(raw_b64),
        "count": len(rows),
    }


# --- validation -----------------------------------------------------------------

def _validate_row(row, index):
    if not isinstance(row, dict):
        raise ValueError(f"holdout row {index} must be a JSON object")
    for field in _REQUIRED_ROW_FIELDS:
        if field not in row:
            raise ValueError(
                f"holdout row {index} is missing required field {field!r}"
            )
    targets = row["targets"]
    if not isinstance(targets, list) or not targets:
        raise ValueError(f"holdout row {index} 'targets' must be a non-empty list")
    for t_index, target in enumerate(targets):
        _validate_target(target, f"holdout row {index} target {t_index}")


def _validate_target(target, where):
    if not isinstance(target, dict):
        raise ValueError(f"{where} must be a JSON object")
    if not target.get("source_record_id"):
        raise ValueError(f"{where} is missing 'source_record_id'")
    if "anchor" not in target:
        raise ValueError(f"{where} is missing 'anchor'")
    _anchor_kind(target["anchor"], where)


def _anchor_kind(anchor, where):
    """Classify (and validate) an anchor as ``"section"`` or ``"byte"``."""
    if not isinstance(anchor, dict):
        raise ValueError(f"{where} anchor must be a JSON object")
    if "section_heading" in anchor:
        if not str(anchor["section_heading"]).strip():
            raise ValueError(f"{where} anchor 'section_heading' must be non-empty")
        return "section"
    if "start_byte" in anchor and "end_byte" in anchor:
        for key in ("start_byte", "end_byte"):
            value = anchor[key]
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{where} anchor {key!r} must be an integer")
        if anchor["end_byte"] < anchor["start_byte"]:
            raise ValueError(
                f"{where} anchor end_byte {anchor['end_byte']} < start_byte "
                f"{anchor['start_byte']}"
            )
        return "byte"
    raise ValueError(
        f"{where} anchor must carry byte offsets or a 'section_heading'"
    )


# --- target matching ------------------------------------------------------------

def _normalize_heading(text):
    return " ".join(str(text).split()).lower()


def _byte_overlap(anchor, chunk):
    """Half-open overlap of the anchor with the chunk's [start, end) span."""
    c_start = chunk.get("source_start_byte")
    c_end = chunk.get("source_end_byte")
    if c_start is None or c_end is None:
        return False
    return anchor["start_byte"] < c_end and c_start < anchor["end_byte"]


def match_target(target, chunks):
    """Resolve a holdout target to the chunk records that carry its content.

    A chunk matches when its ``source_record_id`` equals the (frozen) target
    record AND, per the anchor kind, its byte span overlaps the anchor range or
    its normalized ``section_heading`` equals the anchor heading. Matching never
    inspects literal build-time marker text, so it does not constrain splitting.
    """
    source_record_id = target["source_record_id"]
    anchor = target["anchor"]
    kind = _anchor_kind(anchor, "target")
    matches = []
    for chunk in chunks:
        if chunk.get("source_record_id") != source_record_id:
            continue
        if kind == "byte":
            if _byte_overlap(anchor, chunk):
                matches.append(chunk)
        elif (_normalize_heading(chunk.get("section_heading", ""))
              == _normalize_heading(anchor["section_heading"])):
            matches.append(chunk)
    return matches


# --- run ------------------------------------------------------------------------

def run_holdout(service, holdout_rows, chunks, top_k=DEFAULT_TOP_K):
    """Score each holdout target against ``service.search`` hits.

    Returns a dict shaped like ``eval_harness.run_suite`` output, minus the query
    text: ``{"top_k", "queries": {query_id: {"family", "status",
    "rank1_source_type", "groups": [{"index", "rank", "distance", "hit"}]}}}``.
    A group hits when a top-``top_k`` search hit's ``chunk_id`` is one of the
    chunks the target resolves to. The query text is NEVER logged or returned;
    only the query_id and hit counts are printed.
    """
    results = {"top_k": top_k, "queries": {}}
    for row in holdout_rows:
        response = service.search(row["query"], top_k)
        hits = response.get("hits", [])
        groups = []
        for index, target in enumerate(row["targets"]):
            match_ids = {chunk["id"] for chunk in match_target(target, chunks)}
            rank = None
            distance = None
            for position, hit in enumerate(hits, start=1):
                if hit.get("chunk_id") in match_ids:
                    rank = position
                    distance = hit.get("distance")
                    break
            groups.append({"index": index, "rank": rank, "distance": distance,
                           "hit": rank is not None})
        results["queries"][row["query_id"]] = {
            "family": row["family"],
            "status": response.get("status"),
            "rank1_source_type": hits[0].get("source_type") if hits else None,
            "groups": groups,
        }
        hit_count = sum(group["hit"] for group in groups)
        print(f"holdout {row['query_id']}: {hit_count}/{len(groups)} target "
              f"groups in top {top_k}")
    return results


# --- gates ----------------------------------------------------------------------

def _entry_by_family(results, family):
    for entry in results["queries"].values():
        if entry.get("family") == family:
            return entry
    raise KeyError(f"holdout results have no query with family {family!r}")


def _best_group(entry):
    """The query's best group: lowest rank (misses last), then lowest distance."""
    def _key(group):
        return (group["rank"] is None,
                group["rank"] if group["rank"] is not None else 0,
                group["distance"] if group["distance"] is not None
                else float("inf"))

    return min(entry["groups"], key=_key)


def _rank_or_miss(group):
    return group["rank"] if group["rank"] is not None else 99


def _status_rank(status):
    return _STATUS_ORDER.get(status, 0)


def _official_holdout_failures(results):
    """The hard official gate: best target rank <= 3, distance <= 0.45, and the
    rank-1 hit is source_type=official."""
    failures = []
    official = _entry_by_family(results, OFFICIAL_FAMILY)
    best = _best_group(official)
    if best["rank"] is None or best["rank"] > 3:
        failures.append(f"holdout-official: target rank {best['rank']} exceeds 3")
    if best["distance"] is None or best["distance"] > DISTANCE_GATE:
        failures.append(f"holdout-official: target distance {best['distance']} "
                        f"exceeds {DISTANCE_GATE}")
    if official["rank1_source_type"] != "official":
        failures.append(
            f"holdout-official: rank-1 source_type "
            f"{official['rank1_source_type']!r} is not official"
        )
    return failures


def b567_holdout_gates(results):
    """The B567 sealed-holdout publish gates."""
    failures = []
    advisories = []
    top_k = results.get("top_k", DEFAULT_TOP_K)

    # (1) All six queries pass: every required group is in the top five.
    for entry in results["queries"].values():
        for group in entry["groups"]:
            if not group["hit"]:
                failures.append(
                    f"holdout: {entry['family']} group {group['index']} not in "
                    f"top {top_k}"
                )

    # (2) The hidden generic Set Document Property target: best rank <= 2, ok.
    sdp = _entry_by_family(results, SDP_FAMILY)
    sdp_best = _best_group(sdp)
    if sdp_best["rank"] is None or sdp_best["rank"] > 2:
        failures.append(
            f"holdout-sdp: generic target rank {sdp_best['rank']} exceeds 2"
        )
    if sdp["status"] != "ok":
        failures.append(f"holdout-sdp: status {sdp['status']!r} is not ok")

    # (3) The SDP target-specific distance is reported and, above 0.45, flagged
    # as an advisory — NEVER a failure.
    advisories.append(
        f"holdout-sdp: generic target distance {sdp_best['distance']}"
    )
    if sdp_best["distance"] is None or sdp_best["distance"] > DISTANCE_GATE:
        advisories.append(
            f"holdout-sdp: generic target distance {sdp_best['distance']} "
            f"exceeds {DISTANCE_GATE} (advisory only)"
        )

    # (4) The four other Companion families: all groups top-5 (recall covered by
    # gate (1) above) with status ok. Reading each entry also asserts the family
    # is present in the results.
    for family in NON_SDP_COMPANION_FAMILIES:
        entry = _entry_by_family(results, family)
        if entry["status"] != "ok":
            failures.append(f"holdout: {family} status {entry['status']!r} "
                            "is not ok")

    # (5) The official target stays hard.
    failures.extend(_official_holdout_failures(results))

    return GateReport(passed=not failures, failures=failures,
                      advisories=advisories)


def b56_holdout_gates(results, results_c0):
    """The B56 sealed-holdout gates: official stays hard; every other hidden
    query may not regress in rank or status versus C0; the generic rescue gate
    does not apply."""
    failures = []
    advisories = []

    failures.extend(_official_holdout_failures(results))

    # The rank no-regression check is only meaningful if the C0 baseline is
    # anchor-matchable. Legacy (c0) corpora carry no source_record_id, so
    # match_target resolves nothing and every C0 group is a miss (rank 99) —
    # which would make "rank <= C0 rank" pass vacuously. Refuse to certify
    # no-regression against a baseline whose Companion targets are all misses.
    if all(not _best_group(_entry_by_family(results_c0, family))["hit"]
           for family in COMPANION_FAMILIES):
        failures.append(
            "holdout-regression: the C0 baseline matched zero Companion holdout "
            "targets — its chunks are not anchor-matchable (no source_record_id), "
            "so a rank no-regression comparison cannot be computed. Build the C0 "
            "holdout baseline from source-record-tracked chunks."
        )

    for family in COMPANION_FAMILIES:  # every non-official hidden query
        entry = _entry_by_family(results, family)
        base = _entry_by_family(results_c0, family)
        rank = _rank_or_miss(_best_group(entry))
        base_rank = _rank_or_miss(_best_group(base))
        if rank > base_rank:
            failures.append(
                f"holdout-regression: {family} target rank {rank} regressed vs "
                f"C0 rank {base_rank}"
            )
        if _status_rank(entry["status"]) < _status_rank(base["status"]):
            failures.append(
                f"holdout-regression: {family} status {entry['status']!r} "
                f"regressed vs C0 status {base['status']!r}"
            )

    advisories.append(
        "B56 holdout is not held to the generic Set Document Property rescue "
        "gate (rank<=2, status=ok); a B56 release does not claim the generic "
        "defect is solved."
    )
    return GateReport(passed=not failures, failures=failures,
                      advisories=advisories)
