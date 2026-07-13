"""Tests for the sealed holdout machinery: decode/commitment, span matching,
run redaction, and the B567/B56 holdout gates."""
import base64
import hashlib
import json

import pytest

from eval_holdout import (
    COMPANION_FAMILIES,
    HOLDOUT_FAMILIES,
    NON_SDP_COMPANION_FAMILIES,
    OFFICIAL_FAMILY,
    SDP_FAMILY,
    b56_holdout_gates,
    b567_holdout_gates,
    holdout_commitment,
    load_holdout,
    match_target,
    run_holdout,
)


# --- fixtures / helpers ---------------------------------------------------------

class FakeKbService:
    """Returns canned hits per query string; records call kwargs. Hits carry
    ``chunk_id`` (the field the pinned KbService.search populates)."""

    def __init__(self, hits_by_query, status="ok"):
        self._hits_by_query = hits_by_query
        self._status = status
        self.calls = []

    def search(self, query, top_k=None, source_type=None):
        assert source_type is None, "holdout runs with source_type unset"
        self.calls.append((query, top_k))
        return {"_success": True, "status": self._status,
                "hits": self._hits_by_query.get(query, [])}


def _hit(chunk_id, distance, source_type="companion_reference"):
    return {"chunk_id": chunk_id, "page_key": "pk", "content": "body",
            "distance": distance, "source_type": source_type}


def _chunk(chunk_id, source_record_id, start, end, section_heading="Heading"):
    return {"id": chunk_id, "source_record_id": source_record_id,
            "source_start_byte": start, "source_end_byte": end,
            "section_heading": section_heading}


def _rows():
    return [
        {"query_id": "H1", "query": "SECRET-QUERY-A", "family": SDP_FAMILY,
         "targets": [{"source_record_id": "rec-A",
                      "anchor": {"start_byte": 10, "end_byte": 20}}]},
        {"query_id": "H2", "query": "SECRET-QUERY-B", "family": OFFICIAL_FAMILY,
         "targets": [{"source_record_id": "rec-B",
                      "anchor": {"section_heading": "Overview"}}]},
    ]


def _encode(rows):
    text = "\n".join(json.dumps(row) for row in rows)
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _holdout_results(rank=1, distance=0.30, status="ok", top_k=5):
    """A results dict shaped like run_holdout output: one query per family,
    each with a single group hitting at ``rank``/``distance``."""
    queries = {}
    for index, family in enumerate(HOLDOUT_FAMILIES):
        queries[f"H{index + 1:02d}"] = {
            "family": family,
            "status": status,
            "rank1_source_type": ("official" if family == OFFICIAL_FAMILY
                                  else "companion_reference"),
            "groups": [{"index": 0, "rank": rank, "distance": distance,
                        "hit": rank is not None}],
        }
    return {"top_k": top_k, "queries": queries}


def _entry(results, family):
    return next(e for e in results["queries"].values() if e["family"] == family)


def _set_family_group(results, family, index, *, rank, distance):
    group = _entry(results, family)["groups"][index]
    group["rank"] = rank
    group["distance"] = distance
    group["hit"] = rank is not None


def _set_family_status(results, family, status):
    _entry(results, family)["status"] = status


# --- load_holdout / holdout_commitment ------------------------------------------

def test_load_holdout_round_trips_rows():
    rows = _rows()
    assert load_holdout({"KB_RELEASE_HOLDOUT_B64": _encode(rows)}) == rows


def test_load_holdout_missing_env_var_raises_keyerror():
    with pytest.raises(KeyError):
        load_holdout({})


def test_load_holdout_bad_base64_raises_valueerror():
    with pytest.raises(ValueError):
        load_holdout({"KB_RELEASE_HOLDOUT_B64": "!!!!"})


def test_load_holdout_bad_json_raises_valueerror():
    payload = base64.b64encode(b"not json{{{").decode("ascii")
    with pytest.raises(ValueError):
        load_holdout({"KB_RELEASE_HOLDOUT_B64": payload})


def test_load_holdout_missing_required_field_raises_valueerror():
    rows = _rows()
    del rows[0]["family"]
    with pytest.raises(ValueError):
        load_holdout({"KB_RELEASE_HOLDOUT_B64": _encode(rows)})


def test_load_holdout_missing_anchor_raises_valueerror():
    rows = _rows()
    del rows[0]["targets"][0]["anchor"]
    with pytest.raises(ValueError):
        load_holdout({"KB_RELEASE_HOLDOUT_B64": _encode(rows)})


def test_holdout_commitment_is_sha256_of_raw_b64_and_row_count():
    rows = _rows()
    raw = _encode(rows)
    commitment = holdout_commitment(raw)
    assert commitment["sha256"] == hashlib.sha256(raw.encode("utf-8")).hexdigest()
    assert commitment["count"] == len(rows)


def test_holdout_commitment_contains_no_query_text():
    raw = _encode(_rows())
    blob = json.dumps(holdout_commitment(raw))
    assert "SECRET-QUERY-A" not in blob
    assert "SECRET-QUERY-B" not in blob


# --- match_target: byte anchors -------------------------------------------------

def test_match_target_byte_anchor_inside_span_matches():
    chunks = [_chunk("c1", "rec-A", 0, 100)]
    target = {"source_record_id": "rec-A",
              "anchor": {"start_byte": 40, "end_byte": 50}}
    assert [c["id"] for c in match_target(target, chunks)] == ["c1"]


def test_match_target_byte_anchor_spanning_two_chunks_matches_both():
    chunks = [_chunk("c1", "rec-A", 0, 100), _chunk("c2", "rec-A", 100, 200)]
    target = {"source_record_id": "rec-A",
              "anchor": {"start_byte": 90, "end_byte": 110}}
    assert sorted(c["id"] for c in match_target(target, chunks)) == ["c1", "c2"]


def test_match_target_byte_anchor_non_overlapping_does_not_match():
    chunks = [_chunk("c1", "rec-A", 0, 100)]
    target = {"source_record_id": "rec-A",
              "anchor": {"start_byte": 300, "end_byte": 310}}
    assert match_target(target, chunks) == []


def test_match_target_byte_anchor_requires_matching_source_record_id():
    chunks = [_chunk("c1", "rec-B", 0, 100)]
    target = {"source_record_id": "rec-A",
              "anchor": {"start_byte": 40, "end_byte": 50}}
    assert match_target(target, chunks) == []


# --- match_target: section identity ---------------------------------------------

def test_match_target_section_identity_is_case_and_whitespace_insensitive():
    chunks = [_chunk("c1", "rec-A", 0, 100,
                     section_heading="  Document   Property ")]
    target = {"source_record_id": "rec-A",
              "anchor": {"section_heading": "document property"}}
    assert [c["id"] for c in match_target(target, chunks)] == ["c1"]


def test_match_target_section_identity_requires_matching_source_record_id():
    chunks = [_chunk("c1", "rec-B", 0, 100, section_heading="Document property")]
    target = {"source_record_id": "rec-A",
              "anchor": {"section_heading": "Document property"}}
    assert match_target(target, chunks) == []


# --- run_holdout ----------------------------------------------------------------

def test_run_holdout_scores_rank_distance_and_status():
    chunks = [_chunk("c1", "rec-A", 0, 100, "Document property"),
              _chunk("c2", "rec-A", 100, 200, "Routing")]
    row = {"query_id": "H1", "query": "how do I stamp a document property",
           "family": SDP_FAMILY,
           "targets": [{"source_record_id": "rec-A",
                        "anchor": {"start_byte": 50, "end_byte": 60}}]}
    service = FakeKbService({row["query"]: [_hit("cX", 0.20, "official"),
                                            _hit("c1", 0.30)]})
    results = run_holdout(service, [row], chunks)
    entry = results["queries"]["H1"]
    assert entry["status"] == "ok"
    assert entry["rank1_source_type"] == "official"
    assert entry["family"] == SDP_FAMILY
    assert "query" not in entry
    assert entry["groups"][0] == {"index": 0, "rank": 2, "distance": 0.30,
                                  "hit": True}
    assert service.calls == [(row["query"], 5)]


def test_run_holdout_marks_miss_when_no_hit_maps_to_a_target_chunk():
    chunks = [_chunk("c1", "rec-A", 0, 100)]
    row = {"query_id": "H1", "query": "q", "family": SDP_FAMILY,
           "targets": [{"source_record_id": "rec-A",
                        "anchor": {"start_byte": 10, "end_byte": 20}}]}
    service = FakeKbService({"q": [_hit("other-chunk", 0.30)]})
    group = run_holdout(service, [row], chunks)["queries"]["H1"]["groups"][0]
    assert group == {"index": 0, "rank": None, "distance": None, "hit": False}


def test_run_holdout_redacts_query_text(capsys):
    secret = "SECRET-HOLDOUT-QUERY-TEXT-42"
    chunks = [_chunk("c1", "rec-A", 0, 100)]
    row = {"query_id": "H1", "query": secret, "family": SDP_FAMILY,
           "targets": [{"source_record_id": "rec-A",
                        "anchor": {"start_byte": 10, "end_byte": 20}}]}
    service = FakeKbService({secret: [_hit("c1", 0.30)]})
    results = run_holdout(service, [row], chunks)
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert secret not in json.dumps(results)
    # A redacted progress line may still name the query_id + counts.
    assert "H1" in captured.out


# --- b567 holdout gates ---------------------------------------------------------

def test_b567_holdout_gates_clean_pass():
    report = b567_holdout_gates(_holdout_results())
    assert report.passed, report.failures


def test_b567_holdout_one_missed_group_fails():
    results = _holdout_results()
    _set_family_group(results, "groovy_stream", 0, rank=None, distance=None)
    report = b567_holdout_gates(results)
    assert not report.passed
    assert any("groovy_stream" in f and "top" in f for f in report.failures)


def test_b567_holdout_sdp_rank_3_fails():
    results = _holdout_results()
    _set_family_group(results, SDP_FAMILY, 0, rank=3, distance=0.30)
    report = b567_holdout_gates(results)
    assert any("rank 3 exceeds 2" in f for f in report.failures)


def test_b567_holdout_sdp_distance_over_gate_is_advisory_not_failure():
    results = _holdout_results()
    _set_family_group(results, SDP_FAMILY, 0, rank=1, distance=0.5)
    report = b567_holdout_gates(results)
    assert report.passed, report.failures
    assert any("0.5" in a and "distance" in a for a in report.advisories)
    assert not any("distance" in f for f in report.failures)


def test_b567_holdout_companion_family_status_not_ok_fails():
    results = _holdout_results()
    _set_family_status(results, "edi_qualifier_map", "low_confidence")
    report = b567_holdout_gates(results)
    assert any("edi_qualifier_map" in f and "status" in f
               for f in report.failures)


def test_b567_holdout_official_rank_4_fails():
    results = _holdout_results()
    _set_family_group(results, OFFICIAL_FAMILY, 0, rank=4, distance=0.30)
    report = b567_holdout_gates(results)
    assert any("exceeds 3" in f for f in report.failures)


# --- b56 holdout gates ----------------------------------------------------------

def test_b56_holdout_official_stays_hard():
    results = _holdout_results()
    _entry(results, OFFICIAL_FAMILY)["rank1_source_type"] = "companion_reference"
    report = b56_holdout_gates(results, _holdout_results())
    assert any("holdout-official" in f for f in report.failures)


def test_b56_holdout_rank_regression_vs_c0_fails():
    c0 = _holdout_results(rank=2)
    b56 = _holdout_results(rank=2)
    _set_family_group(b56, "groovy_stream", 0, rank=4, distance=0.30)
    report = b56_holdout_gates(b56, c0)
    assert any("groovy_stream" in f and "regress" in f for f in report.failures)


def test_b56_holdout_status_regression_fails():
    c0 = _holdout_results(status="ok")
    b56 = _holdout_results(status="ok")
    _set_family_status(b56, "groovy_stream", "low_confidence")
    report = b56_holdout_gates(b56, c0)
    assert any("groovy_stream" in f and "status" in f and "regress" in f
               for f in report.failures)


def test_b56_holdout_equal_ranks_pass_with_rescue_advisory():
    report = b56_holdout_gates(_holdout_results(rank=2), _holdout_results(rank=2))
    assert report.passed, report.failures
    assert any("rescue gate" in a for a in report.advisories)


def test_family_taxonomy_matches_the_intent_plan():
    assert COMPANION_FAMILIES == (
        "set_document_property", "rest_operation_step", "groovy_stream",
        "edi_qualifier_map", "mcp_schema_cookie",
    )
    assert NON_SDP_COMPANION_FAMILIES == COMPANION_FAMILIES[1:]
    assert HOLDOUT_FAMILIES == COMPANION_FAMILIES + ("official",)
