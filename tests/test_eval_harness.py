"""Tests for the evaluation harness: suite scoring and run comparison."""
import copy
import json
import os

import pytest

from eval_fixtures import VISIBLE_CALIBRATION
from eval_harness import compare_runs, load_kb_service, run_suite

_MAP_FUNCTIONS_KEY = (
    "companion://OfficialBoomi/boomi-integration/references/components/"
    "map_component_functions.md"
)
_GROOVY_KEY = (
    "companion://OfficialBoomi/boomi-integration/references/steps/"
    "data_process_groovy_step.md"
)


class FakeKbService:
    """Returns canned hits per query substring; records call kwargs."""

    def __init__(self, hits_by_query, status="ok"):
        self._hits_by_query = hits_by_query
        self._status = status
        self.calls = []

    def search(self, query, top_k=None, source_type=None):
        assert source_type is None, "evaluation runs with source_type unset"
        self.calls.append((query, top_k))
        hits = self._hits_by_query.get(query, [])
        return {"_success": True, "status": self._status, "hits": hits}


def _hit(page_key, content, distance, source_type="companion_reference"):
    return {"page_key": page_key, "content": content, "distance": distance,
            "source_type": source_type}


def test_run_suite_scores_rank_distance_and_status():
    g01 = next(q for q in VISIBLE_CALIBRATION if q.query_id == "G01")
    service = FakeKbService({
        g01.query: [
            _hit("https://help.boomi.com/docs/other", "irrelevant", 0.20,
                 "official"),
            _hit(_MAP_FUNCTIONS_KEY, 'has type="DocumentPropertySet" inside',
                 0.30),
        ],
    })
    results = run_suite(service, [g01], top_k=5)
    entry = results["queries"]["G01"]
    assert entry["status"] == "ok"
    assert entry["rank1_source_type"] == "official"
    groups = entry["groups"]
    # Group 0 (Functions optimizeExecutionOrder) misses; group 1 (DPS) at rank 2.
    assert groups[0]["hit"] is False and groups[0]["rank"] is None
    assert groups[1]["hit"] is True
    assert groups[1]["rank"] == 2
    assert groups[1]["distance"] == 0.30
    assert service.calls == [(g01.query, 5)]


def test_run_suite_uses_first_satisfying_hit_for_rank():
    i03 = next(q for q in VISIBLE_CALIBRATION if q.query_id == "I03")
    service = FakeKbService({
        i03.query: [
            _hit(_GROOVY_KEY, "mentions DDP_COUNTER early", 0.25),
            _hit(_GROOVY_KEY, "mentions DDP_COUNTER again", 0.40),
        ],
    })
    results = run_suite(service, [i03])
    group = results["queries"]["I03"]["groups"][0]
    assert group["rank"] == 1
    assert group["distance"] == 0.25


def test_run_suite_results_round_trip_as_json(tmp_path):
    i03 = next(q for q in VISIBLE_CALIBRATION if q.query_id == "I03")
    service = FakeKbService({i03.query: []}, status="low_confidence")
    results = run_suite(service, [i03])
    path = tmp_path / "results.json"
    path.write_text(json.dumps(results), encoding="utf-8")
    assert json.loads(path.read_text()) == results


def test_load_kb_service_builds_the_pinned_track_a_service():
    pytest.importorskip("chromadb")
    pytest.importorskip("sentence_transformers")
    server_path = os.environ.get(
        "BOOMI_MCP_SERVER_PATH",
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "boomi-mcp-server"),
    )
    if not os.path.isdir(os.path.join(server_path, "src")):
        pytest.skip("boomi-mcp-server checkout not available")
    fixtures = os.path.join(server_path, "tests", "fixtures", "kb")
    if not os.path.isfile(os.path.join(fixtures, "manifest.json")):
        pytest.skip("Track A fixture corpus not available")
    # The fixture manifest ships without a prebuilt Chroma dir; build the
    # service against the corpus the Track A tests themselves build.
    import sys
    sys.path.insert(0, os.path.join(server_path, "tests", "kb"))
    from _fixture_corpus import get_fixture_corpus

    service = load_kb_service(get_fixture_corpus(), server_path=server_path)
    result = service.search("database connector", 5)
    assert result["_success"] is True
    assert result["hits"]


# --- compare_runs -------------------------------------------------------------------

def _bundle():
    return {
        "semantic_inputs": [
            {"id": "c1", "raw_document": "doc one",
             "canonical_metadata": {"title": "One"},
             "embedding_input_sha256": "a" * 64},
            {"id": "c2", "raw_document": "doc two",
             "canonical_metadata": {"title": "Two"},
             "embedding_input_sha256": "b" * 64},
        ],
        "semantic_summary": {"semantic_inputs_sha256": "c" * 64},
        "manifest": {"embedding_contract": {"version": 1, "model_id": "m"}},
        "cache_sha256": "d" * 64,
        "results": {"queries": {
            "G01": {"status": "ok", "rank1_source_type": "official",
                    "groups": [{"index": 0, "identity": ["p", []],
                                "rank": 2, "distance": 0.31, "hit": True}]},
        }},
    }


def test_identical_bundles_pass():
    report = compare_runs(_bundle(), _bundle())
    assert report.passed, report.failures


@pytest.mark.parametrize("mutate,expected_class", [
    (lambda b: b["semantic_inputs"].pop(), "ids"),
    (lambda b: b["semantic_inputs"][0].update(raw_document="changed"),
     "documents"),
    (lambda b: b["semantic_inputs"][0].update(embedding_input_sha256="e" * 64),
     "documents"),
    (lambda b: b["semantic_inputs"][0]["canonical_metadata"].update(title="X"),
     "metadata"),
    (lambda b: b["manifest"]["embedding_contract"].update(version=2),
     "contracts"),
    (lambda b: b.update(cache_sha256="f" * 64), "cache"),
    (lambda b: b["semantic_summary"].update(semantic_inputs_sha256="0" * 64),
     "semantic_inputs"),
    (lambda b: b["results"]["queries"]["G01"]["groups"][0].update(rank=3),
     "ranks"),
    (lambda b: b["results"]["queries"]["G01"]["groups"][0].update(hit=False),
     "recall"),
    (lambda b: b["results"]["queries"]["G01"].update(status="low_confidence"),
     "status"),
])
def test_each_hard_gate_class_fails_independently(mutate, expected_class):
    changed = _bundle()
    mutate(changed)
    report = compare_runs(_bundle(), changed)
    assert not report.passed
    assert any(f.startswith(expected_class + ":") for f in report.failures), \
        report.failures


def test_group_count_mismatch_fails_the_hard_gate():
    """A run bundle with fewer target groups must fail, not slip through zip()."""
    fewer = _bundle()
    fewer["results"]["queries"]["G01"]["groups"] = []
    report = compare_runs(_bundle(), fewer)
    assert not report.passed
    assert any("target groups" in f for f in report.failures)


def test_distance_tolerance_boundary():
    within = _bundle()
    within["results"]["queries"]["G01"]["groups"][0]["distance"] = 0.31009
    assert compare_runs(_bundle(), within).passed

    beyond = _bundle()
    beyond["results"]["queries"]["G01"]["groups"][0]["distance"] = 0.31011
    report = compare_runs(_bundle(), beyond)
    assert not report.passed
    assert any(f.startswith("distances:") for f in report.failures)
