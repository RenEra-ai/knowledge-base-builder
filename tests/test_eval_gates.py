"""Boundary tests for the B56 fallback gates, B567 gates, and the identifier
regression rule."""
import copy

from eval_fixtures import EXPOSED_REGRESSION, VISIBLE_CALIBRATION
from eval_gates import (
    IDENTIFIER_IDS,
    OFFICIAL_TARGET_IDS,
    b56_fallback_gates,
    b567_gates,
    identifier_regression,
)

_ALL_QUERIES = VISIBLE_CALIBRATION + EXPOSED_REGRESSION


def _passing_results(rank=1, distance=0.30):
    """Every group hits at rank/distance; every status ok; official rank-1."""
    queries = {}
    for query in _ALL_QUERIES:
        groups = []
        for index, group in enumerate(query.target_groups):
            page, markers = group.identity()
            groups.append({"index": index, "identity": [page, list(markers)],
                           "rank": rank, "distance": distance, "hit": True})
        queries[query.query_id] = {
            "query": query.query,
            "status": "ok",
            "rank1_source_type": ("official"
                                  if query.query_id in OFFICIAL_TARGET_IDS
                                  else "companion_reference"),
            "groups": groups,
        }
    return {"top_k": 5, "queries": queries}


def _baseline_c0():
    """A C0 baseline that every improvement gate clears (distance 0.40)."""
    return _passing_results(rank=2, distance=0.40)


def _set_group(results, qid, index, *, rank, distance):
    group = results["queries"][qid]["groups"][index]
    group["rank"] = rank
    group["distance"] = distance
    group["hit"] = rank is not None


def _miss_query(results, qid):
    for group in results["queries"][qid]["groups"]:
        group["rank"] = None
        group["distance"] = None
        group["hit"] = False


# --- b56 fallback gates ---------------------------------------------------------

def test_b56_gates_pass_on_the_clean_construction():
    report = b56_fallback_gates(_passing_results(), _baseline_c0(),
                                _passing_results())
    assert report.passed, report.failures
    # The G01 rescue gate is explicitly NOT applied to B56.
    assert any("rescue gate" in a for a in report.advisories)


def test_b56_build_invariants_flag_fails():
    report = b56_fallback_gates(_passing_results(), _baseline_c0(),
                                _passing_results(), build_invariants_ok=False)
    assert any(f.startswith("build:") for f in report.failures)


def test_b56_official_rank_and_distance_gates():
    b56 = _passing_results()
    _set_group(b56, "O03", 0, rank=4, distance=0.30)
    report = b56_fallback_gates(b56, _baseline_c0(), _passing_results())
    assert any("O03 official target rank" in f for f in report.failures)

    b56 = _passing_results()
    _set_group(b56, "E06", 0, rank=2, distance=0.46)
    report = b56_fallback_gates(b56, _baseline_c0(), _passing_results())
    assert any("E06 official target distance" in f for f in report.failures)


def test_b56_official_rank1_precedence():
    b56 = _passing_results()
    b56["queries"]["O01"]["rank1_source_type"] = "companion_reference"
    report = b56_fallback_gates(b56, _baseline_c0(), _passing_results())
    assert any("O01 rank-1 hit is not official" in f for f in report.failures)


def test_b56_i06_rank_gate():
    b56 = _passing_results()
    _set_group(b56, "I06", 0, rank=3, distance=0.30)
    report = b56_fallback_gates(b56, _baseline_c0(), _passing_results())
    assert any("I06 rank 3 exceeds 2" in f for f in report.failures)


def test_b56_task_count_and_mandatory_groups():
    b56 = _passing_results()
    _miss_query(b56, "T01")
    _miss_query(b56, "T04")
    report = b56_fallback_gates(b56, _baseline_c0(), _passing_results())
    assert any("only 3/5 task queries pass" in f for f in report.failures)

    b56 = _passing_results()
    _set_group(b56, "T03", 1, rank=None, distance=None)
    report = b56_fallback_gates(b56, _baseline_c0(), _passing_results())
    assert any("mandatory T03 group 1" in f for f in report.failures)


def test_b56_t02_improvement_boundary():
    c0 = _baseline_c0()  # T02 distance 0.40
    b56 = _passing_results(distance=0.35)  # improvement exactly 0.05
    report = b56_fallback_gates(b56, c0, _passing_results(distance=0.35))
    assert not any("T02 distance" in f and "improve" in f
                   for f in report.failures), report.failures

    b56 = _passing_results(distance=0.3510001)  # 0.049 improvement
    report = b56_fallback_gates(b56, c0, _passing_results(distance=0.3510001))
    assert any("does not improve on C0" in f for f in report.failures)


def test_b56_b5_fallout_regression():
    b5 = _passing_results()
    b56 = _passing_results()
    _set_group(b56, "I04", 0, rank=None, distance=None)
    report = b56_fallback_gates(b56, _baseline_c0(), b5)
    assert any("I04 group 0 in B5 top 5 fell out" in f for f in report.failures)


def test_b56_g01_rank_must_not_worsen():
    b56 = _passing_results()
    _set_group(b56, "G01", 0, rank=4, distance=0.30)
    _set_group(b56, "G01", 1, rank=4, distance=0.30)
    report = b56_fallback_gates(b56, _baseline_c0(), _passing_results())
    assert any("G01 rank worsened" in f for f in report.failures)


def test_b56_generic_exposed_recall_floor():
    b56 = _passing_results()
    _miss_query(b56, "E03")
    report = b56_fallback_gates(b56, _baseline_c0(), _passing_results())
    assert any("fell below C0" in f for f in report.failures)


# --- b567 gates ------------------------------------------------------------------

def test_b567_gates_pass_on_the_clean_construction():
    report = b567_gates(_passing_results(), _passing_results(), _baseline_c0())
    assert report.passed, report.failures


def test_b567_companion_primary_count_boundary():
    b567 = _passing_results()
    b56 = _passing_results()
    for qid in ("G02", "G03", "T05"):
        _miss_query(b567, qid)
        _miss_query(b56, qid)  # keep identifier-regression quiet
    report = b567_gates(b567, b56, _baseline_c0())
    assert any("13/16 primary-suite queries" in f for f in report.failures)


def test_b567_unique_group_hits_boundary():
    b567 = _passing_results()
    b56 = _passing_results()
    # Missing G03 + G04 + G05 drops three UNIQUE groups -> 14/17 (< 15).
    for qid in ("G03", "G04", "G05"):
        _miss_query(b567, qid)
        _miss_query(b56, qid)
    report = b567_gates(b567, b56, _baseline_c0())
    assert any("unique target groups in" in f for f in report.failures)


def test_unique_companion_groups_tiebreaks_equal_rank_by_lower_distance():
    """When the same group is seen at equal rank by two queries, the LOWER
    distance wins — so the within-0.45 count reflects the group's best."""
    from eval_gates import _unique_companion_groups

    # G01's DocumentPropertySet group also appears in I01/I06; give one instance
    # rank 1 @ 0.50 and another rank 1 @ 0.30. The 0.30 must be recorded.
    results = _passing_results(rank=1, distance=0.50)
    _set_group(results, "I01", 0, rank=1, distance=0.30)  # DPS group, better dist
    best = _unique_companion_groups(results)
    dps_key = next(k for k in best if 'type="DocumentPropertySet"' in "".join(k[1]))
    assert best[dps_key] == (1, 0.30)


def test_b567_mrr_gate_fails_in_isolation():
    b567 = _passing_results()
    # Every group still hits, but at rank 5 -> MRR 0.2 << 0.65 while the
    # hit-count and distance gates stay green.
    for qid in ("G01", "G02", "G03", "G04", "G05", "I01", "I02", "I03", "I04",
                "I05", "T01", "T04", "T05"):
        for group in b567["queries"][qid]["groups"]:
            group["rank"] = 5
    report = b567_gates(b567, _passing_results(rank=5), _baseline_c0())
    assert any("MRR" in f for f in report.failures)


def test_b567_g01_rescue_gate():
    b567 = _passing_results()
    _set_group(b567, "G01", 0, rank=3, distance=0.30)
    _set_group(b567, "G01", 1, rank=3, distance=0.30)
    report = b567_gates(b567, _passing_results(), _baseline_c0())
    assert any("rescue: G01 target rank 3 exceeds 2" in f
               for f in report.failures)

    b567 = _passing_results()
    b567["queries"]["G01"]["status"] = "low_confidence"
    report = b567_gates(b567, _passing_results(), _baseline_c0())
    assert any("rescue: G01 status" in f for f in report.failures)


def test_b567_e01_and_e02_gates():
    b567 = _passing_results()
    _set_group(b567, "E01", 0, rank=3, distance=0.30)
    report = b567_gates(b567, _passing_results(), _baseline_c0())
    assert any("E01 rank 3 exceeds 2" in f for f in report.failures)

    b567 = _passing_results()
    _set_group(b567, "E02", 1, rank=None, distance=None)
    report = b567_gates(b567, _passing_results(), _baseline_c0())
    assert any("E02 group 1 not in top 5" in f for f in report.failures)


def test_b567_official_controls_reuse_the_b56_official_gates():
    b567 = _passing_results()
    b567["queries"]["O05"]["rank1_source_type"] = "companion_reference"
    report = b567_gates(b567, _passing_results(), _baseline_c0())
    assert any("O05 rank-1 hit is not official" in f for f in report.failures)


# --- identifier regression rule -----------------------------------------------------

def test_identifier_regression_rejects_b567_losing_a_b56_hit():
    b56 = _passing_results()
    b567 = _passing_results()
    _set_group(b567, "I02", 0, rank=None, distance=None)
    report = identifier_regression(b56, b567)
    assert not report.passed
    assert any("kb-26" in f for f in report.failures)


def test_identifier_regression_shared_miss_is_advisory():
    b56 = _passing_results()
    b567 = _passing_results()
    _set_group(b56, "I05", 0, rank=None, distance=None)
    _set_group(b567, "I05", 0, rank=None, distance=None)
    report = identifier_regression(b56, b567)
    assert report.passed
    assert any("broader retrieval failure" in a for a in report.advisories)


def test_identifier_regression_is_scoped_to_identifier_queries():
    assert set(IDENTIFIER_IDS) == {"I01", "I02", "I03", "I04", "I05", "I06"}
    b56 = _passing_results()
    b567 = _passing_results()
    _miss_query(b567, "T05")  # task loss is NOT this rule's concern
    assert identifier_regression(b56, b567).passed
