#!/usr/bin/env python3
"""Release gates over evaluation results (intent plan: evaluation.b56_fallback_gates,
b567_gates, identifier_regression_rule).

All functions consume ``eval_harness.run_suite`` result dicts. Query-level
"pass" means AT LEAST ONE of the query's target groups hit the top five —
except where a gate names specific groups (T02, both T03 groups, E02–E05 all
groups), which is why those are called out. "The G01 target" (and any other
singular-target reading of a multi-group query) is the query's BEST group:
lowest rank, ties broken by lowest distance.

Unique Companion target groups are deduplicated by ``TargetGroup.identity()``
across the 16 Companion primary queries (17 unique groups; a group's rank/
distance is the best over every query instance that references it).
"""
from dataclasses import dataclass, field

from eval_fixtures import COMPANION_PRIMARY_IDS

DISTANCE_GATE = 0.45
IMPROVEMENT_GATE = 0.05
MRR_GATE = 0.65

GENERIC_IDS = ("G01", "G02", "G03", "G04", "G05")
IDENTIFIER_IDS = ("I01", "I02", "I03", "I04", "I05", "I06")
TASK_IDS = ("T01", "T02", "T03", "T04", "T05")
OFFICIAL_IDS = ("O01", "O02", "O03", "O04", "O05")
EXPOSED_IDS = ("E01", "E02", "E03", "E04", "E05", "E06")
# Official-target queries: the O controls plus the exposed official fixture.
OFFICIAL_TARGET_IDS = OFFICIAL_IDS + ("E06",)


@dataclass
class GateReport:
    passed: bool
    failures: list = field(default_factory=list)
    advisories: list = field(default_factory=list)


def _groups(results, qid):
    return results["queries"][qid]["groups"]


def _status(results, qid):
    return results["queries"][qid]["status"]


def _query_hit(results, qid):
    """Default query pass: any target group in the top five."""
    return any(g["hit"] for g in _groups(results, qid))


def _group(results, qid, index):
    return _groups(results, qid)[index]


def _best_group(results, qid):
    """The query's best group: lowest rank (misses last), then lowest distance."""
    def _key(g):
        return (g["rank"] is None, g["rank"] if g["rank"] is not None else 0,
                g["distance"] if g["distance"] is not None else float("inf"))

    return min(_groups(results, qid), key=_key)


def _rank1_official(results, qid):
    return results["queries"][qid]["rank1_source_type"] == "official"


def _official_control_failures(results, label):
    """Shared official gates: target rank <= 3 with distance <= 0.45, and the
    rank-1 hit of every official query is source_type=official."""
    failures = []
    for qid in OFFICIAL_TARGET_IDS:
        best = _best_group(results, qid)
        if best["rank"] is None or best["rank"] > 3:
            failures.append(f"{label}: {qid} official target rank "
                            f"{best['rank']} exceeds 3")
        if best["distance"] is None or best["distance"] > DISTANCE_GATE:
            failures.append(f"{label}: {qid} official target distance "
                            f"{best['distance']} exceeds {DISTANCE_GATE}")
        if not _rank1_official(results, qid):
            failures.append(f"{label}: {qid} rank-1 hit is not official")
    return failures


def _unique_companion_groups(results):
    """identity -> best (rank, distance) across the Companion primary queries."""
    best = {}
    for qid in COMPANION_PRIMARY_IDS:
        for group in _groups(results, qid):
            identity = tuple((group["identity"][0], tuple(group["identity"][1])))
            current = best.get(identity)
            candidate = (group["rank"], group["distance"])
            if current is None:
                best[identity] = candidate
                continue
            cur_rank = current[0] if current[0] is not None else 99
            new_rank = candidate[0] if candidate[0] is not None else 99
            cur_dist = current[1] if current[1] is not None else float("inf")
            new_dist = candidate[1] if candidate[1] is not None else float("inf")
            # Best = lowest rank; on an equal rank, the lower distance — so the
            # within-0.45 count and the distance-adjacent gates see the group's
            # best observation, not whichever query happened to be seen first.
            if (new_rank, new_dist) < (cur_rank, cur_dist):
                best[identity] = candidate
    return best


def _improved(candidate_distance, baseline_distance, margin):
    """candidate beats baseline by >= margin. A baseline MISS (None) counts as
    improved-upon whenever the candidate has a distance at all."""
    if candidate_distance is None:
        return False
    if baseline_distance is None:
        return True
    return (baseline_distance - candidate_distance) >= margin


def b56_fallback_gates(b56, c0, b5, build_invariants_ok=True):
    """The separate B56-only release gates (a B56 release fixes truncation and
    context fidelity but does not claim the generic G01 defect is solved)."""
    failures = []
    advisories = []

    if not build_invariants_ok:
        failures.append("build: source-coverage/fence/wrapper/256-token "
                        "invariants did not pass")

    failures.extend(_official_control_failures(b56, "official"))

    for qid in IDENTIFIER_IDS:
        if not _query_hit(b56, qid):
            failures.append(f"identifier: {qid} target not in top 5 (recall@5)")
    i06 = _best_group(b56, "I06")
    if i06["rank"] is None or i06["rank"] > 2:
        failures.append(f"identifier: I06 rank {i06['rank']} exceeds 2")
    if i06["distance"] is None or i06["distance"] > DISTANCE_GATE:
        failures.append(f"identifier: I06 distance {i06['distance']} exceeds "
                        f"{DISTANCE_GATE}")

    task_passes = sum(_query_hit(b56, qid) for qid in TASK_IDS)
    if task_passes < 4:
        failures.append(f"task: only {task_passes}/5 task queries pass (need 4)")
    if not _group(b56, "T02", 0)["hit"]:
        failures.append("task: mandatory T02 target not in top 5")
    for index in (0, 1):
        if not _group(b56, "T03", index)["hit"]:
            failures.append(f"task: mandatory T03 group {index} not in top 5")

    t02 = _group(b56, "T02", 0)
    t02_c0 = _group(c0, "T02", 0)
    if t02["distance"] is None or t02["distance"] > DISTANCE_GATE:
        failures.append(f"task: T02 distance {t02['distance']} exceeds "
                        f"{DISTANCE_GATE}")
    if not _improved(t02["distance"], t02_c0["distance"], IMPROVEMENT_GATE):
        failures.append(
            f"task: T02 distance {t02['distance']} does not improve on C0 "
            f"{t02_c0['distance']} by {IMPROVEMENT_GATE}"
        )

    # No identifier or task target returned in B5 top five falls out of B56.
    for qid in IDENTIFIER_IDS + TASK_IDS:
        for b5_group, b56_group in zip(_groups(b5, qid), _groups(b56, qid)):
            if b5_group["hit"] and not b56_group["hit"]:
                failures.append(
                    f"regression: {qid} group {b5_group['index']} in B5 top 5 "
                    "fell out of B56 top 5"
                )

    g01 = _best_group(b56, "G01")
    g01_c0 = _best_group(c0, "G01")
    if not _improved(g01["distance"], g01_c0["distance"], IMPROVEMENT_GATE):
        failures.append(
            f"generic: G01 distance {g01['distance']} does not improve on C0 "
            f"{g01_c0['distance']} by {IMPROVEMENT_GATE}"
        )
    g01_rank = g01["rank"] if g01["rank"] is not None else 99
    g01_c0_rank = g01_c0["rank"] if g01_c0["rank"] is not None else 99
    if g01_rank > g01_c0_rank:
        failures.append(f"generic: G01 rank worsened ({g01_c0['rank']} -> "
                        f"{g01['rank']})")

    recall_ids = GENERIC_IDS + EXPOSED_IDS
    b56_recall = sum(g["hit"] for qid in recall_ids for g in _groups(b56, qid))
    c0_recall = sum(g["hit"] for qid in recall_ids for g in _groups(c0, qid))
    if b56_recall < c0_recall:
        failures.append(
            f"recall: generic+exposed group recall {b56_recall} fell below "
            f"C0's {c0_recall}"
        )

    advisories.append(
        "B56 is not held to the G01 rescue gate (rank<=2, distance<=0.45, "
        "status=ok); a B56 release does not claim the generic defect is solved."
    )
    return GateReport(passed=not failures, failures=failures,
                      advisories=advisories)


def b567_gates(b567, b56, c0):
    """The B567 publish gates."""
    failures = []
    advisories = []

    generic_passes = sum(_query_hit(b567, qid) for qid in GENERIC_IDS)
    if generic_passes < 4:
        failures.append(f"generic: only {generic_passes}/5 pass (need 4)")
    for qid in IDENTIFIER_IDS:
        if not _query_hit(b567, qid):
            failures.append(f"identifier: {qid} target not in top 5")
    task_passes = sum(_query_hit(b567, qid) for qid in TASK_IDS)
    if task_passes < 4:
        failures.append(f"task: only {task_passes}/5 pass (need 4)")

    primary_passes = sum(_query_hit(b567, qid) for qid in COMPANION_PRIMARY_IDS)
    if primary_passes < 14:
        failures.append(
            f"companion: only {primary_passes}/16 primary-suite queries pass "
            "(need 14)"
        )

    unique = _unique_companion_groups(b567)
    hits = sum(1 for rank, _distance in unique.values() if rank is not None)
    if hits < 15:
        failures.append(
            f"companion: only {hits}/{len(unique)} unique target groups in "
            "top 5 (need 15)"
        )
    within = sum(
        1 for _rank, distance in unique.values()
        if distance is not None and distance <= DISTANCE_GATE
    )
    if within < 14:
        failures.append(
            f"companion: only {within}/{len(unique)} unique target groups "
            f"within distance {DISTANCE_GATE} (need 14)"
        )
    mrr = sum(
        (1.0 / rank) if rank is not None else 0.0
        for rank, _distance in unique.values()
    ) / len(unique)
    if mrr < MRR_GATE - 1e-9:
        failures.append(f"companion: target-group MRR {mrr:.4f} below {MRR_GATE}")

    # Mandatory queries/groups.
    if not _query_hit(b567, "G01"):
        failures.append("mandatory: G01 has no target in top 5")
    if not _group(b567, "T02", 0)["hit"]:
        failures.append("mandatory: T02 target not in top 5")
    for index in (0, 1):
        if not _group(b567, "T03", index)["hit"]:
            failures.append(f"mandatory: T03 group {index} not in top 5")

    # The G01 rescue gate — the release claim that the generic defect is fixed.
    g01 = _best_group(b567, "G01")
    if g01["rank"] is None or g01["rank"] > 2:
        failures.append(f"rescue: G01 target rank {g01['rank']} exceeds 2")
    if g01["distance"] is None or g01["distance"] > DISTANCE_GATE:
        failures.append(f"rescue: G01 target distance {g01['distance']} "
                        f"exceeds {DISTANCE_GATE}")
    if _status(b567, "G01") != "ok":
        failures.append(f"rescue: G01 status {_status(b567, 'G01')!r} is not ok")

    failures.extend(_official_control_failures(b567, "official"))

    e01 = _best_group(b567, "E01")
    if e01["rank"] is None or e01["rank"] > 2:
        failures.append(f"exposed: E01 rank {e01['rank']} exceeds 2")
    if e01["distance"] is None or e01["distance"] > DISTANCE_GATE:
        failures.append(f"exposed: E01 distance {e01['distance']} exceeds "
                        f"{DISTANCE_GATE}")
    if _status(b567, "E01") != "ok":
        failures.append(f"exposed: E01 status {_status(b567, 'E01')!r} is not ok")
    for qid in ("E02", "E03", "E04", "E05"):
        for group in _groups(b567, qid):
            if not group["hit"]:
                failures.append(
                    f"exposed: {qid} group {group['index']} not in top 5"
                )

    report = identifier_regression(b56, b567)
    failures.extend(report.failures)
    advisories.extend(report.advisories)

    return GateReport(passed=not failures, failures=failures,
                      advisories=advisories)


def identifier_regression(b56, b567):
    """The identifier regression rule: B567 may not LOSE an identifier target
    B56 retrieved; a shared miss is a broader retrieval failure (advisory here —
    the mandatory identifier gates still block B567)."""
    failures = []
    advisories = []
    for qid in IDENTIFIER_IDS:
        for b56_group, b567_group in zip(_groups(b56, qid), _groups(b567, qid)):
            if b56_group["hit"] and not b567_group["hit"]:
                failures.append(
                    f"identifier-regression: {qid} group {b56_group['index']} "
                    "in B56 top 5 lost by B567 — reject B567 and schedule "
                    "raw-plus-description dual-vector evaluation for kb-26"
                )
            elif not b56_group["hit"] and not b567_group["hit"]:
                advisories.append(
                    f"identifier-regression: {qid} group {b56_group['index']} "
                    "missed by BOTH candidates — broader retrieval failure"
                )
    return GateReport(passed=not failures, failures=failures,
                      advisories=advisories)
