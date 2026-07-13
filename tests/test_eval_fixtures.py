"""Tests for the evaluation fixtures and build-time marker integrity."""
from eval_fixtures import (
    COMPANION_PRIMARY_IDS,
    EXPOSED_REGRESSION,
    VISIBLE_CALIBRATION,
    check_fixture_integrity,
    group_hit,
    page_matches,
    unique_group_identities,
)

_MAP_FUNCTIONS_KEY = (
    "companion://OfficialBoomi/boomi-integration/references/components/"
    "map_component_functions.md"
)


def test_suites_have_the_planned_query_counts_and_ids():
    assert len(VISIBLE_CALIBRATION) == 21
    assert len(EXPOSED_REGRESSION) == 6
    ids = [q.query_id for q in VISIBLE_CALIBRATION]
    assert ids == [
        "G01", "G02", "G03", "G04", "G05",
        "I01", "I02", "I03", "I04", "I05", "I06",
        "T01", "T02", "T03", "T04", "T05",
        "O01", "O02", "O03", "O04", "O05",
    ]
    assert [q.query_id for q in EXPOSED_REGRESSION] == [
        "E01", "E02", "E03", "E04", "E05", "E06",
    ]


def test_query_strings_are_verbatim_from_the_intent_plan():
    by_id = {q.query_id: q.query for q in VISIBLE_CALIBRATION + EXPOSED_REGRESSION}
    assert by_id["G01"] == (
        "component XML structure shape configuration reference example"
    )
    assert by_id["I03"] == (
        "dataContext storeStream document.dynamic.userdefined DDP_COUNTER"
    )
    assert by_id["T04"] == (
        "In what order do I create the components for a Boomi MCP server tool?"
    )
    assert by_id["O05"] == "How does a Decision shape route documents?"
    assert by_id["E06"] == (
        "How do I add a qualifier-based instance identifier to a repeating "
        "XML profile element?"
    )


def test_every_query_has_at_least_one_target_group():
    for query in VISIBLE_CALIBRATION + EXPOSED_REGRESSION:
        assert query.target_groups, query.query_id


def test_companion_primary_suite_is_16_queries_with_17_unique_groups():
    """The intent plan's gate arithmetic: 16 Companion primary queries (G/I/T)
    spanning exactly 17 unique target groups — G01's bare marker and I02's
    angle-bracketed marker are ONE group, T02 reuses I03's."""
    assert len(COMPANION_PRIMARY_IDS) == 16
    primary = [q for q in VISIBLE_CALIBRATION if q.query_id in COMPANION_PRIMARY_IDS]
    assert len(unique_group_identities(primary)) == 17


def test_page_matches_survives_url_encoding_differences():
    assert page_matches(
        "Environment extensions",
        "https://help.boomi.com/docs/Atomsphere/Integration/int-Environment_extensions",
    )
    assert page_matches("map_component_functions.md", _MAP_FUNCTIONS_KEY)
    # map_component.md must NOT match the functions page.
    assert not page_matches("map_component.md", _MAP_FUNCTIONS_KEY)


def test_group_hit_requires_page_and_any_marker_wholly_in_content():
    query = next(q for q in VISIBLE_CALIBRATION if q.query_id == "G01")
    dps_group = query.target_groups[1]
    assert group_hit(dps_group, _MAP_FUNCTIONS_KEY,
                     'uses type="DocumentPropertySet" inside FunctionStep')
    assert not group_hit(dps_group, _MAP_FUNCTIONS_KEY, "no marker here")
    assert not group_hit(dps_group, "https://help.boomi.com/docs/other",
                         'type="DocumentPropertySet"')


def _chunk(page_key, content):
    return {"page_key": page_key, "content": content}


def _g01_queries():
    return [q for q in VISIBLE_CALIBRATION if q.query_id == "G01"]


def test_integrity_passes_when_markers_live_wholly_in_one_chunk():
    chunks = [
        _chunk(_MAP_FUNCTIONS_KEY,
               'Function architecture\n<Functions optimizeExecutionOrder="true">'),
        _chunk(_MAP_FUNCTIONS_KEY,
               'Document property\ntype="DocumentPropertySet" example'),
    ]
    assert check_fixture_integrity(chunks, _g01_queries()) == []


def test_integrity_reports_marker_split_across_chunks():
    # The marker text exists on the page, but only when chunks are joined —
    # S5 split the fragment at the whitespace boundary inside the marker.
    chunks = [
        _chunk(_MAP_FUNCTIONS_KEY, "architecture uses Functions"),
        _chunk(_MAP_FUNCTIONS_KEY,
               'optimizeExecutionOrder="true" and also '
               'type="DocumentPropertySet" wholly here'),
    ]
    errors = check_fixture_integrity(chunks, _g01_queries())
    assert len(errors) == 1
    assert errors[0].startswith("fixture_marker_split: G01")


def test_integrity_reports_missing_marker_and_missing_page():
    chunks = [_chunk(_MAP_FUNCTIONS_KEY, "no relevant markers at all")]
    errors = check_fixture_integrity(chunks, _g01_queries())
    assert len(errors) == 2
    assert all(e.startswith("fixture_marker_missing: G01") for e in errors)

    errors = check_fixture_integrity(
        [_chunk("https://help.boomi.com/docs/unrelated", "text")], _g01_queries()
    )
    assert all("has no chunks in the corpus" in e for e in errors)


def test_integrity_page_identity_targets_need_only_the_page():
    o01 = [q for q in VISIBLE_CALIBRATION if q.query_id == "O01"]
    page = "https://help.boomi.com/docs/int-Environment_extensions"
    assert check_fixture_integrity([_chunk(page, "anything")], o01) == []
    assert check_fixture_integrity(
        [_chunk("https://help.boomi.com/docs/other", "x")], o01
    ) != []


def test_integrity_deduplicates_shared_groups_across_queries():
    """I03 and T02 share the DDP_COUNTER group — a miss reports once."""
    queries = [q for q in VISIBLE_CALIBRATION if q.query_id in ("I03", "T02")]
    errors = check_fixture_integrity(
        [_chunk("companion://OfficialBoomi/boomi-integration/references/steps/"
                "data_process_groovy_step.md", "no counter marker")],
        queries,
    )
    assert len(errors) == 1
