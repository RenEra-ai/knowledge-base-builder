#!/usr/bin/env python3
"""Visible calibration + exposed regression fixtures for kb-25 evaluation.

Encodes the intent plan's suites verbatim (docs/plans/
companion-kb-retrieval-fix-plan.json → evaluation.visible_calibration_suite and
exposed_regression_suite). Each query carries one or more TARGET GROUPS; a
group is satisfied by a top-k hit whose page matches ``page_match`` and whose
raw document contains one of the group's ``markers`` (an empty marker tuple is
a page-identity target).

Marker strings are DATA, verified at build time by
:func:`check_fixture_integrity` (after S5, before embedding): a marker that no
longer occurs wholly inside one emitted raw document fails the build with
``fixture_marker_split`` (present on the page but split across chunks) or
``fixture_marker_missing`` (absent from the page entirely). The check never
changes split placement — it only reports.

Group identity: two targets naming the same page and the same marker modulo
surrounding angle brackets are ONE group (e.g. G01's
``Functions optimizeExecutionOrder="true"`` and I02's
``<Functions optimizeExecutionOrder="true">``), which is how the intent plan
counts 17 unique Companion target groups across the 16 Companion primary
queries.
"""
import re
from dataclasses import dataclass, field

_COMPANION_PREFIX = "companion://OfficialBoomi/boomi-integration/"

# Companion source pages (paths under the pinned upstream repo).
_MAP_FUNCTIONS = _COMPANION_PREFIX + "references/components/map_component_functions.md"
_MAP_COMPONENT = _COMPANION_PREFIX + "references/components/map_component.md"
_EDI_PROFILE = _COMPANION_PREFIX + "references/components/edi_profile_component.md"
_REST_CONNECTION = _COMPANION_PREFIX + "references/components/rest_connection_component.md"
_REST_OPERATION = _COMPANION_PREFIX + "references/components/rest_connector_operation_component.md"
_REST_STEP = _COMPANION_PREFIX + "references/steps/rest_connector_step.md"
_GROOVY_STEP = _COMPANION_PREFIX + "references/steps/data_process_groovy_step.md"
_MCP_OPERATION = _COMPANION_PREFIX + "references/components/mcp_server_operation_component.md"
_MCP_PLATFORM = _COMPANION_PREFIX + "references/platform_entities/mcp_server.md"


@dataclass(frozen=True)
class TargetGroup:
    """One target: a page plus alternative marker strings (any satisfies)."""

    page_match: str
    markers: tuple = ()

    def identity(self):
        """Group identity: page + markers modulo surrounding angle brackets."""
        normalized = tuple(sorted(
            marker.strip().lstrip("<").rstrip(">") for marker in self.markers
        ))
        return (self.page_match, normalized)


@dataclass(frozen=True)
class EvalQuery:
    query_id: str
    query: str
    target_groups: tuple = field(default_factory=tuple)


def _q(query_id, query, *groups):
    return EvalQuery(query_id=query_id, query=query, target_groups=tuple(groups))


VISIBLE_CALIBRATION = [
    # --- generic structural/authoring (G) -------------------------------------
    _q("G01", "component XML structure shape configuration reference example",
       TargetGroup(_MAP_FUNCTIONS, ('Functions optimizeExecutionOrder="true"',)),
       TargetGroup(_MAP_FUNCTIONS, ('type="DocumentPropertySet"',))),
    _q("G02", "component configuration skeleton for a mapping transformation",
       TargetGroup(_MAP_COMPONENT, ("key-based mapping",)),
       TargetGroup(_MAP_FUNCTIONS, ("Functions optimizeExecutionOrder",))),
    _q("G03", "connector connection component fields and operation relationship",
       TargetGroup(_REST_CONNECTION, ("The connection component provides:",))),
    _q("G04", "process step configuration shape for runtime connector parameters",
       TargetGroup(_REST_STEP, ("DUAL CONFIGURATION",))),
    _q("G05", "component structure for repeated EDI loops and instance routing",
       TargetGroup(_EDI_PROFILE, ("tagLists are the sole functional mechanism",))),
    # --- identifier (I) --------------------------------------------------------
    _q("I01", "FunctionStep DocumentPropertySet configuration XML",
       TargetGroup(_MAP_FUNCTIONS, ('type="DocumentPropertySet"',))),
    _q("I02", "map function XML skeleton minimal configuration example",
       TargetGroup(_MAP_FUNCTIONS, ('<Functions optimizeExecutionOrder="true">',))),
    _q("I03", "dataContext storeStream document.dynamic.userdefined DDP_COUNTER",
       TargetGroup(_GROOVY_STEP, ("DDP_COUNTER",))),
    _q("I04", "REST connector dynamicProperties childKey queryParameters processproperty",
       TargetGroup(_REST_STEP, ('childKey="lat"',))),
    _q("I05", "toolSchema defaultValue cookie ConnectorObject customOperationType Listen",
       TargetGroup(_MCP_OPERATION, ('<cookie role="OUTPUT">',))),
    _q("I06", "FunctionStep DocumentPropertySet",
       TargetGroup(_MAP_FUNCTIONS, ("DocumentPropertySet",))),
    # --- task (T) ---------------------------------------------------------------
    _q("T01", "I need to map Ship To and Ship From N1 loops to different target "
              "fields. What should I configure?",
       TargetGroup(_EDI_PROFILE, ("Ship From (SF)",)),
       TargetGroup(_MAP_COMPONENT, ("fromTagListKey",))),
    _q("T02", "Groovy check if DDP exists set with default increment counter",
       TargetGroup(_GROOVY_STEP, ("DDP_COUNTER",))),
    _q("T03", "How should I build a REST integration whose query parameters come "
              "from document properties at runtime?",
       TargetGroup(_REST_OPERATION, ("Leave Values Blank",)),
       TargetGroup(_REST_STEP, ('propertyvalue childKey="lat"',))),
    _q("T04", "In what order do I create the components for a Boomi MCP server tool?",
       TargetGroup(_MCP_PLATFORM, ("Components must be created in this order:",))),
    _q("T05", "When should I avoid Groovy and use native Boomi shapes instead?",
       TargetGroup(_GROOVY_STEP, ("Use Map step with profiles",))),
    # --- official controls (O) ---------------------------------------------------
    _q("O01", "How do environment extensions work?",
       TargetGroup("Environment extensions")),
    _q("O02", "Configure a database connector",
       TargetGroup("database connector", ("communicate with your database",))),
    _q("O03", "Deploy a process to production",
       TargetGroup("deployment", ("There are two steps to deployment",))),
    _q("O04", "Trading partner EDI setup",
       TargetGroup("Trading Partner components")),
    _q("O05", "How does a Decision shape route documents?",
       TargetGroup("Decision shape",
                   ("routes documents based on a true/false comparison",))),
]

EXPOSED_REGRESSION = [
    _q("E01", "How is a Set Document Property function represented inside a "
              "Boomi map component?",
       TargetGroup(_MAP_FUNCTIONS, ('type="DocumentPropertySet"',))),
    _q("E02", "Which REST component settings define request paths, headers, and "
              "document-driven query values?",
       TargetGroup(_REST_OPERATION, ("Leave Values Blank", "requestHeaders")),
       TargetGroup(_REST_STEP, ('propertyvalue childKey="lat"',))),
    _q("E03", "A Data Process script is dropping documents after it runs. What "
              "stream handling is required?",
       TargetGroup(_GROOVY_STEP, ("dataContext.storeStream",))),
    _q("E04", "How do I distinguish repeated N1 loops by qualifier when mapping "
              "an EDI profile?",
       TargetGroup(_EDI_PROFILE, ("QualifierList", "tagLists")),
       TargetGroup(_MAP_COMPONENT, ("fromTagListKey",))),
    _q("E05", "Where are an MCP tool's JSON input schema and output cookie "
              "configured?",
       TargetGroup(_MCP_OPERATION, ("JSON schema",)),
       TargetGroup(_MCP_OPERATION, ('<cookie role="OUTPUT">',))),
    _q("E06", "How do I add a qualifier-based instance identifier to a repeating "
              "XML profile element?",
       TargetGroup("instance identifier",
                   ("Use the Data Elements tab to add instance identifiers",))),
]

# The 16 Companion primary-suite queries (G + I + T; the O controls are official).
COMPANION_PRIMARY_IDS = tuple(
    q.query_id for q in VISIBLE_CALIBRATION if q.query_id[0] in "GIT"
)


def _norm_key(text):
    """Casefold and strip separators so a page matcher survives URL encoding
    (%20 / - / _) differences between the scrape config and the plan text."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def page_matches(page_match, page_key):
    """True when the normalized matcher occurs inside the normalized page_key."""
    return _norm_key(page_match) in _norm_key(page_key)


def group_hit(group, page_key, content):
    """True when a single chunk satisfies the target group wholly."""
    if not page_matches(group.page_match, page_key):
        return False
    if not group.markers:
        return True
    return any(marker in content for marker in group.markers)


def unique_group_identities(queries):
    """Deduplicated group identities across a suite (plan's '17 unique
    Companion target groups' counting rule)."""
    return {g.identity() for q in queries for g in q.target_groups}


def check_fixture_integrity(chunks, queries=None):
    """After S5, before embedding: every visible (page, marker) pair must occur
    wholly inside ONE emitted raw document.

    Returns a list of error strings (empty == pass): ``fixture_marker_split``
    when the marker text exists on the page but only across chunk boundaries,
    ``fixture_marker_missing`` when the page or marker is absent entirely.
    Never changes split placement — report-only.
    """
    if queries is None:
        queries = VISIBLE_CALIBRATION + EXPOSED_REGRESSION

    by_norm_page = {}
    for chunk in chunks:
        by_norm_page.setdefault(_norm_key(chunk["page_key"]), []).append(chunk)

    errors = []
    seen = set()
    for query in queries:
        for group in query.target_groups:
            if group.identity() in seen:
                continue
            seen.add(group.identity())

            norm_match = _norm_key(group.page_match)
            page_chunks = [
                c for key, cs in by_norm_page.items() if norm_match in key
                for c in cs
            ]
            if not page_chunks:
                errors.append(
                    f"fixture_marker_missing: {query.query_id} page "
                    f"{group.page_match!r} has no chunks in the corpus"
                )
                continue
            if not group.markers:
                continue  # page-identity target: page exists, done.

            if any(marker in c["content"] for c in page_chunks
                   for marker in group.markers):
                continue
            # Chunk-level match is exact; the page-level probe that
            # distinguishes "split" from "missing" is whitespace-normalized,
            # since fragments join at whitespace boundaries.
            page_text = " ".join(
                " ".join(c["content"].split()) for c in page_chunks
            )
            if any(" ".join(marker.split()) in page_text
                   for marker in group.markers):
                errors.append(
                    f"fixture_marker_split: {query.query_id} marker(s) "
                    f"{group.markers!r} exist on page {group.page_match!r} but "
                    "only across chunk boundaries — no single raw document "
                    "contains one wholly"
                )
            else:
                errors.append(
                    f"fixture_marker_missing: {query.query_id} marker(s) "
                    f"{group.markers!r} absent from page {group.page_match!r}"
                )
    return errors
