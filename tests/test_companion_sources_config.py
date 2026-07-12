"""Curation-policy guards on the real companion_sources.json allowlist.

These lock in the curation invariants the corpus exists to uphold, per
docs/plans/companion-supplemental-kb-ingestion-plan.json:

* curation is FAIL-CLOSED: every source carries a positive `sections` allowlist
  derived from that source's `section_scope` in the plan, so a section the
  upstream repo adds tomorrow cannot silently enter the KB;
* no low-level XML-building content ("Do not ingest canned XML templates ...");
* no doctrine/gotcha/testing guides ("Do not duplicate design_doctrine ...";
  "M9.1 owns curated gotchas through search_boomi_gotchas").

The unit tests elsewhere cover the *mechanisms* (strip_xml_blocks, filter_sections,
contains_raw_component_xml). These cover the *policy* — a new allowlist entry that
quietly reintroduces a component skeleton or a doctrine doc fails here, without
needing the upstream repo checked out.

NOTE ON XML DENSITY: several plan-scoped sections are XML-dense by nature — the REST
`<field id=.../>` vocabulary and the EDI `<tagLists>` attribute reference are the
corpus's highest-value content for a component-building MCP client, and the plan
scopes them in explicitly. Density is NOT a curation criterion; plan fidelity is.
test_plan_scoped_high_value_sections_are_not_dropped is the keep-list that stops a
future "reduce the XML" pass from gutting them.
"""

import json
import os

import companion

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "companion_sources.json")

# Doctrine / gotcha / testing guidance already served by the MCP through
# design_doctrine.py, operational_gotchas.py (search_boomi_gotchas), and the
# list_capabilities operating_doctrine payload. Ingesting them into the vector KB
# duplicates curated, provenance-tagged content and is an explicit plan non-goal.
_DOCTRINE_PATHS = {
    "references/BOOMI_THINKING.md",
    "references/guides/boomi_patterns.md",
    "references/guides/problem_solving_guide.md",
    "references/guides/process_testing_guide.md",
    "references/guides/boomi_error_reference.md",
}

# Upstream sections that are pure <bns:Component> serializations. Each is under or
# near XML_STRIP_MIN_CHARS in at least one file, so the size-based strip cannot be
# relied on — the allowlist must drop the section outright.
#
# Since every file moved to a positive `sections` allowlist, these drops no longer
# *fire*: no allow pattern selects these headings, so the allowlist already excludes
# them (only 2 of the 15 configured drop patterns can currently change the kept set).
# They are kept as a second gate, and asserted here, precisely so that widening an
# allowlist later cannot silently re-admit a component skeleton — the drop would then
# activate. Read them as a backstop, not as what is doing the work today.
_REQUIRED_DROPS = {
    "references/components/map_component.md": {"component structure", "complete examples"},
    "references/components/map_component_functions.md": {"complete working example"},
    "references/components/mcp_server_connection_component.md": {"xml structure"},
    "references/components/mcp_server_operation_component.md": {"xml structure"},
    # The start step's "XML Structure" is a canned <shape> skeleton with placeholder
    # GUIDs — explicit_non_goal "Do not ingest canned XML templates", and its plan
    # section_scope never names it. Its attribute vocabulary is already carried by the
    # allowlisted "Key Attributes" section.
    "references/steps/mcp_server_start_step.md": {"xml structure"},
}

# Drops mandated by the plan for reasons other than component XML — the REST GET
# body-inheritance claim the plan marks "Exclude ... unless separately
# live-verified". Guarding it here keeps that explicit directive from silently
# regressing.
_PLAN_MANDATED_DROPS = {
    "references/steps/rest_connector_step.md": {"document clearing for get requests"},
}

# Every file is curated with a POSITIVE `sections` allowlist. Per file, the topics
# its plan `section_scope` names explicitly — a regression that silently stops
# ingesting a plan-approved topic fails here. (Not exhaustive: these are the load-
# bearing topics, not every pattern in the allowlist.)
_REQUIRED_SECTIONS = {
    "references/components/map_component.md": {
        "map generation rules", "mapping patterns",
        "instance identifier and qualifier mappings", "key management",
    },
    "references/components/map_component_functions.md": {
        "function architecture", "custom scripting", "key observations",
    },
    "references/components/edi_profile_component.md": {
        "instance identifiers", "qualifier", "taglists",
    },
    "references/components/rest_connection_component.md": {
        "authentication patterns", "required fields",
    },
    "references/components/rest_connector_operation_component.md": {
        "authentication headers", "path configuration", "query parameters",
    },
    "references/components/mcp_server_connection_component.md": {"configuration fields"},
    "references/components/mcp_server_operation_component.md": {
        "tool definition fields", "json schema requirements",
    },
    "references/steps/rest_connector_step.md": {"dynamic properties"},
    "references/steps/mcp_server_start_step.md": {
        "key attributes", "response handling", "component dependencies",
    },
    "references/steps/data_process_groovy_step.md": {
        # "stream management" carries the plan's "storeStream requirement": the
        # passthrough vs content-modification patterns. Rule #1 states the rule but
        # not the two ways to satisfy it, so narrowing to the rule alone loses scope.
        "stream management", "working with properties", "critical rules and gotchas",
    },
    "references/steps/data_process_step.md": {
        "purpose", "custom scripting", "available data process type reference",
    },
    "references/platform_entities/mcp_server.md": {
        "architecture", "url structure", "build order",
    },
}

# The plan scopes these sections IN, and each is XML-dense: they carry the exact
# field-id / attribute vocabulary an MCP client needs to BUILD the component
# (<field id="connectTimeout" value="-1"/>, the customproperties nesting, the
# <TagList>/<TagExpression> attribute reference). A curation pass that optimizes for
# "less XML" would delete precisely these. This is the explicit keep-list that stops
# that.
#
# These are the VERBATIM upstream headings, not allowlist patterns, and the test runs
# the real companion.filter_sections over them. Asserting on headings is what makes
# the guard real: a drop_sections entry cancels a keep by matching the same *heading*,
# which need bear no substring relation to the *pattern* that allowed it — e.g.
# drop "minimal configuration" kills the heading "Required Fields (Minimal
# Configuration)" that the keep pattern "required fields" selects. Comparing patterns
# to patterns would pass that green.
# Each entry is the heading's real ANCESTRY CHAIN, outermost first, target last —
# because filter_sections matches a section by its own heading OR any ancestor, so a
# keep is only reproducible with the chain that carries it (e.g. "Basic Authentication"
# is selected by no pattern of its own; it survives via its "Authentication Patterns"
# parent). Modelling the target alone would report a false failure.
_PLAN_SCOPED_KEEPS = {
    "references/components/rest_connection_component.md": [
        # <field id=...> vocabulary + the required -1/GLOBAL defaults
        [(2, "Required Fields (Minimal Configuration)")],
        # incl. the <bns:encryptedValues> preservation rule
        [(2, "Authentication Patterns"), (3, "Basic Authentication")],
    ],
    "references/components/rest_connector_operation_component.md": [
        [(2, "Common Header Patterns")],  # the type="customproperties" nesting
        [(2, "Query Parameters")],        # static-vs-dynamic dual configuration
    ],
    "references/components/edi_profile_component.md": [
        # <TagList>/<TagExpression> attribute reference
        [(2, "Instance Identifiers and Qualifiers"), (3, "tagLists (Instance Identifiers)")],
    ],
}

# Doctrine / XML-scaffolding topics the groovy allowlist must never re-admit
# (plan non-goals: doctrine duplication + low-level XML templates).
_FORBIDDEN_SECTIONS = {
    "references/steps/data_process_groovy_step.md": {
        "common patterns reference", "xml configuration", "complete examples",
        "step naming convention",
    },
}


def _load():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_every_source_strips_oversized_xml_blocks():
    for entry in _load()["files"]:
        assert entry.get("strip_xml_blocks") is True, entry["path"]


def test_no_doctrine_or_gotcha_guides_in_allowlist():
    paths = {e["path"] for e in _load()["files"]}
    assert not (paths & _DOCTRINE_PATHS), sorted(paths & _DOCTRINE_PATHS)


def test_no_guides_area_entries():
    # Catches a doctrine doc added under a path this test does not yet name.
    areas = {e.get("area", "") for e in _load()["files"]}
    assert "Guides" not in areas


def test_component_serialization_sections_are_dropped():
    by_path = {e["path"]: e for e in _load()["files"]}
    for path, required in _REQUIRED_DROPS.items():
        entry = by_path.get(path)
        assert entry is not None, f"{path} missing from allowlist"
        dropped = {s.lower() for s in (entry.get("drop_sections") or [])}
        assert required <= dropped, f"{path}: missing drop_sections {sorted(required - dropped)}"


def test_plan_mandated_sections_are_dropped():
    by_path = {e["path"]: e for e in _load()["files"]}
    for path, required in _PLAN_MANDATED_DROPS.items():
        entry = by_path.get(path)
        assert entry is not None, f"{path} missing from allowlist"
        dropped = {s.lower() for s in (entry.get("drop_sections") or [])}
        assert required <= dropped, f"{path}: missing drop_sections {sorted(required - dropped)}"


def test_every_source_has_a_positive_section_allowlist():
    """Curation is fail-closed: no source may fall back to whole-file ingestion.

    `sections: null` means "keep everything except drop_sections" — a denylist, so a
    section the upstream repo adds at the next commit bump lands in the KB silently,
    and only the sections someone thought to name are excluded. A positive allowlist
    inverts the default to deny, which is the invariant this whole config exists for.
    """
    for entry in _load()["files"]:
        sections = entry.get("sections")
        assert isinstance(sections, list) and sections, (
            f"{entry['path']}: expected a positive `sections` allowlist, got "
            f"{sections!r}. A null/empty list means full-file ingestion, which "
            "re-admits any section upstream adds later. Derive the allowlist from "
            "this file's `section_scope` in the ingestion plan."
        )


def test_positive_section_allowlists_stay_scoped():
    by_path = {e["path"]: e for e in _load()["files"]}
    assert set(_REQUIRED_SECTIONS) == set(by_path), (
        "every source needs its plan-scoped topics asserted here; "
        f"unasserted: {sorted(set(by_path) - set(_REQUIRED_SECTIONS))}"
    )
    for path, required in _REQUIRED_SECTIONS.items():
        entry = by_path[path]
        have = {s.lower() for s in (entry.get("sections") or [])}
        assert required <= have, f"{path}: missing sections {sorted(required - have)}"


def test_plan_scoped_high_value_sections_are_not_dropped():
    """The keep-list: XML-dense but plan-scoped sections must survive curation.

    These carry the field-id / attribute vocabulary needed to BUILD a component, and
    are the sections a density-driven cleanup would remove first. Run the REAL
    filter_sections over each heading so the assertion covers both ways a keep dies:
    the allowlist no longer selecting it, and a drop_sections entry cancelling it
    (drop is applied after allow, so it silently wins).
    """
    by_path = {e["path"]: e for e in _load()["files"]}
    for path, chains in _PLAN_SCOPED_KEEPS.items():
        entry = by_path.get(path)
        assert entry is not None, f"{path} missing from allowlist"
        for chain in chains:
            # Reconstruct the real ancestry. Bodies are deliberately inert (no "Tech
            # Preview" string), so no exemption can mask a curation failure.
            sections = [{"level": lvl, "heading": h, "body": "x"} for lvl, h in chain]
            kept = {
                s["heading"] for s in companion.filter_sections(
                    sections,
                    allow_patterns=entry.get("sections"),
                    drop_sections=entry.get("drop_sections"),
                )
            }
            target = chain[-1][1]
            assert target in kept, (
                f"{path}: curation no longer keeps the plan-scoped section "
                f"{target!r}. Either no `sections` pattern selects it (directly or via "
                f"an ancestor), or a `drop_sections` pattern cancels it. This section "
                f"carries build-critical field/attribute vocabulary — see "
                f"_PLAN_SCOPED_KEEPS."
            )


def test_forbidden_sections_absent_from_allowlists():
    by_path = {e["path"]: e for e in _load()["files"]}
    for path, forbidden in _FORBIDDEN_SECTIONS.items():
        entry = by_path.get(path)
        assert entry is not None, f"{path} missing from allowlist"
        have = {s.lower() for s in (entry.get("sections") or [])}
        assert not (have & forbidden), f"{path}: allowlist re-admits {sorted(have & forbidden)}"
