"""Curation-policy guards on the real companion_sources.json allowlist.

These lock in the two curation invariants the corpus exists to uphold, per
docs/archive/companion-supplemental-kb-ingestion-plan.json:

* no low-level XML-building content ("Do not ingest canned XML templates ...");
* no doctrine/gotcha/testing guides ("Do not duplicate design_doctrine ...";
  "M9.1 owns curated gotchas through search_boomi_gotchas").

The unit tests elsewhere cover the *mechanisms* (strip_xml_blocks, filter_sections,
contains_raw_component_xml). These cover the *policy* — a new allowlist entry that
quietly reintroduces a component skeleton or a doctrine doc fails here, without
needing the upstream repo checked out.
"""

import json
import os

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
_REQUIRED_DROPS = {
    "references/components/map_component.md": {"component structure", "complete examples"},
    "references/components/map_component_functions.md": {"complete working example"},
    "references/components/mcp_server_connection_component.md": {"xml structure"},
    "references/components/mcp_server_operation_component.md": {"xml structure"},
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
