"""Tests for companion.py: pinned-commit enforcement, path hygiene, Markdown
splitting/filtering, and raw-XML stripping."""

import companion


# --- pinned-commit + URL helpers ---------------------------------------------

def test_is_pinned_commit_accepts_full_hex_sha():
    assert companion.is_pinned_commit("19aacdd0aa4c9c83f5d87e9b89fb213044447f52")


def test_is_pinned_commit_rejects_moving_refs_and_short_shas():
    for bad in ("main", "master", "HEAD", "v1.0.22", "19aacdd", "", None,
                "19AACDD0AA4C9C83F5D87E9B89FB213044447F52"):  # uppercase rejected
        assert not companion.is_pinned_commit(bad), bad


def test_build_url_fills_commit_and_path():
    url = companion.build_url(
        "https://raw.githubusercontent.com/o/r/{commit}/{path}",
        "deadbeef" * 5, "references/x.md",
    )
    assert url == "https://raw.githubusercontent.com/o/r/" + "deadbeef" * 5 + "/references/x.md"


def test_build_url_requires_path_placeholder():
    import pytest
    with pytest.raises(ValueError):
        companion.build_url("https://example.com/no-placeholder", "sha", "p.md")


def test_is_denied_path_blocks_traversal_and_denylist():
    assert companion.is_denied_path("../etc/passwd")
    assert companion.is_denied_path("/absolute/path.md")
    assert companion.is_denied_path("dir\\win.md")
    assert companion.is_denied_path("scripts/install.sh")
    assert companion.is_denied_path("references/README.md")
    assert companion.is_denied_path("CLAUDE.md")
    assert companion.is_denied_path("references/guides/boomi_error_reference.md")
    assert companion.is_denied_path("")
    assert companion.is_denied_path(None)


def test_is_denied_path_rejects_non_canonical_forms():
    # Non-canonical aliases of an allowed path must be rejected outright so the
    # denylist and the fetch dedupe cannot be fooled by ./x or x//y.
    assert companion.is_denied_path("./references/x.md")
    assert companion.is_denied_path("references//x.md")
    assert companion.is_denied_path("references/./components/x.md")


def test_is_denied_path_allows_normal_reference_md():
    assert not companion.is_denied_path("references/components/map_component.md")


def test_companion_chunk_id_is_stable_and_unique_per_path():
    a = companion.companion_chunk_id("references/components/map_component.md", 0)
    b = companion.companion_chunk_id("references/components/map_component.md", 0)
    c = companion.companion_chunk_id("references/steps/map.md", 0)  # same basename, diff path
    assert a == b            # deterministic
    assert a != c            # different path -> different id despite same basename
    assert a.endswith("_000")


# --- split_markdown_sections --------------------------------------------------

def test_split_markdown_sections_splits_on_1_to_3_hashes_only():
    md = "# Title\n\nintro line\n\n## Sec A\n\nbody a\n\n### Sub\n\nsub body\n\n#### Deep\n\ndeep stays"
    secs = companion.split_markdown_sections(md)
    headings = [(s["level"], s["heading"]) for s in secs]
    assert headings == [(1, "Title"), (2, "Sec A"), (3, "Sub")]
    # An h4 is not a split point — it remains in the h3 section body.
    assert "#### Deep" in secs[-1]["body"]
    assert "deep stays" in secs[-1]["body"]


def test_split_markdown_sections_ignores_hashes_inside_code_fences():
    md = "## Real\n\n```bash\n# not a heading\necho hi\n```\n\nafter"
    secs = companion.split_markdown_sections(md)
    assert [s["heading"] for s in secs] == ["Real"]
    assert "# not a heading" in secs[0]["body"]


def test_split_markdown_sections_tilde_fence_not_closed_by_backticks():
    # A ~~~ fence must not be closed by a ``` line inside it — the # after the
    # backticks is still fenced body, not a section split (CommonMark).
    md = "## Sec\n\n~~~\n```\n# still body\n~~~\n\n## Next\n\ntail"
    secs = companion.split_markdown_sections(md)
    assert [s["heading"] for s in secs] == ["Sec", "Next"]
    assert "# still body" in secs[0]["body"]


def test_split_markdown_sections_captures_preamble():
    md = "preamble words before any heading\n\n# Title\n\nbody"
    secs = companion.split_markdown_sections(md)
    assert secs[0]["level"] == 0 and secs[0]["heading"] == ""
    assert "preamble words" in secs[0]["body"]
    assert secs[1]["heading"] == "Title"


def test_split_markdown_sections_strips_closed_atx_hashes():
    # A closed-ATX heading's trailing '#' run must not leak into the heading text.
    secs = companion.split_markdown_sections("## Configuration ##\n\nbody")
    assert secs[0]["heading"] == "Configuration"
    # A '#' not preceded by whitespace (e.g. a language name) is preserved.
    secs2 = companion.split_markdown_sections("## Using C#\n\nbody")
    assert secs2[0]["heading"] == "Using C#"


# --- filter_sections ----------------------------------------------------------

def _secs(*pairs):
    return [{"level": lvl, "heading": h, "body": body} for lvl, h, body in pairs]


def test_filter_sections_drops_contents_toc_by_default():
    secs = _secs((1, "Title", "x"), (2, "Contents", "- a\n- b"), (2, "Real", "body"))
    kept = [s["heading"] for s in companion.filter_sections(secs)]
    assert kept == ["Title", "Real"]


def test_filter_sections_allowlist_keeps_matching_and_descendants():
    secs = _secs(
        (1, "Doc", ""),
        (2, "Component Structure", "xml"),
        (2, "Instance Identifiers and Qualifiers", "concept"),
        (3, "tagLists", "detail"),
        (3, "Runtime Behavior", "runtime"),
        (2, "Data Types", "types"),
    )
    kept = [s["heading"] for s in companion.filter_sections(
        secs, allow_patterns=["instance identifier", "taglist", "qualifier"]
    )]
    # The matching H2 and all of its H3 descendants survive; unrelated H2s drop.
    assert kept == ["Instance Identifiers and Qualifiers", "tagLists", "Runtime Behavior"]


def test_filter_sections_denylist_drops_matching_and_descendants():
    secs = _secs(
        (1, "Doc", ""),
        (2, "Overview", "useful"),
        (2, "Component Structure", "xml scaffolding"),
        (3, "Required Elements", "more xml"),
        (2, "Authentication", "auth"),
    )
    kept = [s["heading"] for s in companion.filter_sections(
        secs, drop_sections=["component structure"]
    )]
    assert kept == ["Doc", "Overview", "Authentication"]


def test_filter_sections_never_drops_tech_preview_warning():
    secs = _secs(
        (2, "Contents", "- toc"),
        (2, "Component Structure", "> Technology Preview (Jan 2026): not production ready"),
    )
    kept = [s["heading"] for s in companion.filter_sections(
        secs, drop_sections=["component structure"]
    )]
    # A Tech Preview warning survives both the Contents rule and the denylist.
    assert "Component Structure" in kept


def test_tech_preview_warning_does_not_override_the_allowlist():
    """A deep section is fail-closed even when its body carries the banner.

    The upstream Tech Preview docs repeat the banner throughout, so exempting every
    warning-bearing section from the allowlist would ingest whatever those files
    happen to contain — including sections a curator never named.
    """
    secs = _secs(
        (2, "Wanted", "the curated topic"),
        (2, "Changelog", "> Technology Preview (Jan 2026): not production ready"),
    )
    kept = [s["heading"] for s in companion.filter_sections(secs, allow_patterns=["wanted"])]
    assert kept == ["Wanted"], "an un-allowlisted section rode in on the Tech Preview banner"


def test_document_level_tech_preview_caveat_survives_the_allowlist():
    """The doc's own caveat is kept: there is exactly one, and every chunk needs it.

    It cannot be readmitted via allow_patterns — the "#" title is an ancestor of every
    section, so a pattern matching it would match the whole file.
    """
    secs = _secs(
        (1, "MCP Server Reference", "> Technology Preview: not production ready"),
        (2, "Wanted", "the curated topic"),
        (2, "Changelog", "shipped v2"),
    )
    kept = [s["heading"] for s in companion.filter_sections(secs, allow_patterns=["wanted"])]
    assert kept == ["MCP Server Reference", "Wanted"]


def test_only_the_first_doc_level_caveat_bypasses_the_allowlist():
    """"Exactly one caveat per document" is enforced, not assumed.

    The upstream docs repeat the banner freely, so a SECOND top-level section carrying
    it would otherwise smuggle its whole body past the allowlist.
    """
    secs = _secs(
        (1, "MCP Server Reference", "> Technology Preview: not production ready"),
        (2, "Wanted", "the curated topic"),
        (1, "Appendix: Canned Examples", "> Technology Preview — and 400 lines of XML"),
    )
    kept = [s["heading"] for s in companion.filter_sections(secs, allow_patterns=["wanted"])]
    assert kept == ["MCP Server Reference", "Wanted"]


# --- curation_drift -----------------------------------------------------------

def _cfg(**over):
    base = {
        "path": "references/components/map_component.md",
        "area": "Components", "title": "Map Component", "priority": 1,
        "sections": ["purpose"], "drop_sections": ["complete examples"],
        "strip_xml_blocks": True,
    }
    base.update(over)
    return base


def _doc(files=None, **over):
    """A companion_sources.json / companion_manifest.json pair shape."""
    base = {
        "repo": "OfficialBoomi/boomi-integration",
        "commit": "a" * 40,
        "raw_url_template": "https://raw.githubusercontent.com/o/r/{commit}/{path}",
        "files": [_cfg()] if files is None else files,
    }
    base.update(over)
    return base


def test_curation_drift_is_silent_when_config_and_manifest_agree():
    assert companion.curation_drift(_doc(), _doc()) == []


def test_curation_drift_ignores_pattern_reordering():
    # filter_sections applies the patterns with any(), so order carries no meaning —
    # re-sorting an allowlist for readability must not demand a re-fetch.
    cfg = _doc(files=[_cfg(sections=["alpha", "beta"])])
    man = _doc(files=[_cfg(sections=["beta", "alpha"])])
    assert companion.curation_drift(cfg, man) == []


def test_curation_drift_catches_a_commit_bump_that_was_never_refetched():
    """The pin is the whole point: a bumped commit must never silently no-op.

    process_companion reads upstream repo/commit back from the MANIFEST, so without
    this the corpus is rebuilt at the previous commit and every chunk is stamped with
    it, while the curator believes the sources were upgraded.
    """
    drift = companion.curation_drift(_doc(commit="b" * 40), _doc())
    assert len(drift) == 1 and "commit" in drift[0]
    assert "b" * 40 in drift[0] and "a" * 40 in drift[0]


def test_curation_drift_catches_every_curator_owned_field():
    # Not just the filters: chunk_markdown_file builds a chunk's title and breadcrumb
    # from `title`/`area`, so a stale manifest mislabels every hit from that source.
    for field, new in [
        ("sections", ["purpose", "key management"]),
        ("drop_sections", []),
        ("strip_xml_blocks", False),
        ("title", "Renamed"),
        ("area", "Renamed Area"),
        ("priority", 4),
    ]:
        drift = companion.curation_drift(_doc(files=[_cfg(**{field: new})]), _doc())
        assert len(drift) == 1 and field in drift[0], f"{field} drift went unreported"

    # ...and the top-level provenance fields, not just the per-file ones.
    for field, new in [
        ("repo", "someone/else"),
        ("raw_url_template", "https://evil.example/{commit}/{path}"),
    ]:
        drift = companion.curation_drift(_doc(**{field: new}), _doc())
        assert len(drift) == 1 and field in drift[0], f"{field} drift went unreported"


def test_curation_drift_reports_added_and_removed_sources():
    assert "not in the manifest" in companion.curation_drift(_doc(), _doc(files=[]))[0]
    assert "no longer in companion_sources.json" in companion.curation_drift(
        _doc(files=[]), _doc()
    )[0]


# --- strip_xml_blocks ---------------------------------------------------------

def test_strip_xml_blocks_removes_large_xml_fence():
    big = "<bns:Component>\n" + ("  <field/>\n" * 400) + "</bns:Component>"
    md = f"Intro text\n\n```xml\n{big}\n```\n\nOutro text"
    out = companion.strip_xml_blocks(md)
    assert "bns:Component" not in out
    assert "omitted" in out
    assert "Intro text" in out and "Outro text" in out


def test_strip_xml_blocks_keeps_small_xml_and_other_languages():
    small_xml = "```xml\n<field id=\"path\" value=\"/users\"/>\n```"
    groovy = "```groovy\n" + ("x = 1\n" * 400) + "```"
    md = f"{small_xml}\n\n{groovy}"
    out = companion.strip_xml_blocks(md)
    assert "<field" in out          # small xml kept
    assert "x = 1" in out           # large groovy kept (not xml)


def test_strip_xml_blocks_keeps_large_untagged_non_xml_with_angle_tokens():
    # A large UNTAGGED fence that merely contains an angle-bracket token (Java
    # generics, a "<token>" placeholder) is not XML and must be kept.
    groovy = "def items = new ArrayList<String>()\n" + ("items.add(x)\n" * 300)
    out = companion.strip_xml_blocks(f"```\n{groovy}\n```")
    assert "items.add" in out
    assert "ArrayList<String>" in out


def test_strip_xml_blocks_strips_large_untagged_xml_tree():
    # A large untagged fence whose body BEGINS with an XML element is stripped.
    xml = "<bns:Component>\n" + ("  <field/>\n" * 300) + "</bns:Component>"
    out = companion.strip_xml_blocks(f"```\n{xml}\n```")
    assert "bns:Component" not in out
    assert "omitted" in out


def test_strip_xml_blocks_strips_untagged_xml_led_by_comment_or_doctype():
    # A raw-XML export that opens with a comment or DOCTYPE header (not the root
    # element) must still be recognized as XML and stripped.
    tree = ("<bns:Component>\n" + ("  <field/>\n" * 300) + "</bns:Component>")
    for prefix in ("<!-- exported from AtomSphere -->\n", "<!DOCTYPE bns>\n"):
        out = companion.strip_xml_blocks(f"```\n{prefix}{tree}\n```")
        assert "bns:Component" not in out, prefix
        assert "omitted" in out, prefix


def test_strip_xml_blocks_keeps_component_skeleton_under_threshold():
    # Documents the gap that makes contains_raw_component_xml() necessary:
    # strip_xml_blocks is size-based, so the sub-threshold component skeletons
    # upstream (map_component's examples are 516-699 chars) survive it. Only a
    # section drop removes those.
    small = "<bns:Component type=\"map\">\n  <field/>\n</bns:Component>"
    assert len(small) < companion.XML_STRIP_MIN_CHARS
    out = companion.strip_xml_blocks(f"```xml\n{small}\n```")
    assert "<bns:Component" in out


# --- contains_raw_component_xml -----------------------------------------------

def test_contains_raw_component_xml_detects_component_root():
    assert companion.contains_raw_component_xml('<bns:Component type="map">')
    # Size-independent: the sub-threshold skeletons strip_xml_blocks keeps.
    assert companion.contains_raw_component_xml("a\n```xml\n<bns:Component/>\n```\n")
    # A namespaced Component root is a serialization even with no attributes.
    assert companion.contains_raw_component_xml("<bns:Component>\n  <field/>\n</bns:Component>")


def test_contains_raw_component_xml_ignores_wanted_bns_snippets():
    # The gate must not ban every bns: tag — encrypted-token handling snippets
    # are curated content the allowlist deliberately keeps.
    assert not companion.contains_raw_component_xml("<bns:encryptedValues>x</bns:encryptedValues>")
    assert not companion.contains_raw_component_xml("<bns:encryptedValue/>")
    # Prose naming the element without serializing it is fine.
    assert not companion.contains_raw_component_xml("the bns:Component root element")


def test_contains_raw_component_xml_detects_underscore_prefixed_namespace():
    # `_` is a valid XML NameStartChar, so an underscore-prefixed namespace is a
    # real serialization the fail-closed gate must catch.
    assert companion.contains_raw_component_xml("<_bns:Component xmlns:_bns='x' type='map'/>")
    assert companion.contains_raw_component_xml("<_ns:Component>\n  <field/>\n</_ns:Component>")


def test_contains_raw_component_xml_detects_default_namespace_root():
    # Boomi also serializes a component root without the bns: prefix, as a
    # default-namespaced <Component …> carrying componentId/name/type. The gate
    # must catch that form too (upstream databasev2 docs use it).
    assert companion.contains_raw_component_xml(
        '<Component componentId="" name="Op" type="database">'
    )
    assert companion.contains_raw_component_xml('<Component xmlns="http://api.platform.boomi.com/">')


def test_contains_raw_component_xml_ignores_bare_component_mention():
    # A prose mention of a bare "<Component>" (no serialization attribute) is
    # not a template and must not fail the build.
    assert not companion.contains_raw_component_xml("update the `<Component>` element via the API")
    # A different element whose name merely starts with "Component" is not a hit.
    assert not companion.contains_raw_component_xml('<ComponentReference id="x"/>')


def test_contains_raw_component_xml_ignores_component_prefixed_element_names():
    # `-` and `.` are legal XML name chars, so `<Component-Reference>` /
    # `<Component.Info>` are distinct elements, not a `<Component>` root. The
    # gate must not fail the build on these even when they carry a type= attr.
    assert not companion.contains_raw_component_xml('<Component-Reference type="x">y</Component-Reference>')
    assert not companion.contains_raw_component_xml('<Component.Info type="x"/>')
    # Same boundary rule for the NAMESPACED form: an element whose name merely
    # begins with "Component" is not a <bns:Component> root.
    assert not companion.contains_raw_component_xml('<bns:ComponentReference type="x">y</bns:ComponentReference>')
    assert not companion.contains_raw_component_xml('<bns:Component-Reference type="x"/>')
    assert not companion.contains_raw_component_xml('<bns:Components type="x"/>')


def test_contains_raw_component_xml_requires_real_root_attribute():
    # The default-ns arm must match a real attribute name — not "type" embedded in
    # a hyphenated name, nor an attr= appearing inside a quoted value (either quote
    # style). Any of these would spuriously fail the build.
    assert not companion.contains_raw_component_xml('<Component data-type="widget"/>')
    assert not companion.contains_raw_component_xml('<Component name="a type=b"/>')
    assert not companion.contains_raw_component_xml("<Component name='a type=b'/>")
    assert not companion.contains_raw_component_xml('<Component name="x componentId=y"/>')
    # A genuine attribute still trips it.
    assert companion.contains_raw_component_xml('<Component type="map">x</Component>')
    assert companion.contains_raw_component_xml('<Component\n  componentId="abc">')


def test_contains_raw_component_xml_is_attribute_order_and_quote_agnostic():
    # A root attribute anywhere in the tag counts, even after a leading quoted
    # attribute — otherwise a real template would LEAK past the gate. Both single-
    # and double-quoted attribute values are handled.
    assert companion.contains_raw_component_xml('<Component name="Op" type="database">x</Component>')
    assert companion.contains_raw_component_xml("<Component name='Op' componentId='x'>")
    assert companion.contains_raw_component_xml('<Component  name="a"   type="b">')


def test_contains_raw_component_xml_handles_non_strings():
    assert not companion.contains_raw_component_xml(None)
    assert not companion.contains_raw_component_xml(123)
