#!/usr/bin/env python3
"""Shared helpers for the Companion supplemental corpus.

The Companion corpus is a small, curated set of Markdown reference docs pulled
from a pinned commit of a public BSD-2 GitHub repo (currently
OfficialBoomi/boomi-integration). It is *supplemental* implementation context,
not official Boomi documentation, and every chunk it produces is labelled
``source_type="companion_reference"`` / ``verification_status="companion_unverified"``.

This module holds the pure, dependency-light helpers used by both
``fetch_companion.py`` (download) and ``chunk_docs.py`` (Markdown -> chunks):
pinned-commit enforcement, URL construction, allowlist path hygiene, Markdown
heading splitting with section allow/deny filtering, and raw-XML block
stripping. Keeping them here makes them independently unit-testable and keeps
``chunk_docs.py``'s HTML path untouched.
"""

import hashlib
import os
import re

# --- provenance constants -----------------------------------------------------

COMPANION_SOURCE_TYPE = "companion_reference"
COMPANION_VERIFICATION_STATUS = "companion_unverified"
COMPANION_CATEGORY = "Companion Reference"

# Official chunks carry these defaults so both corpora satisfy one uniform
# chunk contract (see build_index.REQUIRED_CHUNK_FIELDS).
OFFICIAL_SOURCE_TYPE = "official"
OFFICIAL_VERIFICATION_STATUS = "official"

# The seven provenance fields every chunk carries in addition to the original
# citation metadata. Official chunks default the repo/url fields to "".
PROVENANCE_FIELDS = (
    "source_type",
    "verification_status",
    "upstream_repo",
    "upstream_commit",
    "source_path",
    "raw_url",
    "latest_url",
)

# A full git commit sha — 40 lowercase hex chars. Moving refs (branches, tags,
# HEAD, short shas) are rejected so a Companion build can never silently drift.
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

# Raw-XML fenced code blocks at or above this size are the low-level
# <bns:Component> serializations we deliberately do not ingest.
XML_STRIP_MIN_CHARS = 1500

# The root element of a Boomi component serialization. Its presence in a chunk
# means a canned component template reached the KB — the low-level XML-building
# content the corpus never ingests. This gate is the fail-closed backstop for
# that case, matching the two forms the upstream repo emits (see the Companion's
# boomi-common.sh, which handles both):
#
#   1. Namespaced root ``<bns:Component …>`` (or any ``<prefix:Component>``). A
#      namespaced element named exactly Component is a serialization even without
#      attributes, so this arm needs no attribute — it catches bare
#      ``<bns:Component>`` / ``<bns:Component/>`` too.
#   2. Default-namespaced root ``<Component …>`` carrying a serialization
#      attribute (componentId/xmlns/type). The attribute is required here so a
#      bare prose "<Component>" mention does NOT trip the gate.
#
# Both arms use a ``(?=[\s/>])`` delimiter lookahead so the element NAME must end
# at "Component": ``<ComponentReference>``, ``<Components>``, and — since ``-``/``.``
# are legal XML name chars — ``<Component-Reference>`` / ``<Component.Info>`` are
# NOT matched. ``<bns:encryptedValues>`` is a different element and is likewise
# untouched: it is wanted content (encrypted-token handling).
#
# This targets component-root definitions specifically — NOT every fragment of
# low-level XML. Small illustrative XML is permitted as short context per the
# ingestion plan ("Exclude large canned XML examples except where needed as short
# context"): field snippets and <bns:encryptedValues> blocks are governed by
# strip_xml_blocks (which drops the oversized blocks) plus per-file section drops.
#
# strip_xml_blocks() alone cannot enforce even this narrow case: it is size-based,
# and several component skeletons upstream are *under* XML_STRIP_MIN_CHARS
# (map_component's examples are 516-699 chars), so they must also be excluded by
# dropping their sections in the allowlist.
#
# The default-namespaced check PARSES the opening tag rather than substring-
# matching: _DEFAULT_COMPONENT_TAG_RE captures the attribute region, consuming
# whole quoted values of either style (so a '>' inside a value does not end the
# tag early), and _ROOT_ATTR_RE then looks for a component-root attribute NAME
# after quoted values are blanked out. This is attribute-order-independent and
# cannot be fooled by a componentId=/type= appearing INSIDE a value, by a
# single-quoted value, or by a hyphenated name such as data-type.
# The prefix starts with an XML NameStartChar — a letter or ``_`` (``[\w.-]``
# then allows the digits/``.``/``-`` that are legal later in a name), so an
# underscore-prefixed namespace like ``<_bns:Component …>`` is still caught.
_NAMESPACED_COMPONENT_RE = re.compile(r"""<[A-Za-z_][\w.-]*:Component(?=[\s/>])""")
_DEFAULT_COMPONENT_TAG_RE = re.compile(r"""<Component(?=[\s/>])((?:[^>"']|"[^"]*"|'[^']*')*)""")
_QUOTED_VALUE_RE = re.compile(r"""("[^"]*"|'[^']*')""")
_ROOT_ATTR_RE = re.compile(r"""(?:^|\s)(?:componentId|xmlns|type)\s*=""")

# Basenames that must never be fetched even if an allowlist names them.
_DENIED_BASENAMES = {"readme.md", "claude.md", "boomi_error_reference.md"}


# --- pinned-commit + URL helpers ---------------------------------------------

def is_pinned_commit(commit):
    """True only for a full 40-char lowercase hex commit sha."""
    return bool(commit) and bool(_COMMIT_RE.match(commit))


def build_url(template, commit, path):
    """Fill a raw/blob URL template with a pinned commit and repo-relative path.

    The template must embed the literal ``{commit}`` and ``{path}`` placeholders
    (blob-at-latest templates use only ``{path}``). Raises ValueError otherwise
    so a template that hardcodes a moving ref cannot slip through.
    """
    if "{path}" not in template:
        raise ValueError(f"URL template missing {{path}} placeholder: {template!r}")
    return template.format(commit=commit, path=path)


def is_denied_path(path):
    """True if a repo-relative path is unsafe, non-canonical, or denylisted.

    Rejects absolute paths, backslashes, and any non-canonical POSIX form —
    parent traversal (``..``), current-dir (``.``), and empty segments (from a
    leading ``./`` or a ``//``). Rejecting non-canonical paths outright (rather
    than silently normalizing them) keeps the denylist and the fetch
    duplicate-check from being fooled by ``./x`` / ``x//y`` aliases of an
    allowed path. Also rejects shell scripts and the excluded-doc basenames —
    defense in depth on top of the explicit allowlist.
    """
    if not path or not isinstance(path, str):
        return True
    if path.startswith("/") or "\\" in path:
        return True
    if any(seg in ("", ".", "..") for seg in path.split("/")):
        return True
    basename = os.path.basename(path).lower()
    if basename.endswith(".sh"):
        return True
    if basename in _DENIED_BASENAMES:
        return True
    return False


def sha256_hex(data):
    """Return the hex SHA-256 of a bytes or str payload."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def companion_chunk_id(source_path, index):
    """Stable, collision-free chunk id for a companion Markdown section.

    Deep repo paths would collide under the HTML path's 60-char filename
    truncation, so we seed the id with a short hash of the full path.
    """
    base = os.path.splitext(os.path.basename(source_path))[0]
    base = re.sub(r"[^a-zA-Z0-9_]", "_", base).strip("_").lower()
    if len(base) > 40:
        base = base[:40].rstrip("_")
    # 12 hex (48 bits) of the full path so distinct paths sharing a basename do
    # not collide; build_index's duplicate-id gate is the loud backstop.
    digest = hashlib.sha1(source_path.encode("utf-8")).hexdigest()[:12]
    return f"companion_{base}_{digest}_{index:03d}"


# --- Markdown splitting / filtering / XML stripping ---------------------------

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.*\S)\s*$")
_FENCE_RE = re.compile(r"^(`{3,}|~{3,})")
# A closed-ATX heading's trailing "#" run (must be space-separated, so "C#" is
# left intact) — stripped from the parsed heading per CommonMark.
_ATX_CLOSE_RE = re.compile(r"\s+#+\s*$")

# Oversized raw-XML fenced code blocks. The fence marker is 3+ backticks/tildes
# (matching _fence_marker), and the closing \2 backreference requires the same
# run. Hoisted to module scope so it is compiled once, not per strip call.
_XML_FENCE_RE = re.compile(
    r"^([ \t]*)(`{3,}|~{3,})[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)\n[ \t]*\2[ \t]*$",
    re.DOTALL | re.MULTILINE,
)
# An untagged fence counts as XML only when its body BEGINS (after optional
# leading whitespace) with XML/SGML markup: a declaration, comment, DOCTYPE,
# CDATA, or an element. Anchoring to the start avoids matching a mid-body
# angle-bracket token (List<String>, "Bearer <token>") in a Groovy/config block,
# while still catching an export that opens with a comment or DOCTYPE header.
_XML_START_RE = re.compile(
    r"\s*(<\?xml|<!--|<!\[CDATA\[|<!DOCTYPE|<[A-Za-z])", re.IGNORECASE
)


def _clean_heading(text):
    """Strip a closed-ATX heading's trailing ``#`` run, then surrounding space."""
    return _ATX_CLOSE_RE.sub("", text).strip()


def _fence_marker(line):
    """Return (char, length) for a Markdown code-fence line, else None.

    A fence is a run of 3+ backticks or 3+ tildes at the start of the (stripped)
    line. Tracking the marker char + length lets us close a fence only with a
    matching fence (CommonMark), so a ``~~~`` block containing a ``` ``` ``` line
    is not mistakenly closed.
    """
    m = _FENCE_RE.match(line.strip())
    if not m:
        return None
    run = m.group(1)
    return run[0], len(run)


def split_markdown_sections(md_text):
    """Split Markdown into sections on ``#``/``##``/``###`` headings.

    Returns a list of dicts ``{"level": int, "heading": str, "body": str}`` in
    document order. Content before the first heading is a level-0 preamble with
    an empty heading. Headings inside fenced code blocks (``` / ~~~) and headings
    deeper than ``###`` are treated as body, never as split points.
    """
    sections = split_markdown_sections_with_offsets(md_text)
    return [
        {"level": sec["level"], "heading": sec["heading"], "body": sec["body"]}
        for sec in sections
    ]


def split_markdown_sections_with_offsets(md_text):
    """``split_markdown_sections`` plus each section's exact source location.

    Every section dict additionally carries ``body_start_byte`` /
    ``body_end_byte``: the UTF-8 byte range of the STRIPPED body within
    ``md_text`` (``md_text`` sliced by that byte range, decoded, equals
    ``body`` exactly — the S5 pipeline records source spans against it). An
    empty body is ``(end, end)`` at the section's start position.
    """
    lines = md_text.split("\n")
    # Byte offset of each line's first char (lines were joined by "\n" = 1 byte).
    line_starts = []
    pos = 0
    for line in lines:
        line_starts.append(pos)
        pos += len(line.encode("utf-8")) + 1

    sections = []
    current = {"level": 0, "heading": "", "body_lines": [], "line_indices": [],
               "heading_index": None}
    open_fence = None  # (char, length) while inside a fenced code block

    def _flush(sec):
        if sec["heading"] or "".join(sec["body_lines"]).strip():
            sections.append(sec)

    for index, line in enumerate(lines):
        marker = _fence_marker(line)

        if open_fence is not None:
            # Inside a fence: only a matching fence (same char, >= length, no
            # trailing info string) closes it; headings never split here.
            if (marker and marker[0] == open_fence[0]
                    and marker[1] >= open_fence[1]
                    and line.strip()[marker[1]:].strip() == ""):
                open_fence = None
            current["body_lines"].append(line)
            current["line_indices"].append(index)
            continue

        if marker:
            # Opening fence — body content, not a split point.
            open_fence = marker
            current["body_lines"].append(line)
            current["line_indices"].append(index)
            continue

        match = _HEADING_RE.match(line)
        if match:
            # Flush the section in progress before starting a new one.
            _flush(current)
            current = {
                "level": len(match.group(1)),
                "heading": _clean_heading(match.group(2)),
                "body_lines": [],
                "line_indices": [],
                "heading_index": index,
            }
        else:
            current["body_lines"].append(line)
            current["line_indices"].append(index)

    _flush(current)

    for sec in sections:
        raw_body = "\n".join(sec.pop("body_lines"))
        body = raw_body.strip()
        sec["body"] = body
        indices = sec.pop("line_indices")
        heading_index = sec.pop("heading_index")
        if body and indices:
            region_start = line_starts[indices[0]]
            # Leading/trailing whitespace stripped from the body is excluded
            # from the span, so the byte range decodes to ``body`` exactly.
            lead_ws = len(raw_body) - len(raw_body.lstrip())
            start = region_start + len(raw_body[:lead_ws].encode("utf-8"))
            sec["body_start_byte"] = start
            sec["body_end_byte"] = start + len(body.encode("utf-8"))
        else:
            # Empty body: an empty span at the section's OWN position (the line
            # after its heading, or the document start for the level-0
            # preamble) — never at the document end.
            if indices:
                anchor = line_starts[indices[0]]
            elif heading_index is not None and heading_index + 1 < len(line_starts):
                anchor = line_starts[heading_index + 1]
            elif heading_index is not None:
                anchor = line_starts[heading_index] + len(lines[heading_index].encode("utf-8"))
            else:
                anchor = 0
            sec["body_start_byte"] = anchor
            sec["body_end_byte"] = anchor
    return sections


def _section_matches(heading, ancestors, patterns):
    """True if the section heading or any ancestor heading matches a pattern."""
    haystacks = [heading.lower()] + [a.lower() for a in ancestors]
    return any(pat.lower() in h for pat in patterns for h in haystacks)


def filter_sections(sections, allow_patterns=None, drop_sections=None):
    """Apply section allow/deny filtering with heading ancestry.

    - A "Contents" table-of-contents heading is always dropped.
    - ``allow_patterns`` (allowlist): keep a section only if its own or an
      ancestor heading matches — used to slice a large file down to a topic.
    - ``drop_sections`` (denylist): drop a section if its own or an ancestor
      heading matches — used to remove XML-scaffolding sections from an
      otherwise useful file.
    - A section whose body mentions "Tech Preview" is exempt from the *denylist*
      and from the "Contents" rule: once a section is curated in, its warning must
      survive. It overrides ``allow_patterns`` only for the DOCUMENT-LEVEL caveat —
      the FIRST warning-bearing preamble or ``#`` title — see below.

    The Tech-Preview exemption used to short-circuit ``allow_patterns`` for a section
    at any depth, which quietly defeated fail-closed curation: the four Tech-Preview
    ``mcp_*`` upstream docs repeat the banner throughout, so any section of theirs was
    admitted whether or not a curator had named it (a "Changelog" and a "Known
    Limitations" rode in that way, and the next canned-XML section upstream adds would
    have too). Restricting the override to level <= 1 keeps the one thing the rule
    exists for — a document's own Tech-Preview caveat, of which there is exactly one,
    carried by its title section — while a deeper section that merely mentions the
    phrase must now be named by an allow pattern like any other content.

    Note the doc-level caveat CANNOT be readmitted by naming the ``#`` title in
    ``allow_patterns``: the title is an ancestor of every section in the file, so an
    allow pattern matching it would match everything and silently ingest the whole
    document. Hence the level-scoped override rather than a config-level fix.

    ancestry: a section at level L inherits the most recent headings at each
    shallower level as ancestors, so filtering an ``##`` also carries its
    ``###`` children.
    """
    kept = []
    last_by_level = {}
    doc_caveat_taken = False
    for sec in sections:
        level = sec["level"]
        heading = sec["heading"]
        ancestors = [last_by_level[l] for l in sorted(last_by_level) if l < level]
        # Record this heading and forget any deeper ones for later sections.
        last_by_level[level] = heading
        for deeper in [l for l in list(last_by_level) if l > level]:
            del last_by_level[deeper]

        body_lower = sec["body"].lower()
        is_warning = "technology preview" in body_lower or "tech preview" in body_lower

        if heading.strip().lower() == "contents" and not is_warning:
            continue

        # The document's own Tech-Preview caveat — the FIRST warning-bearing preamble
        # or "#" title — always survives. Taking only the first is what makes "exactly
        # one per document" an enforced invariant rather than an assumption: these docs
        # repeat the banner freely, so a second top-level section carrying it would
        # otherwise smuggle its whole body past the allowlist. Every other
        # warning-bearing section is fail-closed like the rest.
        is_doc_caveat = is_warning and level <= 1 and not doc_caveat_taken
        if is_doc_caveat:
            doc_caveat_taken = True

        if allow_patterns and not is_doc_caveat:
            if not _section_matches(heading, ancestors, allow_patterns):
                continue
        if drop_sections and not is_warning:
            if _section_matches(heading, ancestors, drop_sections):
                continue
        kept.append(sec)
    return kept


def strip_xml_blocks(md_text, min_chars=XML_STRIP_MIN_CHARS):
    """Replace oversized raw-XML fenced code blocks with a short placeholder.

    Targets ```xml / ```xsd fences (and untagged fences whose body is clearly a
    raw XML tree) whose content is at or above ``min_chars`` — the low-level
    <bns:Component> serializations. Groovy/JSON/bash fences and small inline XML
    examples are left untouched.
    """
    def _replace(m):
        lang = m.group(3).lower()
        body = m.group(4)
        looks_xml = lang in ("xml", "xsd") or (
            lang == "" and _XML_START_RE.match(body) is not None
        )
        if looks_xml and len(body) >= min_chars:
            return f"{m.group(1)}> _(raw XML component definition omitted — see source)_"
        return m.group(0)

    return _XML_FENCE_RE.sub(_replace, md_text)


# Every field fetch_companion.build_manifest_entry copies out of
# companion_sources.json — i.e. everything in the manifest that came from the curator
# rather than from the fetch itself (raw_url/sha256/bytes are fetch provenance and are
# NOT listed here). chunk_docs reads all of these back from the MANIFEST, so a manifest
# written before a config edit silently chunks the corpus under the old values while
# the build reports success. `sections`/`drop_sections`/`strip_xml_blocks` decide what
# is ingested; `title`/`area` decide how a hit is labelled (they build the chunk title
# and breadcrumb); `priority` is carried for downstream ranking. All of them drift.
#
# The two pattern lists compare as SETS: filter_sections applies them with any(), so
# their order carries no meaning and re-ordering one for readability must not demand a
# re-fetch of all 12 upstream files.
_ORDERED_CONFIG_FIELDS = ("area", "title", "priority", "strip_xml_blocks")
_UNORDERED_CONFIG_FIELDS = ("sections", "drop_sections")

# The TOP-LEVEL fields build_manifest copies from the config. `commit` matters most:
# process_companion reads upstream repo/commit back from the manifest, so bumping the
# pin without re-fetching rebuilds the corpus at the OLD commit and stamps every chunk
# with it — silently undoing the upgrade. That is exactly the drift is_pinned_commit
# exists to make impossible, so it has to be caught here too, not just per file.
_TOPLEVEL_CONFIG_FIELDS = (
    "repo", "repo_url", "commit", "license", "source_type", "verification_status",
    "raw_url_template", "blob_url_template", "latest_blob_url_template",
)


def _curation_of(entry):
    """The config-derived policy of one config/manifest file record, normalized."""
    policy = {
        "area": entry.get("area", ""),
        "title": entry.get("title", ""),
        "priority": entry.get("priority"),
        "strip_xml_blocks": bool(entry.get("strip_xml_blocks", False)),
    }
    for field in _UNORDERED_CONFIG_FIELDS:
        policy[field] = frozenset(entry.get(field) or ())
    return policy


def _show(value):
    """Render a policy value for a drift message (sets back to sorted lists)."""
    return sorted(value) if isinstance(value, frozenset) else value


def curation_drift(config, manifest):
    """Describe every config-vs-manifest difference in curator-owned fields.

    Takes the whole ``companion_sources.json`` and ``companion_manifest.json`` dicts —
    drift lives at BOTH levels: the pinned commit sits at the top, the section filters
    sit per file, and chunking reads both back from the manifest.

    Returns a list of human-readable drift strings (empty when they agree). Callers
    treat a non-empty result as fatal: chunking would apply a stale policy, which fails
    open — the corpus is rebuilt at the previous commit, the sections a curator just
    excluded stay in, a renamed title keeps its old label, and the build says nothing.
    Editing companion_sources.json requires re-running fetch_companion.py so the
    manifest carries the new policy.
    """
    drift = []
    for field in _TOPLEVEL_CONFIG_FIELDS:
        if field not in config:
            continue  # optional in the config; fetch_companion defaults it
        if config[field] != manifest.get(field):
            drift.append(
                f"{field} is {manifest.get(field)!r} in the manifest "
                f"but {config[field]!r} in companion_sources.json"
            )

    cfg = {e["path"]: e for e in config.get("files", [])}
    man = {e["path"]: e for e in manifest.get("files", [])}
    for path in sorted(set(cfg) - set(man)):
        drift.append(f"{path}: in companion_sources.json but not in the manifest")
    for path in sorted(set(man) - set(cfg)):
        drift.append(f"{path}: in the manifest but no longer in companion_sources.json")
    for path in sorted(set(cfg) & set(man)):
        want, have = _curation_of(cfg[path]), _curation_of(man[path])
        for field in _ORDERED_CONFIG_FIELDS + _UNORDERED_CONFIG_FIELDS:
            if want[field] != have[field]:
                drift.append(
                    f"{path}: {field} is {_show(have[field])!r} in the manifest "
                    f"but {_show(want[field])!r} in companion_sources.json"
                )
    return drift


def contains_raw_component_xml(text):
    """True if ``text`` carries a raw component serialization (<bns:Component>
    or a default-namespaced <Component …> root).

    Used as a fail-closed release gate (see build_index.validate_chunks): no
    companion chunk may ship a canned component template, regardless of whether
    it slipped past the size-based strip or an allowlist entry forgot to drop
    its "Component Structure" / "Complete Examples" / "XML Structure" section.
    """
    if not isinstance(text, str):
        return False
    if _NAMESPACED_COMPONENT_RE.search(text):
        return True
    # Default-namespaced <Component …>: inspect real attribute names only,
    # blanking quoted values first so an attr= embedded in a value (or a
    # hyphenated name like data-type) does not spuriously match.
    for attrs in _DEFAULT_COMPONENT_TAG_RE.findall(text):
        if _ROOT_ATTR_RE.search(_QUOTED_VALUE_RE.sub(" ", attrs)):
            return True
    return False
