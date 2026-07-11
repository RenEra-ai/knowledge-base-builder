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
_COMPONENT_ROOT_RE = re.compile(
    r"<[A-Za-z][\w.-]*:Component(?=[\s/>])"                      # namespaced root
    r"|<Component(?=[\s/>])[^>]*\b(?:componentId|xmlns|type)\s*="  # default-ns root w/ attr
)

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
    lines = md_text.split("\n")
    sections = []
    current = {"level": 0, "heading": "", "body_lines": []}
    open_fence = None  # (char, length) while inside a fenced code block

    for line in lines:
        marker = _fence_marker(line)

        if open_fence is not None:
            # Inside a fence: only a matching fence (same char, >= length, no
            # trailing info string) closes it; headings never split here.
            if (marker and marker[0] == open_fence[0]
                    and marker[1] >= open_fence[1]
                    and line.strip()[marker[1]:].strip() == ""):
                open_fence = None
            current["body_lines"].append(line)
            continue

        if marker:
            # Opening fence — body content, not a split point.
            open_fence = marker
            current["body_lines"].append(line)
            continue

        match = _HEADING_RE.match(line)
        if match:
            # Flush the section in progress before starting a new one.
            if current["heading"] or "".join(current["body_lines"]).strip():
                sections.append(current)
            current = {
                "level": len(match.group(1)),
                "heading": _clean_heading(match.group(2)),
                "body_lines": [],
            }
        else:
            current["body_lines"].append(line)

    if current["heading"] or "".join(current["body_lines"]).strip():
        sections.append(current)

    for sec in sections:
        sec["body"] = "\n".join(sec.pop("body_lines")).strip()
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
    - A section whose body mentions "Tech Preview" is never dropped (its warning
      must survive), regardless of the filters.

    ancestry: a section at level L inherits the most recent headings at each
    shallower level as ancestors, so filtering an ``##`` also carries its
    ``###`` children.
    """
    kept = []
    last_by_level = {}
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

        if is_warning:
            kept.append(sec)
            continue

        if allow_patterns:
            if not _section_matches(heading, ancestors, allow_patterns):
                continue
        if drop_sections:
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
    return _COMPONENT_ROOT_RE.search(text) is not None
