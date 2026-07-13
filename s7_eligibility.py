#!/usr/bin/env python3
"""S7 eligibility gate + structural identifier extraction for Companion chunks.

S7 synthetic descriptions are generated ONLY for ``companion_reference``
chunks — official chunks are never eligible — and only when the chunk both:

1. is code-dominated: at least :data:`MIN_CODE_WORDPIECE_RATIO` of its
   WordPieces sit inside fenced code blocks or inline single-backtick code
   spans, and
2. carries anchors: at least :data:`MIN_STRUCTURAL_IDENTIFIERS` structural
   identifiers of the policed forms below, which the S7 validator later uses
   to police identifiers a generated description introduces.

Fence handling follows companion.py's discipline (:func:`companion._fence_marker`):
a fence opens with a run of 3+ backticks or tildes and closes only with a
matching marker — same character, >= length, no info string. The synthetic
``fence_open`` lines the S5 splitter prepends to continuation fragments are
ordinary fence openers under that discipline, so a split code block's content
keeps counting as code on every fragment.

Policed identifier forms (case-sensitive): dotted paths
(``document.dynamic.userdefined``), underscore-delimited tokens
(``DDP_COUNTER``), mixed-case tokens with an internal lowercase->uppercase
transition (``dataContext``, ``DocumentPropertySet``), and — from XML-ish
markup only — element names including their namespace prefix
(``bns:Function``), namespace prefixes in use (``bns:``, ``xsi:``), and the
values of ``type=`` / ``xsi:type=`` attributes. Plain prose words and ordinary
single title-case words (``Boomi``, ``The``, ``Integration``) never qualify:
they have no internal transition, no separator, and no surrounding markup.
"""
import re

import companion

MIN_CODE_WORDPIECE_RATIO = 0.3
MIN_STRUCTURAL_IDENTIFIERS = 2

# Inline code: a single-backtick span on one line. Fenced regions are carved
# out line-wise first, so this only ever runs on non-fenced lines.
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")

# The word-shaped policed forms, verbatim from the plan. A single title-case
# word (Boomi, Integration) matches none of them: no dot, no underscore, and
# no SECOND uppercase after the internal lowercase run.
_DOTTED_PATH_RE = re.compile(r"\b[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+\b")
_UNDERSCORE_RE = re.compile(r"\b[A-Za-z]+(?:_[A-Za-z0-9]+)+\b")
_LOWER_CAMEL_RE = re.compile(r"\b[a-z]+[A-Z][A-Za-z0-9]*\b")
_UPPER_CAMEL_RE = re.compile(r"\b[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*\b")
_WORD_FORM_RES = (_DOTTED_PATH_RE, _UNDERSCORE_RE, _LOWER_CAMEL_RE, _UPPER_CAMEL_RE)

# An XML-ish tag. The name must start with a NameStartChar (letter/underscore)
# so declarations (<?xml), comments (<!--), DOCTYPE, and prose "a < b" never
# match. The attribute region consumes whole quoted values of either style so
# a ">" inside a value cannot end the tag early (the same parse companion's
# component-root gate uses).
_TAG_RE = re.compile(
    r"<(/?)"
    r"([A-Za-z_][\w.-]*(?::[A-Za-z_][\w.-]*)?)"
    r"((?:[^>\"']|\"[^\"]*\"|'[^']*')*?)"
    r"(/?)>"
)
# A structural type attribute. The (?:^|\s) guard keeps subtype=/data-type=
# from matching, and keeps the plain-"type" arm from re-matching inside
# "xsi:type" (":" is neither start nor whitespace).
_TYPE_VALUE_RE = re.compile(
    r"(?:^|\s)(?:xsi:type|type)\s*=\s*(?:\"([^\"]*)\"|'([^']*)')"
)
# Attribute NAMES are scanned only after quoted values are blanked out, so a
# colon-and-equals sequence inside a value can never mint a fake prefix.
_QUOTED_VALUE_RE = re.compile(r"\"[^\"]*\"|'[^']*'")
_PREFIXED_ATTR_RE = re.compile(r"(?:^|\s)([A-Za-z_][\w.-]*):([A-Za-z_][\w.-]*)\s*=")


# --- code WordPiece ratio -------------------------------------------------------

def _code_segments(raw_document):
    """Yield the code portions of a Markdown document as text segments.

    Fenced blocks are yielded whole INCLUDING their fence lines: an all-code
    document is exactly 100% code, and S5's synthetic fence wrappers count
    with the code they wrap. A fence closes only per companion's discipline
    (matching char, >= length, no info string); an unclosed trailing fence
    runs to end of document, so a fragment whose close was lost still counts
    its code. Inline single-backtick spans on non-fenced lines are yielded
    individually.
    """
    fence_lines = []
    open_fence = None  # (char, length) while inside a fenced code block
    for line in raw_document.split("\n"):
        marker = companion._fence_marker(line)
        if open_fence is not None:
            fence_lines.append(line)
            if (marker and marker[0] == open_fence[0]
                    and marker[1] >= open_fence[1]
                    and line.strip()[marker[1]:].strip() == ""):
                yield "\n".join(fence_lines)
                fence_lines = []
                open_fence = None
            continue
        if marker:
            open_fence = marker
            fence_lines = [line]
            continue
        yield from _INLINE_CODE_RE.findall(line)
    if fence_lines:
        yield "\n".join(fence_lines)


def code_wordpiece_ratio(raw_document, tokenizer):
    """Share of the document's WordPieces that are code.

    (pieces inside fenced code blocks, fence lines included, + pieces in
    inline single-backtick spans) / total pieces of the whole document.
    0.0 when the document has no pieces. Clamped to 1.0 — segment-wise
    counting can slightly exceed the whole-document count when inline spans
    abut without whitespace between them.
    """
    total = tokenizer.count(raw_document)
    if total == 0:
        return 0.0
    code = sum(tokenizer.count(segment) for segment in _code_segments(raw_document))
    return min(1.0, code / total)


# --- structural identifiers / facts ---------------------------------------------

def _iter_tags(text):
    """Yield ``(kind, name, attrs)`` for every XML-ish tag in the text.

    ``kind`` is ``"open"``, ``"close"``, or ``"selfclose"``. Names keep their
    namespace prefix (``bns:QueryConfig``).
    """
    for match in _TAG_RE.finditer(text):
        closing, name, attrs, selfclose = match.groups()
        if closing:
            yield "close", name, attrs
        elif selfclose:
            yield "selfclose", name, attrs
        else:
            yield "open", name, attrs


def _attr_type_values(attrs):
    """Values of ``type=`` / ``xsi:type=`` attributes, in attribute order."""
    values = []
    for double_quoted, single_quoted in _TYPE_VALUE_RE.findall(attrs):
        value = double_quoted or single_quoted
        if value:
            values.append(value)
    return values


def _attr_prefixes(attrs):
    """Namespace prefixes used or declared by an attribute region.

    A prefixed attribute name contributes its own prefix (``xsi:type`` ->
    ``xsi:``); an ``xmlns:NAME`` declaration contributes the DECLARED prefix
    (``xmlns:bns`` -> ``bns:``), never the reserved ``xmlns:`` itself.
    """
    blanked = _QUOTED_VALUE_RE.sub('""', attrs)
    prefixes = []
    for prefix, local in _PREFIXED_ATTR_RE.findall(blanked):
        prefixes.append(f"{local}:" if prefix == "xmlns" else f"{prefix}:")
    return prefixes


def extract_structural_identifiers(text):
    """Sorted unique identifiers of the policed, case-sensitive forms.

    Word-shaped forms (dotted paths, underscore tokens, both camel-case
    shapes) are matched anywhere in the text; markup additionally contributes
    element names including their prefix, every namespace prefix in use, and
    ``type=`` / ``xsi:type=`` attribute values. Prose words and ordinary
    single title-case words fall through every form and are never returned.
    """
    found = set()
    for pattern in _WORD_FORM_RES:
        found.update(pattern.findall(text))
    for _kind, name, attrs in _iter_tags(text):
        found.add(name)
        if ":" in name:
            found.add(name.split(":", 1)[0] + ":")
        found.update(_attr_prefixes(attrs))
        found.update(_attr_type_values(attrs))
    return sorted(found)


def extract_structural_facts(raw_document):
    """Structural facts from the document's XML-ish content.

    ``root_elements``: the top (depth-0) element name of each XML snippet, in
    document order — sibling snippets each contribute their own root.
    ``type_values``: every ``type=`` / ``xsi:type=`` attribute value at any
    depth. ``prefixes``: namespace prefixes in use (element names, prefixed
    attribute names, ``xmlns:NAME`` declarations), each with its trailing
    colon. All three lists deduplicate preserving first appearance, so
    ``root_elements[0]`` is the first root seen.
    """
    facts = {"root_elements": [], "type_values": [], "prefixes": []}
    depth = 0
    for kind, name, attrs in _iter_tags(raw_document):
        if kind == "close":
            depth = max(0, depth - 1)
            continue
        if depth == 0:
            _add_unique(facts["root_elements"], name)
        if ":" in name:
            _add_unique(facts["prefixes"], name.split(":", 1)[0] + ":")
        for prefix in _attr_prefixes(attrs):
            _add_unique(facts["prefixes"], prefix)
        for value in _attr_type_values(attrs):
            _add_unique(facts["type_values"], value)
        if kind == "open":
            depth += 1
    return facts


def _add_unique(values, value):
    """Append preserving first-appearance order (list-backed dedup)."""
    if value not in values:
        values.append(value)


# --- eligibility -----------------------------------------------------------------

def is_eligible(chunk, tokenizer):
    """True when a chunk may receive an S7 synthetic description.

    A strict conjunction: ``source_type`` is companion_reference (official
    chunks are NEVER eligible, however code-heavy), the chunk content's code
    WordPiece ratio is >= :data:`MIN_CODE_WORDPIECE_RATIO` (inclusive), and
    the content carries >= :data:`MIN_STRUCTURAL_IDENTIFIERS` policed-form
    structural identifiers.
    """
    if chunk["source_type"] != companion.COMPANION_SOURCE_TYPE:
        return False
    content = chunk["content"]
    if code_wordpiece_ratio(content, tokenizer) < MIN_CODE_WORDPIECE_RATIO:
        return False
    return len(extract_structural_identifiers(content)) >= MIN_STRUCTURAL_IDENTIFIERS
