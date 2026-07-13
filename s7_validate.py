#!/usr/bin/env python3
"""S7 synthetic-description validator for Companion chunks.

A generated description is only usable when it is well-formed, actually names
the chunk's structural anchors, and introduces no identifier that is not already
in the source. :func:`validate_description` returns a list of short, prefixed
error strings — ``[]`` means valid — so a caller (the S7 generation retry loop)
can accumulate and classify every problem at once:

* ``shape:``                 1-3 sentences, <= :data:`MAX_DESCRIPTION_WORDPIECES`
                             WordPieces, no fenced code block (``` or ~~~), no
                             XML/tag-like markup.
* ``mandatory:``             the normalized ``topic`` appears in the FIRST
                             sentence; when the chunk exposes them, a root
                             element, a discriminating type value, and a
                             material namespace prefix are each named somewhere
                             in the description.
* ``introduced_identifier:`` every policed-form identifier the description uses
                             (same extractor as :mod:`s7_eligibility`) is present
                             CASE-SENSITIVELY in the approved ``identifiers`` or
                             as a substring of ``raw_document``.

Mandatory-semantics failures and introduced-identifier failures are deliberately
distinct error classes: the first says the description omitted something it had
to say, the second says it said something it was not allowed to invent.

Mandatory "mentioned" checks are casefold-containment (the description may
naturally lowercase a name in prose), whereas introduced-identifier policing is
case-sensitive by design — a case variant such as ``functionStep`` where the
source only has ``FunctionStep`` is a different, unsourced token and is policed.

Extractor quirk: ``extract_structural_identifiers`` matches a dotted abbreviation
like ``e.g`` as a dotted-path identifier. This validator does not special-case
English abbreviations, so a description containing ``e.g.`` (with the fragment
``e.g`` absent from the source) is policed as introduced — generated text must
avoid such abbreviations rather than the validator silently exempting them.
"""
import re

import s7_eligibility

MAX_DESCRIPTION_WORDPIECES = 128

# Sentence boundary: end punctuation followed by whitespace. Empty fragments
# (from trailing/leading whitespace) are dropped by the caller.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# Tag-like markup: "<" immediately followed by a name-start char, a slash, or a
# declaration/PI marker. Prose "a < b" (space after "<") never matches.
_XML_TAG_RE = re.compile(r"<[A-Za-z_/!?]")


def _sentences(description):
    """Non-empty sentences of ``description``, split on end punctuation."""
    return [s for s in _SENTENCE_SPLIT_RE.split(description) if s.strip()]


def validate_description(description, *, raw_document, identifiers, topic,
                         structural_facts, tokenizer):
    """Validate an S7 description; return prefixed error strings ([] == valid).

    All three check families run and accumulate — a description can fail shape,
    mandatory semantics, and identifier policing independently, and a caller
    benefits from seeing every failure in one pass.
    """
    errors = []
    errors.extend(_shape_errors(description, tokenizer))
    errors.extend(_mandatory_errors(description, raw_document, topic, structural_facts))
    errors.extend(_introduced_identifier_errors(description, raw_document, identifiers))
    return errors


def _shape_errors(description, tokenizer):
    """Structural form: sentence count, WordPiece budget, no fence, no markup."""
    errors = []
    count = len(_sentences(description))
    if not 1 <= count <= 3:
        errors.append(f"shape: expected 1-3 sentences, found {count}")
    pieces = tokenizer.count(description)
    if pieces > MAX_DESCRIPTION_WORDPIECES:
        errors.append(
            f"shape: description is {pieces} WordPieces, over the "
            f"{MAX_DESCRIPTION_WORDPIECES} limit"
        )
    if "```" in description or "~~~" in description:
        errors.append("shape: description contains a fenced code block")
    if _XML_TAG_RE.search(description):
        errors.append("shape: description contains XML/tag markup")
    return errors


def _mandatory_errors(description, raw_document, topic, structural_facts):
    """Mandatory semantics: topic in first sentence; anchors named when present.

    Each of the four sub-checks emits its own error string. "Mentioned" is
    casefold containment; a material prefix may be named with or without its
    trailing colon. Materiality means the colon-suffixed prefix occurs at least
    twice in ``raw_document`` — a one-off prefix is not worth demanding.
    """
    errors = []
    description_cf = description.casefold()
    sentences = _sentences(description)
    first_sentence_cf = sentences[0].casefold() if sentences else ""

    if topic.casefold() not in first_sentence_cf:
        errors.append(f"mandatory: topic {topic!r} is not in the first sentence")

    roots = structural_facts.get("root_elements") or []
    if roots and not any(root.casefold() in description_cf for root in roots):
        errors.append(
            f"mandatory: no root element ({', '.join(roots)}) is mentioned"
        )

    type_values = structural_facts.get("type_values") or []
    if type_values and not any(
        value.casefold() in description_cf for value in type_values
    ):
        errors.append(
            "mandatory: no discriminating type value "
            f"({', '.join(type_values)}) is mentioned"
        )

    material_prefixes = [
        prefix
        for prefix in structural_facts.get("prefixes") or []
        if raw_document.count(prefix) >= 2
    ]
    if material_prefixes and not any(
        prefix.casefold() in description_cf
        or prefix.rstrip(":").casefold() in description_cf
        for prefix in material_prefixes
    ):
        errors.append(
            "mandatory: no material namespace prefix "
            f"({', '.join(material_prefixes)}) is mentioned"
        )
    return errors


def _introduced_identifier_errors(description, raw_document, identifiers):
    """Policing: every policed identifier in the description must be sourced.

    Runs the B6 extractor over the description and requires each token to be a
    case-sensitive member of ``identifiers`` OR a case-sensitive substring of
    ``raw_document``. The extractor already exempts prose and single title-case
    words (they match no policed form), so those never reach this check.
    """
    approved = set(identifiers)
    errors = []
    for token in s7_eligibility.extract_structural_identifiers(description):
        if token in approved or token in raw_document:
            continue
        errors.append(f"introduced_identifier: {token!r} is not present in the source")
    return errors
