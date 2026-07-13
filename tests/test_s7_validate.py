"""Tests for s7_validate.py: the S7 synthetic-description validator.

``validate_description`` returns a list of short, prefixed error strings ([]
means valid). Three error classes are kept apart by prefix so a caller can act
on each independently:

* ``shape:``                 structural form (sentence count, WordPiece budget,
                             no fenced code, no XML/tag markup)
* ``mandatory:``             the description must actually name the topic and
                             the chunk's structural anchors (root element,
                             discriminating type value, material prefix)
* ``introduced_identifier:`` every policed-form identifier the description uses
                             must already exist in the source (case-sensitive)

The Set Document Property fixture is data-driven: identifiers and structural
facts are derived from ``RAW_DOCUMENT`` with the real B6 extractors, never
hand-listed, so the fixture and the validator agree by construction. Fake
tokenizer counts (ceil(len/4) pieces per whitespace token) are hand-computable.
"""

import s7_eligibility
import s7_validate
from kb_tokenizer import FakeWordPieceTokenizer

_TOK = FakeWordPieceTokenizer()

# The Map "Set Document Property" function reference: a fenced XML snippet whose
# root element is FunctionStep, carrying type="DocumentPropertySet", a
# dynamicdocument: prefixed property (twice, so the prefix is material), and the
# ProcessProperty / Inputs / Input / Outputs / Configuration / DocumentProperty
# structural vocabulary.
RAW_DOCUMENT = (
    "# Set Document Property\n"
    "\n"
    "The Set Document Property function writes values into document properties "
    "during a Map. It is represented as a FunctionStep.\n"
    "\n"
    "```xml\n"
    '<FunctionStep type="DocumentPropertySet">\n'
    "  <Inputs>\n"
    '    <Input name="value"/>\n'
    "  </Inputs>\n"
    "  <Configuration>\n"
    '    <dynamicdocument:DocumentProperty name="orderId" type="ProcessProperty"/>\n'
    '    <dynamicdocument:DocumentProperty name="orderTotal"/>\n'
    "  </Configuration>\n"
    "  <Outputs>\n"
    "    <Output/>\n"
    "  </Outputs>\n"
    "</FunctionStep>\n"
    "```\n"
)

# Derived with the real B6 extractors — the validator polices against exactly
# these, so the fixture stays self-consistent.
IDENTIFIERS = s7_eligibility.extract_structural_identifiers(RAW_DOCUMENT)
STRUCTURAL_FACTS = s7_eligibility.extract_structural_facts(RAW_DOCUMENT)
TOPIC = "Set Document Property"

# One sentence, well under budget, no fence/markup; names the topic in the first
# sentence plus every mandatory anchor: FunctionStep (root), DocumentPropertySet
# (type value), dynamicdocument (material prefix, colon dropped).
VALID_DESCRIPTION = (
    "A Set Document Property function is represented as a FunctionStep with "
    'type="DocumentPropertySet" that writes a value into a dynamicdocument '
    "property."
)


def _validate(description, *, identifiers=None, raw_document=None):
    return s7_validate.validate_description(
        description,
        raw_document=RAW_DOCUMENT if raw_document is None else raw_document,
        identifiers=IDENTIFIERS if identifiers is None else identifiers,
        topic=TOPIC,
        structural_facts=STRUCTURAL_FACTS,
        tokenizer=_TOK,
    )


# --- fixture sanity ------------------------------------------------------------

def test_max_wordpieces_constant():
    assert s7_validate.MAX_DESCRIPTION_WORDPIECES == 128


def test_fixture_facts_are_as_expected():
    """The data-driven facts the mandatory checks read from must be what we
    reason about below: FunctionStep root, DocumentPropertySet type, and a
    dynamicdocument: prefix that is material (appears twice in the source)."""
    assert STRUCTURAL_FACTS["root_elements"] == ["FunctionStep"]
    assert "DocumentPropertySet" in STRUCTURAL_FACTS["type_values"]
    assert "dynamicdocument:" in STRUCTURAL_FACTS["prefixes"]
    assert RAW_DOCUMENT.count("dynamicdocument:") >= 2


# --- happy path ----------------------------------------------------------------

def test_valid_description_passes():
    assert _validate(VALID_DESCRIPTION) == []


def test_optional_tokens_are_allowed_but_not_required():
    """VALID_DESCRIPTION names none of the optional structural tokens and still
    passes; a description that DOES name them (ProcessProperty via Inputs) also
    passes — they are permitted, never required."""
    with_optional = (
        "A Set Document Property function is represented as a FunctionStep with "
        'type="DocumentPropertySet" that writes a value into a dynamicdocument '
        "property. It reads a ProcessProperty through the Inputs element."
    )
    assert _validate(VALID_DESCRIPTION) == []
    assert _validate(with_optional) == []


def test_identifier_only_in_raw_document_passes():
    """A policed identifier the description reuses need only be a case-sensitive
    substring of raw_document — it does not have to be in the approved
    identifiers list. With an empty identifiers list, FunctionStep and
    DocumentPropertySet are backed solely by the source text and still pass."""
    assert _validate(VALID_DESCRIPTION, identifiers=[]) == []


# --- shape ---------------------------------------------------------------------

def test_empty_description_fails_shape():
    errors = _validate("")
    assert any(err.startswith("shape:") for err in errors)


def test_four_sentences_fails_shape():
    four_sentences = (
        "A Set Document Property function writes a value. It is represented as a "
        "FunctionStep. The type is DocumentPropertySet. It targets a "
        "dynamicdocument property."
    )
    errors = _validate(four_sentences)
    assert any(err.startswith("shape:") and "sentence" in err for err in errors)


def test_code_fence_fails_shape():
    errors = _validate(VALID_DESCRIPTION + "\n```")
    assert any(err.startswith("shape:") and "fenced" in err for err in errors)


def test_xml_markup_fails_shape():
    with_xml = VALID_DESCRIPTION + " See the <FunctionStep> element for details."
    errors = _validate(with_xml)
    assert any(err.startswith("shape:") and "XML" in err for err in errors)


def test_over_wordpiece_budget_fails_shape():
    """Padding with a long lowercase word (not a policed identifier) 30 times at
    ceil(22/4)=6 pieces each adds 180 pieces, so the single sentence blows the
    128-piece budget while every mandatory anchor stays present."""
    over_budget = (
        "A Set Document Property function is a FunctionStep of type "
        '"DocumentPropertySet" targeting a dynamicdocument '
        + "elaborationelaboration " * 30
        + "property."
    )
    assert _TOK.count(over_budget) > 128
    errors = _validate(over_budget)
    assert any(err.startswith("shape:") and "WordPiece" in err for err in errors)


# --- mandatory semantics -------------------------------------------------------

def test_topic_absent_from_first_sentence_fails_mandatory():
    """The topic sits in the second sentence, so the first sentence never names
    it — a mandatory failure distinct from any identifier policing."""
    topic_late = (
        "A function writes a value into a document property. It is a Set "
        'Document Property function represented as a FunctionStep with '
        'type="DocumentPropertySet" targeting a dynamicdocument property.'
    )
    errors = _validate(topic_late)
    assert any(err.startswith("mandatory:") and "topic" in err for err in errors)


def test_missing_type_value_fails_mandatory():
    """Dropping DocumentPropertySet leaves no discriminating type value named,
    so the type mandatory check fails while shape and policing stay clean."""
    missing_type = (
        "A Set Document Property function is represented as a FunctionStep that "
        "writes a value into a dynamicdocument property."
    )
    errors = _validate(missing_type)
    assert any(
        err.startswith("mandatory:") and "type value" in err for err in errors
    )
    assert not any(err.startswith("shape:") for err in errors)


# --- introduced-identifier policing --------------------------------------------

def test_introduced_identifier_fails_policing():
    """FooBar.baz_qux is a policed dotted path absent from source and
    identifiers, so it is reported as introduced."""
    introduces = (
        "A Set Document Property function is represented as a FunctionStep with "
        'type="DocumentPropertySet" that writes a value into a dynamicdocument '
        "property using FooBar.baz_qux."
    )
    errors = _validate(introduces)
    assert any(
        err.startswith("introduced_identifier:") and "FooBar.baz_qux" in err
        for err in errors
    )


def test_identifier_policing_is_case_sensitive():
    """Only "FunctionStep" exists in the source; "functionStep" is a different
    policed token (lower->upper camel) absent case-sensitively from both the
    source text and the identifiers, so it is policed as introduced even though
    the mandatory root check (casefold) is satisfied."""
    wrong_case = (
        "A Set Document Property function is represented as a functionStep with "
        'type="DocumentPropertySet" that writes a value into a dynamicdocument '
        "property."
    )
    errors = _validate(wrong_case)
    assert any(
        err.startswith("introduced_identifier:") and "functionStep" in err
        for err in errors
    )


def test_dotted_abbreviation_is_policed():
    """Known extractor quirk: extract_structural_identifiers matches a dotted
    abbreviation like "e.g" as a dotted path. The validator does not special-
    case English abbreviations, so an unsourced "e.g." is policed. Generated
    descriptions must therefore avoid such abbreviations."""
    with_abbrev = (
        "A Set Document Property function is represented as a FunctionStep with "
        'type="DocumentPropertySet" that writes a value into a dynamicdocument '
        "property, e.g. an order total."
    )
    errors = _validate(with_abbrev)
    assert any(
        err.startswith("introduced_identifier:") and "e.g" in err
        for err in errors
    )
