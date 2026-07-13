"""Tests for s7_eligibility.py: the S7 gate for Companion chunks.

Eligibility is a strict conjunction — source_type companion_reference AND a
code WordPiece ratio >= 0.3 AND >= 2 policed structural identifiers. Every
expected ratio in this file is hand-computed against FakeWordPieceTokenizer
(ceil(len/4) pieces per whitespace-separated token), so the numbers here are
derivable by hand, never snapshot output.
"""

import s7_eligibility
from kb_tokenizer import FakeWordPieceTokenizer

_TOK = FakeWordPieceTokenizer()

# A companion fragment that clears both content gates: fenced Groovy dominates
# the piece count and the code carries several policed identifiers.
_ELIGIBLE_CONTENT = (
    "## Set properties\n"
    "```groovy\n"
    "dataContext.storeStream(is, props)\n"
    "props.setProperty(DDP_COUNTER)\n"
    "```"
)


def _chunk(**overrides):
    chunk = {
        "id": "companion_map_functions_0123456789ab_000",
        "content": _ELIGIBLE_CONTENT,
        "source_type": "companion_reference",
    }
    chunk.update(overrides)
    return chunk


# --- thresholds ----------------------------------------------------------------

def test_thresholds_pin_the_s7_gate():
    assert s7_eligibility.MIN_CODE_WORDPIECE_RATIO == 0.3
    assert s7_eligibility.MIN_STRUCTURAL_IDENTIFIERS == 2


# --- code_wordpiece_ratio -------------------------------------------------------

def test_ratio_is_one_for_all_code_document():
    """A document that is entirely one fenced block is 100% code — the fence
    marker lines belong to the block, not to prose."""
    doc = "```groovy\ndef props = dataContext\n```"
    assert s7_eligibility.code_wordpiece_ratio(doc, _TOK) == 1.0


def test_ratio_is_zero_for_document_without_code():
    doc = "Maps set dynamic document properties at run time."
    assert s7_eligibility.code_wordpiece_ratio(doc, _TOK) == 0.0


def test_ratio_is_zero_for_document_with_no_pieces():
    """No pieces means ratio 0.0, never a ZeroDivisionError."""
    assert s7_eligibility.code_wordpiece_ratio("", _TOK) == 0.0
    assert s7_eligibility.code_wordpiece_ratio("  \n\t ", _TOK) == 0.0


def test_ratio_mid_hand_computed_with_fake_counts():
    """Prose 8 pieces (Read=1 the=1 stream=2 then=1 store=2 it.=1) + fenced
    code 8 pieces (```groovy=3, def x = 1 -> 4, ```=1) -> exactly 0.5."""
    doc = "Read the stream then store it.\n```groovy\ndef x = 1\n```"
    assert s7_eligibility.code_wordpiece_ratio(doc, _TOK) == 8 / 16


def test_inline_code_spans_count_toward_ratio():
    """A single-backtick span on one line is code: Call=1 `storeStream`=4
    here.=2 -> total 7, code 4."""
    doc = "Call `storeStream` here."
    assert s7_eligibility.code_wordpiece_ratio(doc, _TOK) == 4 / 7


def test_synthetically_reopened_fence_counts_as_code():
    """A continuation fragment from the S5 splitter starts with the repeated
    section heading and a synthetic fence_open (same marker + language) so it
    is valid fenced Markdown — its fenced content must count as code.

    Heading pieces: ##=1 Groovy=2 example=2 -> 5. Code: ```groovy=3,
    props.setProperty(name) (23 chars) = 6, ```=1 -> 10. Ratio 10/15."""
    doc = "## Groovy example\n```groovy\nprops.setProperty(name)\n```"
    assert s7_eligibility.code_wordpiece_ratio(doc, _TOK) == 10 / 15


def test_tilde_fence_not_closed_by_backtick_line():
    """companion.py fence discipline: only a matching marker (same char,
    >= length, no info string) closes a fence, so a ``` line inside a ~~~
    block is still fenced code. Code: ~~~=1 code=1 line=1 ```=1 still=2
    code=1 ~~~=1 -> 8; prose tail: prose=2 tail=1 here=1 -> 4."""
    doc = "~~~\ncode line\n```\nstill code\n~~~\nprose tail here"
    assert s7_eligibility.code_wordpiece_ratio(doc, _TOK) == 8 / 12


# --- extract_structural_identifiers ---------------------------------------------

def test_extracts_dotted_path():
    ids = s7_eligibility.extract_structural_identifiers(
        "Set document.dynamic.userdefined before the map runs."
    )
    assert ids == ["document.dynamic.userdefined"]


def test_extracts_underscore_token():
    ids = s7_eligibility.extract_structural_identifiers(
        "Increment DDP_COUNTER for each document."
    )
    assert ids == ["DDP_COUNTER"]


def test_extracts_mixed_case_tokens():
    """Both camel forms qualify — an internal lowercase->uppercase transition
    is the tell (dataContext, storeStream, DocumentPropertySet, FunctionStep);
    the surrounding prose words are not identifiers."""
    ids = s7_eligibility.extract_structural_identifiers(
        "The dataContext exposes storeStream while DocumentPropertySet "
        "backs FunctionStep."
    )
    assert ids == [
        "DocumentPropertySet", "FunctionStep", "dataContext", "storeStream",
    ]


def test_extracts_xml_names_prefixes_and_type_values():
    """Markup yields the element name including its namespace prefix, the
    prefixes in use (element and attribute), and type=/xsi:type= values.
    bns:Function and execute prove the markup arms: neither matches any
    word-shaped form on its own."""
    ids = s7_eligibility.extract_structural_identifiers(
        'The step emits <bns:Function type="execute" '
        'xsi:type="DocumentPropertySet"/> at run time.'
    )
    assert ids == [
        "DocumentPropertySet", "bns:", "bns:Function", "execute", "xsi:",
    ]


def test_prose_and_title_case_words_are_not_identifiers():
    """Ordinary prose and single title-case words are exempt: Boomi, The, and
    Integration have no internal transition, separator, or markup."""
    ids = s7_eligibility.extract_structural_identifiers(
        "The Boomi Integration platform maps documents."
    )
    assert ids == []


def test_identifiers_are_sorted_and_unique():
    ids = s7_eligibility.extract_structural_identifiers(
        "Use dataContext and dataContext with DDP_COUNTER."
    )
    assert ids == ["DDP_COUNTER", "dataContext"]


# --- extract_structural_facts ---------------------------------------------------

def test_structural_facts_from_xml_snippet():
    """The snippet's depth-0 element is the root (nested QueryFilter is not);
    xsi:type values and every prefix in use (element names, prefixed
    attribute names, xmlns declarations) are collected."""
    doc = (
        "Query config skeleton:\n"
        "```xml\n"
        '<bns:QueryConfig xmlns:bns="http://api.platform.boomi.com/">\n'
        '  <bns:QueryFilter xsi:type="GroupingExpression" operator="and"/>\n'
        "</bns:QueryConfig>\n"
        "```\n"
    )
    facts = s7_eligibility.extract_structural_facts(doc)
    assert facts["root_elements"] == ["bns:QueryConfig"]
    assert facts["type_values"] == ["GroupingExpression"]
    assert facts["prefixes"] == ["bns:", "xsi:"]


def test_structural_facts_empty_without_markup():
    facts = s7_eligibility.extract_structural_facts(
        "Maps move data between profiles."
    )
    assert facts == {"root_elements": [], "type_values": [], "prefixes": []}


def test_structural_facts_multiple_sibling_roots():
    """Two self-closing top-level elements are two roots, in document order."""
    facts = s7_eligibility.extract_structural_facts(
        '<Map version="2"/>\n<Profile/>'
    )
    assert facts["root_elements"] == ["Map", "Profile"]
    assert facts["type_values"] == []
    assert facts["prefixes"] == []


# --- is_eligible ----------------------------------------------------------------

def test_companion_chunk_with_code_and_identifiers_is_eligible():
    assert s7_eligibility.is_eligible(_chunk(), _TOK)


def test_official_chunk_never_eligible_even_with_heavy_code():
    """source_type gates first: identical all-code, identifier-rich content is
    ineligible the moment the chunk is official."""
    assert not s7_eligibility.is_eligible(_chunk(source_type="official"), _TOK)


def test_low_code_ratio_chunk_ineligible():
    """Identifier-rich pure prose fails on the ratio arm alone."""
    content = "The dataContext and DDP_COUNTER properties drive the mapping step."
    assert len(s7_eligibility.extract_structural_identifiers(content)) >= 2
    assert not s7_eligibility.is_eligible(_chunk(content=content), _TOK)


def test_chunk_with_too_few_identifiers_ineligible():
    """All-code content (ratio 1.0) with a single identifier fails on the
    identifier arm alone."""
    content = "```groovy\ndataContext\n```"
    assert s7_eligibility.code_wordpiece_ratio(content, _TOK) == 1.0
    assert s7_eligibility.extract_structural_identifiers(content) == ["dataContext"]
    assert not s7_eligibility.is_eligible(_chunk(content=content), _TOK)


def test_eligibility_at_exact_ratio_threshold():
    """The ratio gate is inclusive: prose 7 pieces (dataContext=3
    DDP_COUNTER=3 x.=1) + code 3 pieces (```=1 x=1 ```=1) -> exactly 0.3,
    with exactly the 2 required identifiers."""
    content = "dataContext DDP_COUNTER x.\n```\nx\n```"
    assert s7_eligibility.code_wordpiece_ratio(content, _TOK) == 3 / 10
    assert s7_eligibility.is_eligible(_chunk(content=content), _TOK)
