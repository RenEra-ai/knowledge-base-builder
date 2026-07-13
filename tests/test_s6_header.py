"""Tests for s6_header.py: candidate lines per source_type, casefolded
whitespace-normalized dedup against the fragment's first raw line (display
text preserved), the 64-token cap that drops leftmost breadcrumb segments
first, and HeaderCoreTooLongError when the title alone cannot fit.

All token counts are hand-computed under FakeWordPieceTokenizer: each
whitespace-separated token costs ceil(len/4) pieces (">" costs 1).
"""

import pytest

import s6_header
from kb_tokenizer import FakeWordPieceTokenizer

TOK = FakeWordPieceTokenizer()


def _build(**overrides):
    args = dict(
        source_type="companion_reference",
        title="Map Component Functions",
        breadcrumb="Companion Reference > Components > Map Component",
        raw_first_line="Function Categories",
        tokenizer=TOK,
    )
    args.update(overrides)
    return s6_header.build_s6_header(**args)


# --- candidate lines per source_type -------------------------------------------

def test_official_header_is_title_only():
    """Official pages get no breadcrumb line even when one is supplied."""
    header = _build(source_type="official")
    assert header.lines == ("Map Component Functions",)
    assert header.text == "Map Component Functions"
    # Map=1 + Component=3 + Functions=3.
    assert header.token_count == 7


def test_companion_header_is_breadcrumb_then_title():
    header = _build()
    assert header.lines == (
        "Companion Reference > Components > Map Component",
        "Map Component Functions",
    )
    assert header.text == (
        "Companion Reference > Components > Map Component"
        "\nMap Component Functions"
    )
    # Breadcrumb 3+3+1+3+1+1+3 = 15; title 1+3+3 = 7; "\n" is free whitespace.
    assert header.token_count == 22


def test_unknown_source_type_is_rejected():
    with pytest.raises(ValueError, match="unknown source_type"):
        _build(source_type="wiki")


# --- dedup rules (display text preserved) --------------------------------------

def test_breadcrumb_final_segment_matching_title_is_dropped_display_preserved():
    """Rule (a): the final segment dedups against the title case- and
    whitespace-insensitively, while the SURVIVING display text (the double
    space inside the first segment) is untouched."""
    header = _build(
        breadcrumb="Companion   Reference > Components > MAP  component FUNCTIONS",
    )
    assert header.lines == (
        "Companion   Reference > Components",
        "Map Component Functions",
    )


def test_raw_first_line_repeating_title_omits_title_line():
    """Rule (b): the section heading IS the title, so the title line would be
    an exact repetition of the first raw-content line."""
    header = _build(raw_first_line="Map Component Functions")
    assert header.lines == ("Companion Reference > Components > Map Component",)
    assert "Map Component Functions" not in header.text


def test_raw_first_line_extending_title_still_omits_title_line():
    """Rule (b) is a prefix match: a heading like '<title> Overview' already
    states the title."""
    header = _build(raw_first_line="MAP COMPONENT FUNCTIONS Overview")
    assert header.lines == ("Companion Reference > Components > Map Component",)


def test_raw_first_line_matching_breadcrumb_drops_that_line():
    """Rule (c): ANY header line the raw first line already states is
    dropped, not just the title."""
    header = _build(
        breadcrumb="Connectors",
        title="HTTP Client Operation",
        raw_first_line="Connectors and Listeners",
    )
    assert header.lines == ("HTTP Client Operation",)


def test_adjacent_duplicate_lines_collapse_keeping_first_display():
    """Rule (d): lines that normalize equal collapse to one, keeping the
    first occurrence's display text."""
    header = _build(
        breadcrumb="Companion Reference > Glossary",
        title="companion  reference > GLOSSARY",
    )
    assert header.lines == ("Companion Reference > Glossary",)


def test_fully_deduplicated_header_is_legal_and_empty():
    """A section-heading-equals-title chunk dedups everything away; the empty
    header is legal, not an error."""
    header = _build(
        source_type="official",
        raw_first_line="Map Component Functions",
    )
    assert header == s6_header.S6Header(lines=(), text="", token_count=0)
    # Companion variant: single-segment breadcrumb == title (rule a) and the
    # heading repeats the title (rule b) -> nothing survives.
    header = _build(
        breadcrumb="Map Component Functions",
        raw_first_line="Map Component Functions",
    )
    assert header.lines == () and header.text == "" and header.token_count == 0


# --- token cap ------------------------------------------------------------------

def test_cap_drops_leftmost_breadcrumb_segments_keeping_rightmost():
    """Full header is 14 tokens (Alpha=2 + 3x'>'=3 + Beta=1 + Gamma=2 +
    Delta=2 + Epsilon=2 + Function=2); cap 12 forces exactly one drop, and it
    is the LEFTMOST segment — the rightmost, most specific ones survive."""
    header = _build(
        breadcrumb="Alpha > Beta > Gamma > Delta",
        title="Epsilon Function",
        cap=12,
    )
    assert header.text == "Beta > Gamma > Delta\nEpsilon Function"
    assert header.token_count == 11


def test_cap_can_drop_the_whole_breadcrumb_before_touching_the_title():
    """Even the last segment is droppable; the title is never shrunk. Cap 4
    fits exactly the title (Epsilon=2 + Function=2)."""
    header = _build(
        breadcrumb="Alpha > Beta > Gamma > Delta",
        title="Epsilon Function",
        cap=4,
    )
    assert header.lines == ("Epsilon Function",)
    assert header.token_count == 4


def test_title_alone_over_default_cap_raises_header_core_too_long():
    """22 x 'identifier' = 66 pieces > HEADER_TOKEN_CAP; the error carries the
    offending title and its token count."""
    assert s6_header.HEADER_TOKEN_CAP == 64
    title = " ".join(["identifier"] * 22)
    with pytest.raises(
        s6_header.HeaderCoreTooLongError, match="header core too long"
    ) as excinfo:
        _build(source_type="official", title=title)
    assert excinfo.value.title == title
    assert excinfo.value.token_count == 66


def test_companion_exhausts_breadcrumb_then_raises_on_oversized_title():
    """Shrinking is tried first: both breadcrumb segments go (4 then 2
    tokens), but the 13-piece title alone still exceeds cap 8."""
    title = "Interminable Documentation Title Concatenation"
    with pytest.raises(
        s6_header.HeaderCoreTooLongError, match="header core too long"
    ) as excinfo:
        _build(breadcrumb="Area > Topic", title=title, cap=8)
    assert excinfo.value.title == title
    assert excinfo.value.token_count == 13


# --- token accounting -----------------------------------------------------------

def test_token_count_is_the_tokenizer_count_of_the_exact_text():
    """token_count is always the tokenizer's count of exactly `text` — the
    string the pipeline will prepend to the embedding input."""
    for header in (
        _build(),
        _build(source_type="official"),
        _build(breadcrumb="Alpha > Beta > Gamma > Delta",
               title="Epsilon Function", cap=12),
        _build(source_type="official",
               raw_first_line="Map Component Functions"),
    ):
        assert header.token_count == TOK.count(header.text)
