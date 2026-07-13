"""Tests for s5_splitter.py: budget-driven fragmentation with exact UTF-8
source spans, fence-preserving code splits, and fail-closed unsplittable spans.

Every splitting scenario funnels through ``_split()``, which enforces the two
module-wide invariants after every split: ``assert_spans_tile`` (ordered spans
cover ``[base_offset, base_offset + len(payload utf-8))`` with no gap and no
overlap) and byte-exact reconstruction of the payload from the spans alone —
so no test can pass while a source byte is dropped, duplicated, or a span
boundary bisects a multi-byte codepoint (strict per-span UTF-8 decode).

Token counts use FakeWordPieceTokenizer: each whitespace-separated token costs
ceil(len(token)/4) pieces, so budgets below are hand-computed against that.
"""

import pytest

import companion
import s5_splitter
from kb_tokenizer import FakeWordPieceTokenizer
from s5_splitter import Fragment, SourceSpan, UnsplittableSpanError

_TOK = FakeWordPieceTokenizer()


# --- shared helpers -------------------------------------------------------------

def _reconstruct(payload, base_offset, fragments):
    """Rebuild the payload from ordered span byte slices — must be byte-exact."""
    data = payload.encode("utf-8")
    parts = []
    for frag in fragments:
        for span in frag.spans:
            piece = data[span.start_byte - base_offset:span.end_byte - base_offset]
            # Strict decode: a span boundary inside a codepoint's bytes fails here.
            piece.decode("utf-8")
            parts.append(piece)
    return b"".join(parts).decode("utf-8")


def _split(payload, budget, base_offset=0, source_path="docs/sample.md"):
    """split_payload + the invariants every scenario must uphold: exact span
    tiling and byte-exact payload reconstruction from spans."""
    budget_fn = budget if callable(budget) else (lambda index, b=budget: b)
    fragments = s5_splitter.split_payload(
        payload,
        base_offset=base_offset,
        budget_fn=budget_fn,
        tokenizer=_TOK,
        source_path=source_path,
    )
    s5_splitter.assert_spans_tile(payload, base_offset, fragments, source_path)
    assert _reconstruct(payload, base_offset, fragments) == payload
    return fragments


def _fragment_text(frag):
    """The fragment's reconstructed Markdown: synthetic open + payload lines +
    synthetic close (the exact reassembly rule the B4 pipeline uses)."""
    lines = []
    if frag.synthetic["fence_open"] is not None:
        lines.append(frag.synthetic["fence_open"])
    lines.extend(frag.raw_lines)
    if frag.synthetic["fence_close"] is not None:
        lines.append(frag.synthetic["fence_close"])
    return "\n".join(lines)


def _assert_fences_balanced(frag):
    """Every fragment's reconstructed text must be valid fenced Markdown: any
    opened fence is closed by a matching marker (same char, >= length)."""
    open_fence = None
    for line in _fragment_text(frag).split("\n"):
        marker = companion._fence_marker(line)
        if open_fence is None:
            if marker:
                open_fence = marker
        elif (marker and marker[0] == open_fence[0]
                and marker[1] >= open_fence[1]
                and line.strip()[marker[1]:].strip() == ""):
            open_fence = None
    assert open_fence is None, f"unclosed fence in fragment text: {_fragment_text(frag)!r}"


# --- single fragment ------------------------------------------------------------

def test_short_payload_single_fragment():
    """A payload within one fragment budget stays whole: one fragment, one span
    covering the entire byte range, no synthetic fences."""
    payload = "Alpha beta gamma.\n\nDelta epsilon zeta."
    frags = _split(payload, budget=100, base_offset=7)
    assert len(frags) == 1
    assert frags[0].raw_lines == ["Alpha beta gamma.", "", "Delta epsilon zeta."]
    assert frags[0].spans == [SourceSpan(7, 7 + len(payload.encode("utf-8")))]
    assert frags[0].synthetic == {"fence_open": None, "fence_close": None}


# --- prose split levels ---------------------------------------------------------

def test_paragraph_split_attaches_separator_to_leading_fragment():
    """Paragraphs that cannot share a fragment split on the \\n\\n boundary, and
    the separator bytes belong to the PRECEDING fragment's last span (raw_lines
    exclude them; the span end extends over them)."""
    p1 = "Alpha beta gamma."          # 2+1+2 = 5 pieces
    p2 = "Delta epsilon zeta."        # 2+2+2 = 6 pieces
    payload = p1 + "\n\n" + p2
    frags = _split(payload, budget=6, base_offset=1000)
    assert len(frags) == 2
    assert frags[0].raw_lines == [p1]
    assert frags[1].raw_lines == [p2]
    # Separator "\n\n" (bytes 17..19) rides in the leading fragment's span.
    assert frags[0].spans == [SourceSpan(1000, 1000 + 19)]
    assert frags[1].spans == [SourceSpan(1000 + 19, 1000 + len(payload.encode("utf-8")))]


def test_sentence_level_descent():
    """A paragraph too big for a fresh fragment descends to sentence units
    ((?<=[.!?])\\s+); the inter-sentence space goes to the preceding span."""
    payload = "One two. Three four. Five six."   # sentences cost 2 / 4 / 2 pieces
    frags = _split(payload, budget=4)
    assert [f.raw_lines for f in frags] == [["One two."], ["Three four."], ["Five six."]]
    assert frags[0].spans == [SourceSpan(0, 9)]      # includes the boundary space
    assert frags[1].spans == [SourceSpan(9, 21)]
    assert frags[2].spans == [SourceSpan(21, 30)]


def test_newline_level_descent():
    """Prose with no sentence punctuation descends past sentences to newline
    units; each line lands whole in its own fragment."""
    payload = "alpha beta\ngamma delta\nepsilon zeta"   # lines cost 3 / 4 / 3
    frags = _split(payload, budget=4)
    assert [f.raw_lines for f in frags] == [["alpha beta"], ["gamma delta"], ["epsilon zeta"]]


def test_whitespace_level_descent():
    """A single line too big for a fresh fragment descends to whitespace units,
    greedily packing words until the budget is exhausted."""
    payload = "aaaa bbbb cccc dddd"   # each word costs exactly 1 piece
    frags = _split(payload, budget=2)
    assert [f.raw_lines for f in frags] == [["aaaa bbbb"], ["cccc dddd"]]


# --- fenced code blocks ---------------------------------------------------------

def test_code_block_kept_whole_when_it_fits():
    """A fenced block that fits a fresh fragment is never split: it moves whole
    to the next fragment, keeping its source open AND close lines (both are
    payload with spans) and no synthetic wrappers anywhere."""
    payload = "Intro words here.\n\n```groovy\nx = 1\ny = 2\n```"
    frags = _split(payload, budget=12)
    assert len(frags) == 2
    assert frags[0].raw_lines == ["Intro words here."]
    assert frags[1].raw_lines == ["```groovy", "x = 1", "y = 2", "```"]
    assert frags[0].synthetic == {"fence_open": None, "fence_close": None}
    assert frags[1].synthetic == {"fence_open": None, "fence_close": None}
    # The blank separator line belongs to the prose fragment's span.
    assert frags[0].spans == [SourceSpan(0, 19)]
    for frag in frags:
        _assert_fences_balanced(frag)


def test_fence_split_adds_synthetic_wrappers_preserving_language():
    """A fence split across fragments stays valid Markdown in every fragment:
    the leading fragment gains a synthetic fence_close (same marker chars), the
    continuation a synthetic fence_open carrying the original language. Source
    fence lines are payload (their bytes are in the spans); synthetic lines are
    not (span reconstruction contains each source fence exactly once)."""
    payload = "```groovy\naaa bbb\nccc ddd\neee fff\n```"
    frags = _split(payload, budget=5)
    assert len(frags) == 2
    assert frags[0].raw_lines == ["```groovy", "aaa bbb"]
    assert frags[0].synthetic == {"fence_open": None, "fence_close": "```"}
    assert frags[1].raw_lines == ["ccc ddd", "eee fff", "```"]
    assert frags[1].synthetic == {"fence_open": "```groovy", "fence_close": None}
    # Source fence lines carry spans; synthetic ones do not: the byte slices
    # hold the source open fence in frag 0 and the source close fence in frag 1,
    # and neither synthetic line's text is in any span.
    data = payload.encode("utf-8")
    slice0 = data[frags[0].spans[0].start_byte:frags[0].spans[-1].end_byte].decode("utf-8")
    slice1 = data[frags[1].spans[0].start_byte:frags[1].spans[-1].end_byte].decode("utf-8")
    assert slice0 == "```groovy\naaa bbb\n"
    assert slice1 == "ccc ddd\neee fff\n```"
    for frag in frags:
        _assert_fences_balanced(frag)


def test_tilde_fence_marker_and_language_preserved():
    """Synthetic wrappers reproduce the EXACT source marker (char and length —
    here four tildes) plus the info-string language on the re-open."""
    payload = "~~~~xml\n<a>1</a>\n<b>2</b>\n~~~~"
    frags = _split(payload, budget=4)
    assert len(frags) == 2
    assert frags[0].synthetic == {"fence_open": None, "fence_close": "~~~~"}
    assert frags[1].synthetic == {"fence_open": "~~~~xml", "fence_close": None}
    for frag in frags:
        _assert_fences_balanced(frag)


def test_oversized_code_line_splits_at_whitespace_boundary():
    """One code line over a fresh fragment's budget splits at the largest
    whitespace boundary that fits; every resulting fragment is a valid fenced
    block and the consumed boundary space rides in the preceding span.

    An oversized code line descends on a FRESH fragment (the code path closes
    before descending so a fence-marker prefix can't be stranded onto the
    fence-open fragment — see test_oversized_fence_marker_prefixed_line_stays_balanced).
    """
    payload = "```\nfoo bar\nalpha beta gamma delta\n```"
    frags = _split(payload, budget=5)
    assert [f.raw_lines for f in frags] == [
        ["```", "foo bar"],
        ["alpha beta gamma"],
        ["delta", "```"],
    ]
    assert frags[0].synthetic == {"fence_open": None, "fence_close": "```"}
    assert frags[1].synthetic == {"fence_open": "```", "fence_close": "```"}
    assert frags[2].synthetic == {"fence_open": "```", "fence_close": None}
    # The space after "gamma" is consumed at the boundary -> preceding span.
    assert frags[1].spans == [SourceSpan(12, 29)]
    for frag in frags:
        _assert_fences_balanced(frag)


def test_minified_line_splits_at_codepoint_boundary_multibyte():
    """No-whitespace content splits at the largest Unicode codepoint boundary
    that fits. With 4-byte emoji, every span boundary must sit BETWEEN
    codepoints: each per-span byte slice strict-decodes (asserted by _split),
    and the byte offsets land on exact 4-byte emoji multiples."""
    payload = "```\n" + "\U0001f680" * 24 + "\n```"   # rocket line = 6 pieces
    frags = _split(payload, budget=3, base_offset=200)
    # The oversized rocket line descends on a FRESH fragment (12 rockets = 3
    # pieces each); the fence-open and -close lines sit in their own fragments.
    assert [f.raw_lines for f in frags] == [
        ["```"],
        ["\U0001f680" * 12],
        ["\U0001f680" * 12],
        ["```"],
    ]
    # UTF-8 offsets: "```\n" = 4 bytes, each rocket = 4 bytes, "\n" = 1.
    assert frags[1].spans == [SourceSpan(200 + 4, 200 + 4 + 48)]
    assert frags[2].spans == [SourceSpan(200 + 52, 200 + 52 + 48 + 1)]  # + closing "\n"
    for frag in frags:
        _assert_fences_balanced(frag)
        assert frag.synthetic["fence_open"] == (None if frag is frags[0] else "```")


def test_synthetic_wrapper_lengthens_past_a_bare_marker_run():
    """A fenced body line that codepoint-splits into a bare fence-marker run
    must not close its synthetic wrapper — the wrapper is lengthened past the
    run (CommonMark-correct), keeping every fragment valid without altering a
    source byte. Reachable only at pathologically tiny budgets, but a stated
    hard invariant."""
    payload = "```\n````x"  # 3-backtick open, body line begins with a 4-run
    frags = _split(payload, budget=1)
    for frag in frags:
        _assert_fences_balanced(frag)
    # The fragment carrying the bare 4-backtick run wraps it in a >=5 fence.
    wrapped = [f for f in frags if "````" in f.raw_lines]
    assert wrapped
    for frag in wrapped:
        marker = frag.synthetic["fence_open"] or frag.synthetic["fence_close"]
        assert marker.count("`") >= 5


def test_unbalanceable_fenced_split_fails_closed():
    """When a synthetic open would be paired with an unmodifiable SOURCE close
    and a codepoint-split bare-run tail between them (a pathological tiny-budget
    case unreachable with the real tokenizer), the splitter FAILS CLOSED rather
    than silently emit invalid fenced Markdown."""
    class _CharTokenizer:
        def count(self, text):
            return len(text)

    payload = "```py\naaaaaaa```\n```"
    with pytest.raises(UnsplittableSpanError, match="valid Markdown"):
        s5_splitter.split_payload(payload, base_offset=0, budget_fn=lambda _i: 7,
                                  tokenizer=_CharTokenizer(), source_path="t.md")


def test_oversized_fence_marker_prefixed_line_stays_balanced():
    """Regression: an oversized minified code line that BEGINS with a fence-
    marker run must not be codepoint-split onto the fence-open fragment — a
    small remaining budget could strand a bare marker prefix that closes the
    fence early and emits invalid Markdown. Descend on a fresh fragment."""
    payload = "````py\n````longtoken\nemoji a alpha.\n````"
    frags = _split(payload, budget=3)
    for frag in frags:
        _assert_fences_balanced(frag)
    # The oversized body line splits at a non-bare-marker boundary.
    assert not any(frag.raw_lines and frag.raw_lines[0] == "````"
                   for frag in frags), "a bare fence marker was stranded as content"


def test_descent_backfills_the_open_fragment_instead_of_stranding_it():
    """Regression: a unit too big for even a FRESH fragment must be descended
    WITHOUT first closing the open fragment, so its leading sub-units backfill
    the room already available.

    Closing first stranded that room and emitted degenerate fragments — a lone
    emoji, a two-word line — whose embeddings are noise in the corpus.
    """
    # "alpha." fits fragment 0 (2 pieces of a 10-piece budget); the following
    # sentence is far too big for any fragment, so it must be split — and its
    # first words belong in fragment 0's remaining 8 pieces.
    long_sentence = " ".join(f"word{i:03d}" for i in range(40))  # 2 pieces each
    payload = f"alpha.\n{long_sentence}"
    frags = _split(payload, budget=10)

    assert frags[0].raw_lines[0].startswith("alpha.")
    # Fragment 0 is filled to its budget, not left at the 2 pieces of "alpha.".
    assert _TOK.count("\n".join(frags[0].raw_lines)) == 10
    # No fragment is degenerate while later text remains to be placed.
    for frag in frags[:-1]:
        assert _TOK.count("\n".join(frag.raw_lines)) > 5


# --- unsplittable spans ---------------------------------------------------------

def test_zero_budget_raises_unsplittable_span():
    """A budget of zero can fit no non-empty source slice: fail closed with the
    source path, payload-relative line number, and the byte range of the slice
    where splitting stalled (the first whitespace unit, "abc")."""
    with pytest.raises(UnsplittableSpanError,
                       match=r"docs/zero\.md.*line 1.*bytes \[50, 53\)") as excinfo:
        s5_splitter.split_payload(
            "abc def",
            base_offset=50,
            budget_fn=lambda index: 0,
            tokenizer=_TOK,
            source_path="docs/zero.md",
        )
    err = excinfo.value
    assert err.source_path == "docs/zero.md"
    assert err.line_number == 1
    assert err.start_byte == 50
    assert err.end_byte == 53


def test_tiny_budget_error_names_later_line_and_bytes():
    """When a later fragment's budget cannot fit even one codepoint, the error
    pinpoints the line and byte range where splitting stalled."""
    with pytest.raises(UnsplittableSpanError,
                       match=r"docs/tiny\.md.*line 2.*bytes \[103, 105\)") as excinfo:
        s5_splitter.split_payload(
            "ab\ncd",
            base_offset=100,
            budget_fn=lambda index: 1 if index == 0 else 0,
            tokenizer=_TOK,
            source_path="docs/tiny.md",
        )
    err = excinfo.value
    assert err.line_number == 2
    assert (err.start_byte, err.end_byte) == (103, 105)


# --- assert_spans_tile ----------------------------------------------------------

def _frag(*spans):
    return Fragment(
        raw_lines=["x"],
        spans=[SourceSpan(s, e) for s, e in spans],
        synthetic={"fence_open": None, "fence_close": None},
    )


def test_assert_spans_tile_rejects_gap():
    with pytest.raises(ValueError, match=r"gap.*docs/gap\.md"):
        s5_splitter.assert_spans_tile("abc", 0, [_frag((0, 1)), _frag((2, 3))], "docs/gap.md")


def test_assert_spans_tile_rejects_overlap():
    with pytest.raises(ValueError, match=r"overlap.*docs/overlap\.md"):
        s5_splitter.assert_spans_tile("abc", 0, [_frag((0, 2)), _frag((1, 3))], "docs/overlap.md")


def test_assert_spans_tile_rejects_missing_tail_coverage():
    """Spans that stop short of the payload end are a gap too — full coverage
    of [base_offset, base_offset + payload bytes) is mandatory."""
    with pytest.raises(ValueError, match="gap"):
        s5_splitter.assert_spans_tile("abc", 0, [_frag((0, 2))], "docs/tail.md")


def test_assert_spans_tile_respects_base_offset():
    """Coverage runs over [base_offset, base_offset + payload bytes): spans at
    payload-relative offsets (below base_offset) must be rejected."""
    s5_splitter.assert_spans_tile("abc", 40, [_frag((40, 43))], "docs/ok.md")
    with pytest.raises(ValueError, match="overlap"):
        s5_splitter.assert_spans_tile("abc", 40, [_frag((0, 3))], "docs/ok.md")
