#!/usr/bin/env python3
"""S5 payload splitter: budget-driven fragmentation with exact source spans.

Splits one *payload* (a section body handed over by the S5 pipeline) into
fragments whose token cost fits per-fragment budgets, upholding the S5 hard
rules from the kb-25 plan:

- **Never truncate or omit source bytes.** Every payload byte lands in exactly
  one span; the ordered spans of the returned fragments tile
  ``[base_offset, base_offset + len(payload.encode("utf-8")))`` exactly.
  ``assert_spans_tile`` verifies that invariant (gap / overlap / coverage) and
  is exported so callers and tests can enforce it after every split.
- **Spans are UTF-8 byte offsets into the payload's source record** (start
  inclusive, end exclusive). Boundaries always fall between codepoints — the
  splitter only ever cuts at Python ``str`` indices, so a multi-byte codepoint
  can never be bisected.
- **Split fenced code stays valid fenced Markdown.** A fence split across
  fragments gets a *synthetic* ``fence_close`` (the exact source marker chars)
  on the leading fragment and a *synthetic* ``fence_open`` (marker + original
  info-string language) on the continuation. Fence lines that exist in the
  source are payload lines WITH spans; synthetic lines exist only in
  ``Fragment.synthetic`` and carry no span. Synthetic lines are also never
  charged against the payload budget — ``budget_fn`` has already reserved
  their cost (along with header/heading overheads).
- **Fail closed.** ``UnsplittableSpanError`` (with source path, payload-
  relative line number, and byte range) is raised ONLY when no non-empty
  source slice can fit a fresh fragment's budget — a budget <= 0, a single
  codepoint costing more than the budget, or (documented tie-break below) a
  fence marker line that alone exceeds a fresh budget.

Fence discipline is ``companion.split_markdown_sections``'s, reused via
``companion._fence_marker``: a fence opens with 3+ backticks or tildes (the
info string is the language) and closes only on the same char with an equal or
longer run and no info string.

Split priorities:

- Prose: paragraph (``\\n\\n``) -> sentence (``(?<=[.!?])\\s+``) -> newline ->
  whitespace. Greedy fill appends the largest whole unit that fits the
  remaining budget; a unit that alone exceeds a FRESH fragment's budget is
  split one level down.
- Code: complete lines -> for one oversized line, the largest whitespace
  boundary that fits -> for no-whitespace (minified) content, the largest
  Unicode codepoint boundary that fits.

Recorded tie-breaks (points the plan leaves open, resolved here the simplest
correct way):

1. A unit that no longer fits the REMAINING budget moves whole to the next
   fragment; descent into sub-units happens only when the unit exceeds a
   FRESH fragment's budget (the plan conditions descent on "a FRESH
   fragment's budget", so partially-filled fragments are closed rather than
   topped up with sub-units).
2. Prose descends below whitespace to the codepoint level as a last resort, so
   a single unbroken word longer than a whole budget is split rather than
   truncated (the error contract permits raising only when *no* non-empty
   slice fits).
3. Fence marker lines (open/close) are atomic: splitting one would break
   Markdown validity, so an open/close line that alone exceeds a fresh budget
   raises ``UnsplittableSpanError`` instead of being split.
4. Fragment boundary whitespace (e.g. the ``\\n\\n`` between paragraphs) is
   attributed to the PRECEDING fragment's last span; whitespace before the
   first fragment's content has no preceding fragment and rides at the front
   of the FIRST fragment's span. A payload that is entirely whitespace becomes
   a single zero-token fragment so tiling still holds.
5. Greedy fill is contiguous, so every fragment covers one contiguous byte
   range and carries exactly ONE span; ``spans`` stays a list because the
   chunk contract records a span list.
6. The largest-fitting-prefix search assumes token counts grow monotonically
   with prefix length (true for the fake tokenizer, near-true for WordPiece);
   a linear fix-down after the binary search guarantees the returned prefix
   always fits even if a real tokenizer is locally non-monotone.

The module is pure: no I/O, stdlib + ``companion`` only. Token counting uses
the injected ``tokenizer.count(text) -> int`` seam from ``kb_tokenizer``.
"""

import re
from dataclasses import dataclass

import companion


class UnsplittableSpanError(ValueError):
    """No non-empty source slice can fit a fresh fragment's budget.

    Carries the failing location so the release build can point at the exact
    source bytes: ``source_path``, ``line_number`` (1-based, payload-relative),
    and the record-absolute UTF-8 byte range ``[start_byte, end_byte)``.
    """

    def __init__(self, source_path, line_number, start_byte, end_byte, reason=""):
        self.source_path = source_path
        self.line_number = line_number
        self.start_byte = start_byte
        self.end_byte = end_byte
        message = (
            f"unsplittable span in {source_path} at line {line_number}: "
            f"bytes [{start_byte}, {end_byte}) cannot fit the fragment budget"
        )
        if reason:
            message += f" ({reason})"
        super().__init__(message)


@dataclass(frozen=True)
class SourceSpan:
    """UTF-8 byte offsets into the payload's source record.

    ``start_byte`` is inclusive, ``end_byte`` exclusive. Boundaries never fall
    inside a codepoint's byte sequence.
    """

    start_byte: int
    end_byte: int


@dataclass
class Fragment:
    """One budget-sized piece of the payload.

    ``raw_lines`` are payload lines only — no repeated heading, no synthetic
    fence lines. ``spans`` are the ordered source spans backing those lines
    (plus any boundary whitespace attributed to this fragment). ``synthetic``
    holds the fence lines ADDED by splitting — ``{"fence_open": str | None,
    "fence_close": str | None}`` — which carry no span and are excluded from
    coverage accounting.
    """

    raw_lines: list[str]
    spans: list[SourceSpan]
    synthetic: dict


# Prose unit separators, coarse to fine. A run of 2+ newlines is one paragraph
# separator (so triple newlines do not leave a stray leading "\n" on the next
# unit); sentences follow the plan's regex verbatim.
_PARAGRAPH_SEP = re.compile(r"\n{2,}")
_SENTENCE_SEP = re.compile(r"(?<=[.!?])\s+")
_NEWLINE_SEP = re.compile(r"\n")
_WHITESPACE_SEP = re.compile(r"\s+")
_PROSE_SEPARATORS = (_PARAGRAPH_SEP, _SENTENCE_SEP, _NEWLINE_SEP, _WHITESPACE_SEP)


def _units(text, separator):
    """Relative (start, end) content ranges of ``text`` split on ``separator``.

    Separator matches fall BETWEEN units; whitespace-only pieces are dropped
    (their bytes ride along as boundary whitespace in the span tiling).
    """
    ranges = []
    pos = 0
    for match in separator.finditer(text):
        if match.start() > pos:
            ranges.append((pos, match.start()))
        pos = match.end()
    if pos < len(text):
        ranges.append((pos, len(text)))
    return [(s, e) for s, e in ranges if text[s:e].strip()]


def _line_bounds(payload):
    """(start, end) char offsets of every payload line, ends excluding "\\n"."""
    bounds = []
    pos = 0
    n = len(payload)
    while pos <= n:
        nl = payload.find("\n", pos)
        if nl == -1:
            bounds.append((pos, n))
            break
        bounds.append((pos, nl))
        pos = nl + 1
    return bounds


def _parse_segments(payload):
    """Split the payload into prose runs and fenced code blocks.

    Yields ``("prose", start, end)`` char ranges and
    ``("code", lines, marker, language, has_close)`` where ``lines`` is
    ``[(start, end, kind)]`` with kind in {"open", "body", "close"} and
    ``marker`` is the exact fence marker string (e.g. ``"~~~~"``). Fence
    discipline matches companion.split_markdown_sections: an unclosed fence
    runs to the payload end (``has_close`` False).
    """
    segments = []
    bounds = _line_bounds(payload)
    prose_start = None
    i = 0
    while i < len(bounds):
        ls, le = bounds[i]
        marker = companion._fence_marker(payload[ls:le])
        if marker is None:
            if prose_start is None:
                prose_start = ls
            i += 1
            continue
        if prose_start is not None:
            segments.append(("prose", prose_start, ls))
            prose_start = None
        char, length = marker
        stripped = payload[ls:le].strip()
        language = stripped[length:].strip()
        lines = [(ls, le, "open")]
        has_close = False
        i += 1
        while i < len(bounds):
            ls2, le2 = bounds[i]
            m2 = companion._fence_marker(payload[ls2:le2])
            i += 1
            if (m2 and m2[0] == char and m2[1] >= length
                    and payload[ls2:le2].strip()[m2[1]:].strip() == ""):
                lines.append((ls2, le2, "close"))
                has_close = True
                break
            lines.append((ls2, le2, "body"))
        segments.append(("code", lines, char * length, language, has_close))
    if prose_start is not None:
        segments.append(("prose", prose_start, len(payload)))
    return segments


class _Splitter:
    """Greedy fragment builder over one payload.

    Tracks the open fragment as a contiguous char range ``[frag_start,
    content_end)``; finished fragments record (content_start, content_end,
    synthetic_open, synthetic_close). Span boundaries are derived afterwards
    from consecutive fragments' content starts, which is what attributes
    boundary whitespace to the preceding span and makes tiling exact by
    construction.
    """

    def __init__(self, payload, base_offset, budget_fn, tokenizer, source_path):
        self._payload = payload
        self._base = base_offset
        self._budget_fn = budget_fn
        self._tokenizer = tokenizer
        self._source_path = source_path
        # byte_of[i] = UTF-8 byte offset of char i (byte_of[len] = total bytes).
        byte_of = [0] * (len(payload) + 1)
        total = 0
        for i, ch in enumerate(payload):
            total += len(ch.encode("utf-8"))
            byte_of[i + 1] = total
        self._byte_of = byte_of
        self._finished = []          # (content_start, content_end, syn_open, syn_close)
        self._frag_start = None      # char start of the open fragment's content
        self._content_end = None     # char end of the open fragment's content
        self._syn_open = None        # synthetic fence_open of the open fragment
        self._pending_open = None    # synthetic fence_open owed to the NEXT fragment
        self._in_fence = None        # (marker, language) between source open/close

    # -- budget / counting -------------------------------------------------

    def _budget(self):
        """Budget of the fragment being filled: index == finished count."""
        return self._budget_fn(len(self._finished))

    def _fresh_budget(self):
        """Budget the NEXT fragment would have (closing the open one, if any)."""
        index = len(self._finished) + (1 if self._frag_start is not None else 0)
        return self._budget_fn(index)

    def _fits_fresh(self, start, end):
        """Would this whole slice fit a fragment of its own?"""
        return self._count(start, end) <= self._fresh_budget()

    def _count(self, start, end):
        return self._tokenizer.count(self._payload[start:end])

    # -- fragment lifecycle --------------------------------------------------

    def _open(self, start, end):
        self._frag_start = start
        self._content_end = end
        self._syn_open = self._pending_open
        self._pending_open = None

    def _close(self):
        """Finish the open fragment (no-op when none is open).

        Closing while inside a source fence is what splits the fence: the
        finished fragment gets the synthetic close marker, and the marker plus
        original language is owed to the next fragment as a synthetic open.
        """
        if self._frag_start is None:
            return
        syn_close = None
        if self._in_fence is not None:
            marker, language = self._in_fence
            syn_close = marker
            self._pending_open = marker + language
        self._finished.append((self._frag_start, self._content_end, self._syn_open, syn_close))
        self._frag_start = None
        self._content_end = None
        self._syn_open = None

    def _try_append(self, start, end):
        """Extend the open fragment (or open a fresh one) through char ``end``
        when the fragment's whole contiguous text still fits its budget."""
        if self._frag_start is None:
            if self._count(start, end) <= self._budget():
                self._open(start, end)
                return True
            return False
        if self._count(self._frag_start, end) <= self._budget():
            self._content_end = end
            return True
        return False

    # -- placement ----------------------------------------------------------

    def _place_prose(self, start, end, level):
        """Greedy fill at one prose level, descending per tie-breaks 1 and 2."""
        if level == len(_PROSE_SEPARATORS):
            self._place_codepoints(start, end)
            return
        text = self._payload[start:end]
        for rel_start, rel_end in _units(text, _PROSE_SEPARATORS[level]):
            unit_start, unit_end = start + rel_start, start + rel_end
            if self._try_append(unit_start, unit_end):
                continue
            if self._fits_fresh(unit_start, unit_end):
                # Tie-break 1: a unit that no longer fits the REMAINING budget
                # but would fit a fragment of its own moves whole to the next one.
                self._close()
                self._try_append(unit_start, unit_end)
                continue
            # The unit exceeds even a fresh fragment, so it must be split
            # anyway (tie-break 2). Descend WITHOUT closing: its leading
            # sub-units backfill the open fragment's remaining room first.
            # Closing here instead would strand that room and emit degenerate
            # fragments (a lone emoji, a two-word line) whose vectors are noise.
            self._place_prose(unit_start, unit_end, level + 1)

    def _place_code(self, lines, marker, language, has_close):
        """Place one fenced block: whole if it fits, else line by line, else
        (for one oversized body line) whitespace -> codepoint descent."""
        seg_start, seg_end = lines[0][0], lines[-1][1]
        if self._try_append(seg_start, seg_end):
            if not has_close:
                # Unclosed source fence: the final _close() adds the
                # synthetic close so the fragment stays valid Markdown.
                self._in_fence = (marker, language)
            return
        if self._fits_fresh(seg_start, seg_end):
            self._close()
            if self._try_append(seg_start, seg_end):
                if not has_close:
                    self._in_fence = (marker, language)
                return
        # The block must be split line by line. Do NOT close first: its leading
        # lines backfill the open fragment's remaining room (a fragment that
        # ends inside the fence gets the synthetic close from _close()).
        for line_start, line_end, kind in lines:
            placed = self._try_append(line_start, line_end)
            if not placed and self._fits_fresh(line_start, line_end):
                self._close()
                placed = self._try_append(line_start, line_end)
            if not placed:
                if kind != "body":
                    # Tie-break 3: fence marker lines are atomic.
                    raise UnsplittableSpanError(
                        self._source_path,
                        self._line_number(line_start),
                        self._base + self._byte_of[line_start],
                        self._base + self._byte_of[line_end],
                        reason="fence marker line is atomic",
                    )
                self._place_prose(line_start, line_end, len(_PROSE_SEPARATORS) - 1)
            if kind == "open":
                self._in_fence = (marker, language)
            elif kind == "close":
                self._in_fence = None

    def _place_codepoints(self, start, end):
        """Last resort: the largest codepoint-boundary prefix that fits.

        Cuts at Python str indices only, so UTF-8 byte offsets always land
        between codepoints. Raises when even a single codepoint exceeds a
        fresh fragment's budget — the one truly unsplittable case.
        """
        pos = start
        while pos < end:
            if self._frag_start is not None:
                fit = self._largest_fit(self._frag_start, pos, end)
                if fit > pos:
                    self._content_end = fit
                    pos = fit
                    continue
                self._close()
            fit = self._largest_fit(pos, pos, end)
            if fit == pos:
                raise UnsplittableSpanError(
                    self._source_path,
                    self._line_number(pos),
                    self._base + self._byte_of[pos],
                    self._base + self._byte_of[end],
                )
            self._open(pos, fit)
            pos = fit

    def _largest_fit(self, anchor, lo, hi):
        """Largest char index k in [lo, hi] with count(payload[anchor:k]) within
        budget (binary search per tie-break 6, then a fit-guaranteeing fix-down)."""
        budget = self._budget()
        low, high = lo, hi
        while low < high:
            mid = (low + high + 1) // 2
            if self._count(anchor, mid) <= budget:
                low = mid
            else:
                high = mid - 1
        while low > lo and self._count(anchor, low) > budget:
            low -= 1
        return low

    def _line_number(self, char_pos):
        """1-based line number of a char position, payload-relative."""
        return self._payload.count("\n", 0, char_pos) + 1

    # -- driver ---------------------------------------------------------------

    def run(self):
        for segment in _parse_segments(self._payload):
            if segment[0] == "prose":
                self._place_prose(segment[1], segment[2], 0)
            else:
                self._place_code(*segment[1:])
        self._close()
        if not self._finished and self._payload:
            # Tie-break 4: an all-whitespace payload forms one zero-token
            # fragment so its bytes are still span-covered, never dropped.
            self._finished.append((0, len(self._payload), None, None))
        return self._build_fragments()

    def _build_fragments(self):
        """Materialize fragments; span boundaries are consecutive content
        starts (first span starts at 0, last ends at the payload end), which
        attributes boundary whitespace to the preceding fragment's span and
        tiles the payload byte range exactly by construction."""
        fragments = []
        total_chars = len(self._payload)
        for i, (content_start, content_end, syn_open, syn_close) in enumerate(self._finished):
            span_start = 0 if i == 0 else content_start
            span_end = self._finished[i + 1][0] if i + 1 < len(self._finished) else total_chars
            fragments.append(Fragment(
                raw_lines=self._payload[content_start:content_end].split("\n"),
                spans=[SourceSpan(self._base + self._byte_of[span_start],
                                  self._base + self._byte_of[span_end])],
                synthetic={"fence_open": syn_open, "fence_close": syn_close},
            ))
        return fragments


def split_payload(payload, *, base_offset, budget_fn, tokenizer, source_path):
    """Split ``payload`` into budget-sized fragments with exact source spans.

    - ``base_offset``: UTF-8 byte offset of the payload's first byte within its
      source record; all emitted spans are record-absolute.
    - ``budget_fn(fragment_index) -> int``: the PAYLOAD token budget of each
      fragment. The caller has already reserved header, repeated-heading, and
      synthetic-fence-wrapper overheads, so only payload text (including
      source fence lines, which ARE payload) is charged here.
    - ``tokenizer``: the ``count(text) -> int`` seam from ``kb_tokenizer``.
    - ``source_path``: used only for ``UnsplittableSpanError`` reporting.

    Returns fragments whose ordered spans tile
    ``[base_offset, base_offset + len(payload.encode("utf-8")))`` exactly; an
    empty payload yields no fragments. Raises ``UnsplittableSpanError`` when no
    non-empty source slice can fit a fresh fragment's budget.
    """
    if not payload:
        return []
    return _Splitter(payload, base_offset, budget_fn, tokenizer, source_path).run()


def assert_spans_tile(payload, base_offset, fragments, source_path):
    """Verify the coverage invariant: ordered spans tile the payload exactly.

    Raises ValueError naming the first gap or overlap (and the source path)
    when any byte of ``[base_offset, base_offset + payload bytes)`` is
    uncovered, covered twice, or a span is empty/inverted.
    """
    expected = base_offset
    end_limit = base_offset + len(payload.encode("utf-8"))
    for frag_index, fragment in enumerate(fragments):
        for span in fragment.spans:
            if span.end_byte <= span.start_byte:
                raise ValueError(
                    f"empty or inverted span [{span.start_byte}, {span.end_byte}) "
                    f"in fragment {frag_index} of {source_path}"
                )
            if span.start_byte > expected:
                raise ValueError(
                    f"span coverage gap of {span.start_byte - expected} byte(s) at "
                    f"offset {expected} before fragment {frag_index} in {source_path}"
                )
            if span.start_byte < expected:
                raise ValueError(
                    f"span overlap at offset {span.start_byte} (coverage already "
                    f"reached {expected}) in fragment {frag_index} of {source_path}"
                )
            expected = span.end_byte
    if expected < end_limit:
        raise ValueError(
            f"span coverage gap of {end_limit - expected} byte(s) at offset "
            f"{expected}: spans end before the payload end in {source_path}"
        )
    if expected > end_limit:
        raise ValueError(
            f"span overlap: coverage {expected} extends beyond the payload end "
            f"{end_limit} in {source_path}"
        )
