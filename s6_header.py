#!/usr/bin/env python3
"""S6 deduplicated contextual header for chunk embedding inputs.

The S6 header prefixes a chunk's embedding text with the retrieval context the
raw fragment does not carry: the page title for official docs, breadcrumb +
page title for companion references. Two hard rules govern it:

- Dedup, never repeat: the fragment's first raw-content line is the repeated
  section heading, so any header line that heading already states is dropped —
  this is how "the section heading is removed from the S6 header through
  deduplication". Comparison is casefolded and whitespace-normalized; the
  SURVIVING display text is never rewritten.
- Token cap (HEADER_TOKEN_CAP = 64), counted by the same WordPiece seam that
  budgets the payload split. Priority under the cap: the full title always
  survives whole, then breadcrumb segments from the RIGHT — leftmost segments
  are the least specific and are dropped first. A title that alone overflows
  the cap cannot be fixed here and raises HeaderCoreTooLongError.

An empty header is legal (a fully-deduplicated chunk): ``S6Header((), "", 0)``
— the caller then embeds the raw document without a header prefix.
"""
from dataclasses import dataclass

from companion import COMPANION_SOURCE_TYPE, OFFICIAL_SOURCE_TYPE

HEADER_TOKEN_CAP = 64

# Breadcrumb display strings join/split segments on this exact separator
# ("Companion Reference > Components > Map Component Functions").
_SEGMENT_SEPARATOR = " > "


class HeaderCoreTooLongError(ValueError):
    """The required title alone exceeds the header token cap.

    Breadcrumb segments are droppable; the title is the header's core and is
    never shrunk. Carries ``title`` and ``token_count`` (the title's own
    WordPiece count) so the pipeline can name the unembeddable page without
    re-tokenizing.
    """

    def __init__(self, title, token_count, cap):
        super().__init__(
            f"header core too long: title {title!r} alone counts "
            f"{token_count} WordPieces against the {cap}-token cap"
        )
        self.title = title
        self.token_count = token_count


@dataclass(frozen=True)
class S6Header:
    """Final header: display lines in order, their "\\n"-joined text, and the
    WordPiece count of exactly that text (0 for the legal empty header)."""

    lines: tuple[str, ...]
    text: str
    token_count: int


def _norm(s):
    """Dedup comparison key: casefold + collapse whitespace runs + strip.

    Used ONLY for comparing — display text in the output is never normalized.
    """
    return " ".join(s.split()).casefold()


def _dedup_breadcrumb(breadcrumb, title):
    """Rule (a): drop the breadcrumb's final segment when it repeats the title.

    The title is emitted as its own header line, so a breadcrumb ending in the
    same page name would state it twice. Remaining segments keep their display
    text; a single-segment breadcrumb equal to the title empties out entirely
    (the empty line is then dropped by the line-level dedup).
    """
    segments = breadcrumb.split(_SEGMENT_SEPARATOR)
    if segments and _norm(segments[-1]) == _norm(title):
        segments = segments[:-1]
    return _SEGMENT_SEPARATOR.join(segments)


def build_s6_header(*, source_type, title, breadcrumb, raw_first_line,
                    tokenizer, cap=HEADER_TOKEN_CAP):
    """Build the deduplicated, capped S6 header for one chunk.

    ``raw_first_line`` is the fragment's first raw-content line — the repeated
    section heading — so a heading that restates the title (or a whole header
    line) removes that line HERE rather than appearing twice in the embedding
    input. ``breadcrumb`` is ignored for official pages. Raises
    HeaderCoreTooLongError only when the title alone cannot fit ``cap`` after
    every breadcrumb segment has been dropped; a fully-deduplicated empty
    header is a legal result, not an error.
    """
    if source_type == OFFICIAL_SOURCE_TYPE:
        candidates = [("title", title)]
    elif source_type == COMPANION_SOURCE_TYPE:
        candidates = [
            ("breadcrumb", _dedup_breadcrumb(breadcrumb or "", title)),
            ("title", title),
        ]
    else:
        raise ValueError(f"unknown source_type: {source_type!r}")

    # Rules (b)/(c): a line the raw first line already states is redundant —
    # prefix match, so a "<title> Overview" heading still dedups the title.
    # An emptied breadcrumb also falls here ("" is a prefix of everything).
    # Rule (d): adjacent lines that normalize equal collapse to the first.
    norm_first = _norm(raw_first_line)
    kept = []
    for kind, line in candidates:
        norm_line = _norm(line)
        if norm_first.startswith(norm_line):
            continue
        if kept and _norm(kept[-1][1]) == norm_line:
            continue
        kept.append((kind, line))

    # Cap enforcement: drop LEFTMOST breadcrumb segments until the header
    # fits — rightmost segments are the most specific and survive longest;
    # the title is never shrunk. The breadcrumb line disappears entirely
    # before the title is declared unembeddable.
    while True:
        lines = tuple(line for _, line in kept)
        text = "\n".join(lines)
        token_count = tokenizer.count(text)
        if token_count <= cap:
            return S6Header(lines=lines, text=text, token_count=token_count)

        bc_index = next(
            (i for i, (kind, _) in enumerate(kept) if kind == "breadcrumb"),
            None,
        )
        if bc_index is not None:
            segments = kept[bc_index][1].split(_SEGMENT_SEPARATOR)
            if len(segments) > 1:
                joined = _SEGMENT_SEPARATOR.join(segments[1:])
                kept[bc_index] = ("breadcrumb", joined)
            else:
                del kept[bc_index]
            continue

        # Only the title remains (an over-cap header always still holds a
        # line), so token_count here is the title's own count.
        raise HeaderCoreTooLongError(title, token_count, cap)
