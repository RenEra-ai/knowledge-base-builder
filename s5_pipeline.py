#!/usr/bin/env python3
"""S5 tokenizer-aware chunking pipeline (kb-25 candidates B5/B56/B567).

Pipeline order per the intent plan, per section:

1. Split raw Markdown / extracted source text BEFORE rendering HTML.
2. Compute the deterministic S6 header and reserve its token overhead.
3. Reserve one repeated section-heading line and any required code-fence
   wrappers.
4. Split the source payload into the remaining tokenizer budget
   (s5_splitter — real WordPiece counts, never the legacy len//4 estimate).
5. Reassemble each fragment's raw document and re-tokenize the final
   embedding input: every input must fit the model window (model-side
   truncation is forbidden — a violation aborts the build).
6. Render content_html only after final fragments exist.

Chunk records carry the legacy 15-field contract PLUS the S5 fields
(S5_CHUNK_FIELDS): exact token counts, the source record identity and byte
span, the synthetic-wrapper metadata, and the S6 header text. token_estimate
is retained for backward-compatible diagnostics only.

Source records: for companion files, the sha256-verified staged Markdown (the
record IS the verified bytes); for official pages, a deterministic per-page
extracted text document (heading + extracted section text blocks joined by
blank lines). ``source_start_byte``/``source_end_byte`` are UTF-8 offsets into
that record — except for sections rewritten by ``strip_xml_blocks``, which
record the ORIGINAL section body range with
``synthetic_wrapper_metadata.stripped_xml = true`` (coverage is asserted
against the payload actually split).
"""
import argparse
import json
import os
import sys

from companion import (
    COMPANION_CATEGORY,
    COMPANION_SOURCE_TYPE,
    COMPANION_VERIFICATION_STATUS,
    OFFICIAL_SOURCE_TYPE,
    OFFICIAL_VERIFICATION_STATUS,
    PROVENANCE_FIELDS,
    _fence_marker,
    companion_chunk_id,
    filter_sections,
    split_markdown_sections_with_offsets,
    strip_xml_blocks,
)
from chunk_docs import (
    DEFAULT_COMPANION_CONFIG,
    DEFAULT_COMPANION_STAGING,
    DEFAULT_INPUT,
    assign_chunk_indices,
    chunk_markdown_html,
    derive_page_key,
    detect_category,
    elements_to_html,
    elements_to_text,
    estimate_tokens,
    make_chunk_id,
    split_on_headings,
    verified_companion_texts,
)
from kb_tokenizer import EFFECTIVE_BUDGET, FakeWordPieceTokenizer, load_tokenizer
from s5_splitter import assert_spans_tile, split_payload
from s6_header import build_s6_header

# The fields every S5 chunk carries beyond the legacy contract.
S5_CHUNK_FIELDS = {
    "source_token_count", "content_token_count", "embedding_token_count",
    "source_record_id", "source_start_byte", "source_end_byte",
    "synthetic_wrapper_metadata", "s6_header",
}


def _markdown_title(sections):
    """First level-1 heading among parsed Markdown sections, else ''."""
    for sec in sections:
        if sec["level"] == 1 and sec["heading"]:
            return sec["heading"]
    return ""


def _fence_wrapper_reserve(payload, tokenizer):
    """Worst-case synthetic fence-wrapper cost for one fragment of ``payload``.

    A fragment carved out of the middle of a fenced block carries a synthetic
    re-open line (marker + language) AND a synthetic close line, so reserve the
    costliest open line plus its bare close marker among the payload's fences.
    Payloads with no fences reserve nothing.
    """
    reserve = 0
    for line in payload.split("\n"):
        marker = _fence_marker(line)
        if marker is None:
            continue
        char, length = marker
        open_cost = tokenizer.count(line.strip())
        close_cost = tokenizer.count(char * length)
        reserve = max(reserve, open_cost + close_cost)
    return reserve


def _fragment_body_text(fragment):
    """Reassemble a fragment's raw body: synthetic re-open, payload lines,
    synthetic close — valid fenced Markdown by the splitter's contract."""
    lines = []
    if fragment.synthetic.get("fence_open"):
        lines.append(fragment.synthetic["fence_open"])
    lines.extend(fragment.raw_lines)
    if fragment.synthetic.get("fence_close"):
        lines.append(fragment.synthetic["fence_close"])
    return "\n".join(lines)


# The companion Markdown -> HTML renderer is shared with the legacy path so an
# extension change lands in exactly one place.
_render_markdown_html = chunk_markdown_html


def _chunk_section(*, source_type, title, breadcrumb, heading_display, payload,
                   base_offset, record_id, source_path, tokenizer,
                   stripped_xml=False, original_span=None, html_for=None):
    """Split one section payload and assemble the per-fragment S5 fields.

    Returns a list of partial chunk dicts (content/content_html/counts/spans/
    header); the caller adds identity, citation, and provenance fields.
    """
    header = build_s6_header(
        source_type=source_type, title=title, breadcrumb=breadcrumb,
        raw_first_line=heading_display, tokenizer=tokenizer,
    )
    reserved = (
        header.token_count
        + tokenizer.count(heading_display)
        + _fence_wrapper_reserve(payload, tokenizer)
    )
    budget = EFFECTIVE_BUDGET - reserved
    if budget <= 0:
        # The header + repeated heading + fence wrappers alone exhaust the model
        # window, so no source text can accompany them. Fail with a clear,
        # actionable error rather than the splitter's generic UnsplittableSpan.
        raise ValueError(
            f"S5 reservation leaves no room for content in {source_path} "
            f"({heading_display!r}): header+heading+fence wrappers reserve "
            f"{reserved} of {EFFECTIVE_BUDGET} WordPieces. Shorten the section "
            "heading or split the source upstream."
        )

    fragments = split_payload(
        payload, base_offset=base_offset, budget_fn=lambda _i: budget,
        tokenizer=tokenizer, source_path=source_path,
    )
    assert_spans_tile(payload, base_offset, fragments, source_path)

    # source_token_count is the whole section payload's count — identical for
    # every fragment, so compute it once (not once per fragment).
    source_token_count = tokenizer.count(payload)
    partials = []
    for index, fragment in enumerate(fragments):
        body_text = _fragment_body_text(fragment)
        if body_text.strip():
            raw_document = heading_display + "\n" + body_text
        else:
            raw_document = heading_display
        embedding_input = (
            header.text + "\n" + raw_document if header.text else raw_document
        )
        embedding_tokens = tokenizer.count(embedding_input)
        if embedding_tokens > EFFECTIVE_BUDGET:
            # Hard rule: model-side truncation is forbidden. The reservation
            # math should make this unreachable; if it fires, the build stops.
            raise RuntimeError(
                f"embedding input exceeds the model window: {embedding_tokens} "
                f"> {EFFECTIVE_BUDGET} WordPieces for {source_path} "
                f"({heading_display!r} fragment {index})"
            )

        if stripped_xml:
            # The payload is a synthetic rewrite (oversized XML replaced), split
            # at base_offset=0, so its fragment spans are payload-relative and do
            # NOT map to original record bytes. Anchor every stripped fragment to
            # the ORIGINAL section span so all emitted byte offsets — both the
            # source_start/end fields and the metadata spans — stay in one
            # record-absolute coordinate system.
            start_byte, end_byte = original_span
            spans = [[start_byte, end_byte]]
        else:
            start_byte = fragment.spans[0].start_byte
            end_byte = fragment.spans[-1].end_byte
            spans = [[s.start_byte, s.end_byte] for s in fragment.spans]

        partials.append({
            "content": raw_document,
            "content_html": (html_for or _render_markdown_html)(body_text)
                            if body_text.strip() else "",
            "s6_header": header.text,
            "source_token_count": source_token_count,
            "content_token_count": tokenizer.count(raw_document),
            "embedding_token_count": embedding_tokens,
            "source_record_id": record_id,
            "source_start_byte": start_byte,
            "source_end_byte": end_byte,
            "synthetic_wrapper_metadata": {
                "heading_line": True,
                "fence_open": fragment.synthetic.get("fence_open"),
                "fence_close": fragment.synthetic.get("fence_close"),
                "stripped_xml": stripped_xml,
                "spans": spans,
            },
            "token_estimate": estimate_tokens(raw_document),
        })
    return partials


# --- companion path ----------------------------------------------------------------

def chunk_companion_text(entry, upstream, text, tokenizer):
    """Chunk one verified companion Markdown text into S5 chunk records."""
    repo = upstream.get("repo", "").strip("/")
    page_key = "companion://{}/{}".format(repo, entry["path"])
    provenance = {
        "source_type": COMPANION_SOURCE_TYPE,
        "verification_status": COMPANION_VERIFICATION_STATUS,
        "upstream_repo": upstream.get("repo", ""),
        "upstream_commit": upstream.get("commit", ""),
        "source_path": entry["path"],
        "raw_url": entry.get("raw_url", ""),
        "latest_url": entry.get("latest_url", ""),
    }
    source_url = entry.get("blob_url", "")

    sections = split_markdown_sections_with_offsets(text)
    title = (
        entry.get("title")
        or _markdown_title(sections)
        or os.path.splitext(os.path.basename(entry["path"]))[0]
    )
    area = entry.get("area", "")
    breadcrumb = " > ".join(p for p in (COMPANION_CATEGORY, area, title) if p)

    sections = filter_sections(
        sections,
        allow_patterns=entry.get("sections"),
        drop_sections=entry.get("drop_sections"),
    )

    chunks = []
    for sec in sections:
        body = sec["body"]
        stripped = False
        if entry.get("strip_xml_blocks"):
            replaced = strip_xml_blocks(body)
            stripped = replaced != body
            body = replaced
        if not body.strip():
            # A heading with no body carries no source bytes to embed.
            continue
        heading_display = sec["heading"] or title
        partials = _chunk_section(
            source_type=COMPANION_SOURCE_TYPE, title=title, breadcrumb=breadcrumb,
            heading_display=heading_display, payload=body,
            # A stripped body is a rewritten payload: spans are payload-relative
            # and the chunk records the ORIGINAL section range instead.
            base_offset=0 if stripped else sec["body_start_byte"],
            record_id=page_key, source_path=entry["path"], tokenizer=tokenizer,
            stripped_xml=stripped,
            original_span=(sec["body_start_byte"], sec["body_end_byte"]),
        )
        for partial in partials:
            chunk = {
                "id": companion_chunk_id(entry["path"], len(chunks)),
                "title": title,
                "section_heading": heading_display,
                "breadcrumb": breadcrumb,
                "source_url": source_url,
                "page_key": page_key,
                "category": COMPANION_CATEGORY,
            }
            chunk.update(partial)
            for key in PROVENANCE_FIELDS:
                chunk[key] = provenance[key]
            chunks.append(chunk)
    return chunks


# --- official path -----------------------------------------------------------------

def _official_sections(input_dir):
    """Extract per-section text from every HTML file, in deterministic order."""
    from bs4 import BeautifulSoup  # chunk_docs already depends on bs4

    extracted = []
    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        sections = split_on_headings(soup)
        page_title = filename
        for sec in sections:
            if sec["heading_tag"] == "h1":
                page_title = sec["heading_text"]
                break
        for sec in sections:
            payload = elements_to_text(sec["elements"])
            if not payload.strip():
                continue
            breadcrumb = sec["breadcrumb"]
            title = (breadcrumb.split(" > ")[-1].strip() or page_title
                     if breadcrumb else page_title)
            extracted.append({
                "filename": filename,
                "page_key": derive_page_key(sec["source_url"], filename),
                "title": title,
                "heading_display": sec["heading_text"] or title,
                "payload": payload,
                "breadcrumb": breadcrumb,
                "source_url": sec["source_url"],
                "section_html": elements_to_html(sec["elements"]),
            })
    return extracted


def build_official_chunks(input_dir, tokenizer):
    """Chunk official HTML pages through the S5 pipeline.

    The per-page source record is the deterministic extracted text document:
    ``heading\\npayload`` blocks joined by blank lines, in document order.
    Splitting runs on the extracted section text (never on HTML); each
    fragment's HTML is the original section HTML when the section did not
    split, else a diagnostic-fidelity Markdown render of the fragment text
    (``content`` is what is embedded and served).
    """
    extracted = _official_sections(input_dir)

    # Assemble page records and each section's payload offset within them.
    byte_pos = {}
    for sec in extracted:
        record_id = "official:" + sec["page_key"]
        sec["record_id"] = record_id
        pos = byte_pos.get(record_id, 0)
        if pos:
            pos += len(b"\n\n")
        pos += len((sec["heading_display"] + "\n").encode("utf-8"))
        sec["base_offset"] = pos
        byte_pos[record_id] = pos + len(sec["payload"].encode("utf-8"))

    chunks = []
    per_file_index = {}
    for sec in extracted:
        partials = _chunk_section(
            source_type=OFFICIAL_SOURCE_TYPE, title=sec["title"], breadcrumb="",
            heading_display=sec["heading_display"], payload=sec["payload"],
            base_offset=sec["base_offset"], record_id=sec["record_id"],
            source_path=sec["filename"], tokenizer=tokenizer,
        )
        if len(partials) == 1:
            # Unsplit section: the original section HTML still maps one-to-one.
            # Split sections keep the diagnostic Markdown render of each
            # fragment's text (content is what is embedded and served).
            partials[0]["content_html"] = sec["section_html"]
        for partial in partials:
            index = per_file_index.get(sec["filename"], 0)
            per_file_index[sec["filename"]] = index + 1
            chunk = {
                "id": make_chunk_id(sec["filename"], index),
                "title": sec["title"],
                "section_heading": sec["heading_display"],
                "breadcrumb": sec["breadcrumb"],
                "source_url": sec["source_url"],
                "page_key": sec["page_key"],
                "category": detect_category(sec["breadcrumb"])
                            if sec["breadcrumb"] else "Unknown",
            }
            chunk.update(partial)
            chunk.update({
                "source_type": OFFICIAL_SOURCE_TYPE,
                "verification_status": OFFICIAL_VERIFICATION_STATUS,
                "upstream_repo": "",
                "upstream_commit": "",
                "source_path": "",
                "raw_url": "",
                "latest_url": "",
            })
            chunks.append(chunk)
    return chunks


# --- full pipeline ------------------------------------------------------------------

def build_s5_chunks(*, official_input_dir, companion_manifest_path,
                    companion_staging, companion_config_path, tokenizer):
    """Chunk both corpora through S5 and assign page-local chunk indices."""
    chunks = []
    if os.path.isdir(official_input_dir):
        chunks.extend(build_official_chunks(official_input_dir, tokenizer))
    else:
        print(f"WARNING: HTML input directory not found: {official_input_dir}")

    if companion_manifest_path:
        upstream, entries = verified_companion_texts(
            companion_manifest_path, companion_staging,
            config_path=companion_config_path,
        )
        for entry, staged_text in entries:
            file_chunks = chunk_companion_text(entry, upstream, staged_text,
                                               tokenizer)
            if not file_chunks:
                print(f"  WARNING: companion file produced no chunks: {entry['path']}")
            chunks.extend(file_chunks)
            print(f"[companion] {entry['path']}: {len(file_chunks)} chunk(s)")

    assign_chunk_indices(chunks)
    for chunk in chunks:
        assert chunk["page_key"], f"chunk {chunk['id']} has an empty page_key"
    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Chunk Boomi docs with the S5 tokenizer-aware pipeline"
    )
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="Input directory of official HTML files")
    parser.add_argument("--output", default="chunks_s5.jsonl")
    parser.add_argument("--companion-input", default=DEFAULT_COMPANION_STAGING)
    parser.add_argument("--companion-manifest", default=None)
    parser.add_argument("--companion-config", default=DEFAULT_COMPANION_CONFIG)
    parser.add_argument(
        "--snapshot", default=None,
        help="Frozen source snapshot directory; verifies then resolves ALL "
             "inputs from it (mutually exclusive with the input flags)",
    )
    parser.add_argument(
        "--tokenizer", choices=("pinned", "fake"), default="pinned",
        help="'pinned' loads the real MiniLM WordPiece tokenizer at the "
             "contract revision; 'fake' is the deterministic development/test "
             "double and must never be used for a release build",
    )
    args = parser.parse_args()

    if args.snapshot:
        from snapshot import apply_snapshot_to_args

        apply_snapshot_to_args(args, (
            ("--input", args.input, DEFAULT_INPUT),
            ("--companion-input", args.companion_input, DEFAULT_COMPANION_STAGING),
            ("--companion-manifest", args.companion_manifest, None),
            ("--companion-config", args.companion_config, DEFAULT_COMPANION_CONFIG),
        ))

    if args.tokenizer == "fake":
        # The deterministic double must never split a release corpus: its
        # counts are not the model's, so the 256-WordPiece guarantee would not
        # hold. Loud warning so a misuse is visible in build logs.
        print("WARNING: --tokenizer fake is a development/test double and must "
              "NOT be used for a release build (its WordPiece counts are not "
              "the pinned model's).", file=sys.stderr)
        tokenizer = FakeWordPieceTokenizer()
    else:
        tokenizer = load_tokenizer()

    chunks = build_s5_chunks(
        official_input_dir=args.input,
        companion_manifest_path=args.companion_manifest,
        companion_staging=args.companion_input,
        companion_config_path=args.companion_config or None,
        tokenizer=tokenizer,
    )
    if not chunks:
        print("FAILED: no chunks produced")
        sys.exit(1)

    with open(args.output, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    max_embed = max(c["embedding_token_count"] for c in chunks)
    print(f"S5 chunking complete: {len(chunks)} chunks -> {args.output}")
    print(f"max embedding_token_count: {max_embed} (budget {EFFECTIVE_BUDGET})")


if __name__ == "__main__":
    main()
