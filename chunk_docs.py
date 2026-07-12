#!/usr/bin/env python3
"""
Split scraped Boomi HTML docs into semantically meaningful chunks.

Reads HTML files from knowledge_base/, splits on heading boundaries (any
heading level — recursive scrapes nest child pages under deeper headings),
extracts metadata, and outputs chunks.jsonl in JSON Lines format.

Usage:
    python chunk_docs.py [--input knowledge_base/] [--output chunks.jsonl]
"""

import argparse
import json
import os
import re

from bs4 import BeautifulSoup, NavigableString, Tag

from companion import (
    COMPANION_CATEGORY,
    COMPANION_SOURCE_TYPE,
    COMPANION_VERIFICATION_STATUS,
    OFFICIAL_SOURCE_TYPE,
    OFFICIAL_VERIFICATION_STATUS,
    PROVENANCE_FIELDS,
    companion_chunk_id,
    curation_drift,
    filter_sections,
    is_denied_path,
    split_markdown_sections,
    strip_xml_blocks,
)

DEFAULT_INPUT = "knowledge_base/"
DEFAULT_OUTPUT = "chunks.jsonl"
DEFAULT_COMPANION_STAGING = "companion_sources/"
DEFAULT_COMPANION_CONFIG = "companion_sources.json"
DEFAULT_MIN_TOKENS = 100
DEFAULT_MAX_TOKENS = 1200
HEADING_TAGS = {f"h{i}" for i in range(1, 9)}


def unwrap_divs(soup):
    """Unwrap all <div> elements so headings/paragraphs become top-level children."""
    for div in soup.find_all("div"):
        div.unwrap()


def estimate_tokens(text):
    """Estimate token count as len(text) / 4."""
    return len(text) // 4


def detect_category(breadcrumb):
    """Extract top-level category from breadcrumb path.

    "Integration > Process Building > ..." -> "Integration"
    """
    parts = breadcrumb.split(" > ")
    return parts[0].strip() if parts else "Unknown"


def make_chunk_id(filename, index):
    """Generate a unique chunk ID from filename and index."""
    base = os.path.splitext(filename)[0]
    base = re.sub(r'[^a-zA-Z0-9_]', '_', base).strip('_').lower()
    if len(base) > 60:
        base = base[:60].rstrip('_')
    return f"{base}_{index:03d}"


def derive_page_key(source_url, filename):
    """Return a stable, non-empty page identity for a chunk.

    Uses the normalized source URL when available, otherwise falls back to a
    deterministic key derived from the source HTML filename. The result is
    always non-empty so downstream lookup and diversification never depend on
    source_url being present.
    """
    if source_url:
        normalized = source_url.strip()
        normalized = normalized.split("#", 1)[0]
        normalized = normalized.split("?", 1)[0]
        normalized = normalized.rstrip("/")
        if normalized:
            return normalized
    return "file:" + os.path.splitext(filename)[0]


def parse_breadcrumb(element):
    """Extract breadcrumb text and source URL from a Path paragraph element.

    Returns (breadcrumb_text, source_url) tuple.
    """
    links = element.find_all("a")
    parts = []
    source_url = ""
    for link in links:
        text = link.get_text(strip=True)
        href = link.get("href", "")
        if text:
            parts.append(text)
        if href:
            source_url = href
    breadcrumb = " > ".join(parts)
    return breadcrumb, source_url


def is_breadcrumb_element(element):
    """Check if an element is a Path: breadcrumb paragraph."""
    if element.name != "p":
        return False
    strong = element.find("strong")
    return bool(strong and strong.get_text(strip=True).startswith("Path:"))


def split_on_headings(soup):
    """Walk top-level children of soup, splitting into sections on heading boundaries.

    Returns a list of sections. Each section is a dict:
        {
            "heading_tag": "h1" .. "h8" | None,
            "heading_text": str,
            "breadcrumb": str,
            "source_url": str,
            "elements": [list of BS4 elements in this section],
        }
    """
    unwrap_divs(soup)
    sections = []
    current = {
        "heading_tag": None,
        "heading_text": "",
        "breadcrumb": "",
        "source_url": "",
        "elements": [],
    }

    active_breadcrumb = ""
    active_source_url = ""

    for child in soup.children:
        if isinstance(child, NavigableString):
            text = child.strip()
            if text:
                current["elements"].append(child)
            continue

        if not isinstance(child, Tag):
            continue

        if child.name in HEADING_TAGS:
            if current["elements"] or current["heading_text"]:
                if not current["breadcrumb"]:
                    current["breadcrumb"] = active_breadcrumb
                    current["source_url"] = active_source_url
                sections.append(current)

            current = {
                "heading_tag": child.name,
                "heading_text": child.get_text(strip=True),
                "breadcrumb": "",
                "source_url": "",
                "elements": [],
            }
            continue

        if is_breadcrumb_element(child):
            breadcrumb, source_url = parse_breadcrumb(child)
            current["breadcrumb"] = breadcrumb
            current["source_url"] = source_url
            active_breadcrumb = breadcrumb
            active_source_url = source_url
            continue

        current["elements"].append(child)

    if current["elements"] or current["heading_text"]:
        if not current["breadcrumb"]:
            current["breadcrumb"] = active_breadcrumb
            current["source_url"] = active_source_url
        sections.append(current)

    return sections


def elements_to_html(elements):
    """Convert a list of BS4 elements to an HTML string."""
    return "\n".join(str(el) for el in elements).strip()


def elements_to_text(elements):
    """Convert a list of BS4 elements to plain text."""
    parts = []
    for el in elements:
        if isinstance(el, NavigableString):
            text = str(el).strip()
            if text:
                parts.append(text)
        else:
            text = el.get_text(separator=" ", strip=True)
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def split_html_on_paragraphs(html_str, max_tokens):
    """Split an HTML string on paragraph boundaries to stay under max_tokens.

    Returns a list of HTML strings, each under max_tokens.
    """
    soup = BeautifulSoup(html_str, "html.parser")
    unwrap_divs(soup)
    children = [
        c for c in soup.children
        if isinstance(c, Tag) or (isinstance(c, NavigableString) and str(c).strip())
    ]

    if not children:
        return [html_str] if html_str.strip() else []

    chunks = []
    current_parts = []
    current_text = ""

    for child in children:
        if isinstance(child, NavigableString):
            child_html = str(child)
            child_text = str(child).strip()
        else:
            child_html = str(child)
            child_text = child.get_text(separator=" ", strip=True)

        combined_text = (current_text + "\n" + child_text).strip() if current_text else child_text

        if estimate_tokens(combined_text) > max_tokens and current_parts:
            chunks.append("\n".join(current_parts).strip())
            current_parts = [child_html]
            current_text = child_text
        else:
            current_parts.append(child_html)
            current_text = combined_text

    if current_parts:
        chunks.append("\n".join(current_parts).strip())

    return chunks


def chunk_file(filepath, filename, min_tokens, max_tokens, verbose):
    """Process a single HTML file and return a list of chunk dicts."""
    with open(filepath, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    sections = split_on_headings(soup)

    if not sections:
        if verbose:
            print(f"  WARNING: No sections found in {filename}")
        return []

    page_title = filename
    for sec in sections:
        if sec["heading_tag"] == "h1":
            page_title = sec["heading_text"]
            break

    raw_chunks = []
    for sec in sections:
        content_html = elements_to_html(sec["elements"])
        content_text = elements_to_text(sec["elements"])

        heading = sec["heading_text"]
        if heading and sec["heading_tag"] != "h1":
            full_text = heading + "\n" + content_text if content_text else heading
        else:
            full_text = content_text

        raw_chunks.append({
            "heading_tag": sec["heading_tag"],
            "heading_text": heading,
            "breadcrumb": sec["breadcrumb"],
            "source_url": sec["source_url"],
            "page_key": derive_page_key(sec["source_url"], filename),
            "content_text": full_text,
            "content_html": content_html,
        })

    merged = merge_small_chunks(raw_chunks, min_tokens)

    final_raw = []
    for chunk in merged:
        tokens = estimate_tokens(chunk["content_text"])
        if tokens > max_tokens:
            sub_htmls = split_html_on_paragraphs(chunk["content_html"], max_tokens)
            for i, sub_html in enumerate(sub_htmls):
                sub_soup = BeautifulSoup(sub_html, "html.parser")
                sub_text = sub_soup.get_text(separator=" ", strip=True)
                heading = chunk["heading_text"]
                if heading:
                    sub_full_text = heading + "\n" + sub_text if sub_text else heading
                else:
                    sub_full_text = sub_text
                final_raw.append({
                    "heading_text": chunk["heading_text"],
                    "breadcrumb": chunk["breadcrumb"],
                    "source_url": chunk["source_url"],
                    "page_key": chunk["page_key"],
                    "content_text": sub_full_text,
                    "content_html": sub_html,
                })
        else:
            final_raw.append(chunk)

    chunks = []
    for i, raw in enumerate(final_raw):
        content_text = raw["content_text"]
        token_est = estimate_tokens(content_text)

        breadcrumb = raw["breadcrumb"]
        category = detect_category(breadcrumb) if breadcrumb else "Unknown"

        # Title is page-local: a recursive HTML file holds many child pages, so
        # the file-level <h1> is only correct for the root page. The breadcrumb's
        # last segment names the page the chunk actually belongs to; fall back to
        # the file-level title when a section carries no breadcrumb.
        if breadcrumb:
            title = breadcrumb.split(" > ")[-1].strip() or page_title
        else:
            title = page_title

        if not content_text.strip():
            # Heading-only / URL-only landing chunks have no semantic body to
            # index. Drop them rather than emit content the validator rejects.
            continue

        chunk = {
            "id": make_chunk_id(filename, i),
            "title": title,
            "section_heading": raw["heading_text"] or title,
            "breadcrumb": breadcrumb,
            "source_url": raw["source_url"],
            "page_key": raw["page_key"],
            "category": category,
            "content": content_text,
            "content_html": raw["content_html"],
            "token_estimate": token_est,
            # Official docs carry blank provenance for all five extended fields
            # (source_url is their citation); companion chunks override these.
            "source_type": OFFICIAL_SOURCE_TYPE,
            "verification_status": OFFICIAL_VERIFICATION_STATUS,
            "upstream_repo": "",
            "upstream_commit": "",
            "source_path": "",
            "raw_url": "",
            "latest_url": "",
        }
        chunks.append(chunk)

    return chunks


def merge_small_chunks(raw_chunks, min_tokens):
    """Merge chunks that are below min_tokens with their neighbors."""
    if not raw_chunks:
        return []

    merged = [raw_chunks[0]]

    for chunk in raw_chunks[1:]:
        prev = merged[-1]
        prev_tokens = estimate_tokens(prev["content_text"])
        # Merge boundaries are page identity, not citation metadata: page_key is
        # always non-empty, so two sections only merge when they are the same page.
        same_page = prev["page_key"] == chunk["page_key"]

        if prev_tokens < min_tokens and same_page:
            prev_had_content = bool(prev["content_text"].strip())
            prev["content_text"] = (prev["content_text"] + "\n" + chunk["content_text"]).strip()
            prev["content_html"] = (prev["content_html"] + "\n" + chunk["content_html"]).strip()
            # If the previous chunk was heading-only (e.g. the page-title <h1>
            # with no body before the first <h2>), the merged chunk's real
            # section is the incoming one, so its heading should win.
            if chunk["heading_text"] and (not prev["heading_text"] or not prev_had_content):
                prev["heading_text"] = chunk["heading_text"]
            if not prev["breadcrumb"] and chunk["breadcrumb"]:
                prev["breadcrumb"] = chunk["breadcrumb"]
                prev["source_url"] = chunk["source_url"]
        else:
            merged.append(chunk)

    if len(merged) > 1 and estimate_tokens(merged[-1]["content_text"]) < min_tokens:
        last = merged[-1]
        prev = merged[-2]
        same_page = prev["page_key"] == last["page_key"]
        if same_page:
            merged.pop()
            prev["content_text"] = (prev["content_text"] + "\n" + last["content_text"]).strip()
            prev["content_html"] = (prev["content_html"] + "\n" + last["content_html"]).strip()

    return merged


def assign_chunk_indices(all_chunks):
    """Assign a 0-based, page-local chunk_index to every chunk, in document order.

    all_chunks is already in document order: files are processed in sorted order
    and chunk_file emits chunks in document order. A single HTML file can
    contribute chunks to multiple page_keys (the breadcrumb/source_url can change
    mid-file), so the counter is keyed on page_key, not on the file.
    """
    page_counters = {}
    for chunk in all_chunks:
        page_key = chunk["page_key"]
        index = page_counters.get(page_key, 0)
        chunk["chunk_index"] = index
        page_counters[page_key] = index + 1


def _markdown_title(sections):
    """First level-1 heading among parsed Markdown sections, else ''."""
    for sec in sections:
        if sec["level"] == 1 and sec["heading"]:
            return sec["heading"]
    return ""


def chunk_markdown_file(md_path, entry, upstream, min_tokens, max_tokens, verbose=False):
    """Chunk one companion Markdown file into provenance-tagged chunk dicts.

    Reuses the HTML path's merge/oversize-split machinery (merge_small_chunks,
    split_html_on_paragraphs, estimate_tokens, the empty-content drop) while
    carrying the companion provenance fields, a stable ``companion://`` page_key,
    and the "Companion Reference" category. Section allow/deny filtering and
    raw-XML stripping run before emit, so they only change *which* sections
    exist — never the contiguous per-page chunk_index assigned later in main().
    """
    import markdown  # lazy: the HTML path must stay importable without markdown

    with open(md_path, "r", encoding="utf-8") as f:
        md = f.read()

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
    # source_url is the human-viewable pinned blob permalink for companion chunks.
    source_url = entry.get("blob_url", "")

    sections = split_markdown_sections(md)
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

    raw_chunks = []
    for sec in sections:
        body = strip_xml_blocks(sec["body"]) if entry.get("strip_xml_blocks") else sec["body"]
        heading = sec["heading"]
        # Mirror the HTML path: prepend the heading for sub-sections (>= h2) so a
        # section's title is embedded with its body; the h1/preamble is body-only.
        if heading and sec["level"] >= 2:
            content_text = heading + "\n" + body if body.strip() else heading
        else:
            content_text = body
        # content_html is body-only (heading excluded), mirroring the HTML path's
        # elements_to_html. The oversize split re-prepends the heading exactly
        # once; rendering from heading-inclusive content_text would duplicate it.
        content_html = markdown.markdown(body, extensions=["fenced_code", "tables"])

        raw = {
            "heading_text": heading,
            "breadcrumb": breadcrumb,
            "source_url": source_url,
            "page_key": page_key,
            "content_text": content_text,
            "content_html": content_html,
        }
        raw.update(provenance)
        raw_chunks.append(raw)

    merged = merge_small_chunks(raw_chunks, min_tokens)

    final_raw = []
    for chunk in merged:
        if estimate_tokens(chunk["content_text"]) > max_tokens:
            sub_htmls = split_html_on_paragraphs(chunk["content_html"], max_tokens)
            for sub_html in sub_htmls:
                sub_soup = BeautifulSoup(sub_html, "html.parser")
                sub_text = sub_soup.get_text(separator=" ", strip=True)
                heading = chunk["heading_text"]
                if heading:
                    sub_full = heading + "\n" + sub_text if sub_text else heading
                else:
                    sub_full = sub_text
                sub = dict(chunk)
                sub["content_text"] = sub_full
                sub["content_html"] = sub_html
                final_raw.append(sub)
        else:
            final_raw.append(chunk)

    chunks = []
    for raw in final_raw:
        content_text = raw["content_text"]
        if not content_text.strip():
            continue
        idx = len(chunks)
        chunk = {
            "id": companion_chunk_id(entry["path"], idx),
            "title": title,
            "section_heading": raw["heading_text"] or title,
            "breadcrumb": breadcrumb,
            "source_url": source_url,
            "page_key": page_key,
            "category": COMPANION_CATEGORY,
            "content": content_text,
            "content_html": raw["content_html"],
            "token_estimate": estimate_tokens(content_text),
        }
        for key in PROVENANCE_FIELDS:
            chunk[key] = raw[key]
        chunks.append(chunk)

    return chunks


def process_companion(manifest_path, staging_dir, min_tokens, max_tokens, verbose=False,
                      config_path=None):
    """Load a companion manifest and chunk every staged Markdown file it lists.

    Fails if the manifest or a staged file is missing (every companion source is
    mandatory); warns if an allowlisted file yields no chunks.

    The section allow/deny filters are read from the MANIFEST, which fetch_companion
    snapshotted from companion_sources.json at fetch time. When ``config_path`` is
    given, refuse to build on a manifest whose curation policy has since drifted from
    the config: chunking a stale allowlist fails OPEN (the sections a curator just
    excluded quietly stay in the corpus) and the build otherwise reports success.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        drift = curation_drift(config.get("files", []), manifest.get("files", []))
        if drift:
            raise ValueError(
                "Companion curation policy has drifted from the manifest — the "
                "manifest was written before the current companion_sources.json.\n  "
                + "\n  ".join(drift)
                + f"\nRe-run: python fetch_companion.py --config {config_path} "
                  f"--manifest {manifest_path}"
            )

    upstream = {"repo": manifest["repo"], "commit": manifest["commit"]}
    all_companion = []
    for entry in manifest["files"]:
        path = entry["path"]
        # Re-validate even though fetch_companion already did: an absolute or
        # ../ path in a corrupted manifest would escape the staging dir via
        # os.path.join. Fail closed rather than read an arbitrary file.
        if is_denied_path(path):
            raise ValueError(f"Companion manifest contains a disallowed path: {path!r}")
        md_path = os.path.join(staging_dir, path)
        if not os.path.isfile(md_path):
            raise FileNotFoundError(f"Staged companion file missing: {md_path}")
        file_chunks = chunk_markdown_file(
            md_path, entry, upstream, min_tokens, max_tokens, verbose
        )
        if not file_chunks:
            print(f"  WARNING: companion file produced no chunks: {entry['path']}")
        all_companion.extend(file_chunks)
        print(f"[companion] {entry['path']}: {len(file_chunks)} chunk(s)")

    return all_companion


def main():
    parser = argparse.ArgumentParser(description="Chunk Boomi HTML docs for indexing")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input directory of HTML files")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSONL file")
    parser.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_TOKENS, help="Minimum chunk size in tokens")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Maximum chunk size in tokens")
    parser.add_argument("--verbose", action="store_true", help="Print each chunk as it is created")
    parser.add_argument(
        "--companion-input", default=DEFAULT_COMPANION_STAGING,
        help="Directory of staged companion Markdown (from fetch_companion.py)",
    )
    parser.add_argument(
        "--companion-manifest", default=None,
        help="companion_manifest.json describing the staged companion files; "
             "enables the supplemental Companion corpus when provided",
    )
    parser.add_argument(
        "--companion-config", default=DEFAULT_COMPANION_CONFIG,
        help="companion_sources.json to check the manifest's curation policy against; "
             "the build fails if the manifest was written before a policy edit. "
             "Pass '' to skip the check",
    )
    args = parser.parse_args()

    all_chunks = []
    category_counts = {}

    if os.path.isdir(args.input):
        html_files = sorted(f for f in os.listdir(args.input) if f.endswith(".html"))
    else:
        html_files = []
        print(f"WARNING: HTML input directory not found: {args.input}")

    if not html_files and not args.companion_manifest:
        print(f"WARNING: No HTML files found in {args.input} and no companion manifest")
        return

    total_files = len(html_files)
    for i, filename in enumerate(html_files):
        filepath = os.path.join(args.input, filename)
        print(f"[{i + 1}/{total_files}] Processing: {filename}")

        try:
            chunks = chunk_file(filepath, filename, args.min_tokens, args.max_tokens, args.verbose)
            for chunk in chunks:
                cat = chunk["category"]
                category_counts[cat] = category_counts.get(cat, 0) + 1

                if args.verbose:
                    print(f"  Chunk {chunk['id']}: {chunk['section_heading'][:50]} ({chunk['token_estimate']} tokens)")

            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  FAILED: {filename} — {e}")
            continue

    if args.companion_manifest:
        print(f"\nProcessing companion sources from {args.companion_manifest}...")
        companion_chunks = process_companion(
            args.companion_manifest, args.companion_input,
            args.min_tokens, args.max_tokens, args.verbose,
            config_path=args.companion_config or None,
        )
        for chunk in companion_chunks:
            category_counts[chunk["category"]] = category_counts.get(chunk["category"], 0) + 1
        all_chunks.extend(companion_chunks)

    assign_chunk_indices(all_chunks)
    for chunk in all_chunks:
        assert chunk["page_key"], f"chunk {chunk['id']} has an empty page_key"

    with open(args.output, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    token_counts = [c["token_estimate"] for c in all_chunks]
    avg_tokens = sum(token_counts) // len(token_counts) if token_counts else 0
    min_tok = min(token_counts) if token_counts else 0
    max_tok = max(token_counts) if token_counts else 0
    output_size = os.path.getsize(args.output)

    if output_size >= 1024 * 1024:
        size_str = f"{output_size / (1024 * 1024):.1f} MB"
    else:
        size_str = f"{output_size / 1024:.1f} KB"

    cat_summary = ", ".join(f"{k} ({v})" for k, v in sorted(category_counts.items(), key=lambda x: -x[1]))

    print(f"\n{'=' * 60}")
    print("Chunking Complete")
    print(f"{'=' * 60}")
    print(f"Input files:      {total_files}")
    print(f"Total chunks:     {len(all_chunks):,}")
    print(f"Avg chunk size:   {avg_tokens} tokens")
    print(f"Min chunk size:   {min_tok} tokens")
    print(f"Max chunk size:   {max_tok} tokens")
    print(f"Categories:       {cat_summary}")
    print(f"Output:           {args.output} ({size_str})")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
