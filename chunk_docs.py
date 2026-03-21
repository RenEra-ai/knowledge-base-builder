#!/usr/bin/env python3
"""
Split scraped Boomi HTML docs into semantically meaningful chunks.

Reads HTML files from knowledge_base/, splits on heading boundaries (h2/h3),
extracts metadata, and outputs chunks.jsonl in JSON Lines format.

Usage:
    python chunk_docs.py [--input knowledge_base/] [--output chunks.jsonl]
"""

import argparse
import json
import os
import re

from bs4 import BeautifulSoup, NavigableString, Tag

DEFAULT_INPUT = "knowledge_base/"
DEFAULT_OUTPUT = "chunks.jsonl"
DEFAULT_MIN_TOKENS = 100
DEFAULT_MAX_TOKENS = 1200
HEADING_TAGS = {"h1", "h2", "h3"}


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
    """Walk top-level children of soup, splitting into sections on h1/h2/h3 boundaries.

    Returns a list of sections. Each section is a dict:
        {
            "heading_tag": "h1" | "h2" | "h3" | None,
            "heading_text": str,
            "breadcrumb": str,
            "source_url": str,
            "elements": [list of BS4 elements in this section],
        }
    """
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

        chunk = {
            "id": make_chunk_id(filename, i),
            "title": page_title,
            "section_heading": raw["heading_text"] or page_title,
            "breadcrumb": breadcrumb,
            "source_url": raw["source_url"],
            "category": category,
            "content": content_text,
            "content_html": raw["content_html"],
            "token_estimate": token_est,
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

        if prev_tokens < min_tokens:
            prev["content_text"] = (prev["content_text"] + "\n" + chunk["content_text"]).strip()
            prev["content_html"] = (prev["content_html"] + "\n" + chunk["content_html"]).strip()
            if not prev["heading_text"] and chunk["heading_text"]:
                prev["heading_text"] = chunk["heading_text"]
            if not prev["breadcrumb"] and chunk["breadcrumb"]:
                prev["breadcrumb"] = chunk["breadcrumb"]
                prev["source_url"] = chunk["source_url"]
        else:
            merged.append(chunk)

    if len(merged) > 1 and estimate_tokens(merged[-1]["content_text"]) < min_tokens:
        last = merged.pop()
        merged[-1]["content_text"] = (merged[-1]["content_text"] + "\n" + last["content_text"]).strip()
        merged[-1]["content_html"] = (merged[-1]["content_html"] + "\n" + last["content_html"]).strip()

    return merged


def main():
    parser = argparse.ArgumentParser(description="Chunk Boomi HTML docs for indexing")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input directory of HTML files")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSONL file")
    parser.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_TOKENS, help="Minimum chunk size in tokens")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Maximum chunk size in tokens")
    parser.add_argument("--verbose", action="store_true", help="Print each chunk as it is created")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"FAILED: Input directory not found: {args.input}")
        return

    html_files = sorted(f for f in os.listdir(args.input) if f.endswith(".html"))
    if not html_files:
        print(f"WARNING: No HTML files found in {args.input}")
        return

    all_chunks = []
    total_files = len(html_files)
    category_counts = {}

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
