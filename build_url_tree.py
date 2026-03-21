#!/usr/bin/env python3
"""
Crawl Boomi documentation sidebar to generate config.json URL tree.

Usage:
    python build_url_tree.py [--output config.json] [--validate] [--verbose]
"""

import argparse
import json
import os
import random
import shutil
import time
from urllib.parse import quote

import httpx
from bs4 import BeautifulSoup

BASE_URL = "https://help.boomi.com"

ROOT_URLS = {
    "Integration": BASE_URL + "/docs/Atomsphere/Integration/Getting%20started/"
    "c-atm-Integration_and_iPaaS_257fcf2c-7e93-48d0-be67-bd53fb444930",
    "Connectors": BASE_URL + "/docs/Atomsphere/Integration/Connectors/connectors_overview",
    "Flow": BASE_URL + "/docs/Atomsphere/Flow/Flow_overview",
    "Event Streams": BASE_URL + "/docs/Atomsphere/Event%20Streams/event_streams_overview",
    "Boomi for SAP": BASE_URL + "/docs/Atomsphere/Boomi_for_SAP/boomi_for_sap_overview",
}


def make_client():
    return httpx.Client(
        follow_redirects=True,
        timeout=30.0,
        headers={"User-Agent": "KnowledgeBaseBuilder/2.0 (Boomi Documentation Indexer)"},
    )


def fetch_page(url, client, delay, retries=3):
    """Fetch a page with retry logic and rate limiting."""
    time.sleep(delay)
    for attempt in range(retries):
        try:
            response = client.get(url)
            if response.status_code == 404:
                print(f"  WARNING: 404 for {url}")
                return None
            response.raise_for_status()
            return response.text
        except httpx.HTTPError as e:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  Retry {attempt + 1}/{retries} (waiting {wait}s): {e}")
                time.sleep(wait)
            else:
                print(f"  FAILED after {retries} attempts: {url} — {e}")
                return None


def normalize_url(href):
    """Convert sidebar href to full absolute URL with proper encoding."""
    if href.startswith("http"):
        return href
    return BASE_URL + quote(href, safe="/:@")


def parse_sidebar_links(soup):
    """Extract all sidebar links with hierarchy levels in document order.

    Uses the level-N class on parent <li> elements to determine depth,
    avoiding reliance on DOM nesting which can break with malformed HTML.
    """
    sidebar = soup.find("nav", class_="menu")
    if not sidebar:
        return []

    items = []
    seen = {}  # href -> index in items list

    for link in sidebar.find_all("a", class_="menu__link"):
        href = link.get("href", "")
        if not href or href.startswith("#"):
            continue
        if href.startswith("http") and "help.boomi.com" not in href:
            continue

        is_current = link.get("aria-current") == "page"

        # Some pages have two sidebar links with the same href (category + page).
        # If we already saw this href, just update is_current if the new one has it.
        if href in seen:
            if is_current:
                items[seen[href]]["is_current"] = True
            continue

        parent_li = link.find_parent("li")
        level = None
        if parent_li:
            for c in parent_li.get("class", []):
                if "level-" in c:
                    try:
                        level = int(c.rsplit("level-", 1)[-1])
                    except ValueError:
                        pass

        if level is None:
            continue

        classes = link.get("class", [])
        seen[href] = len(items)
        items.append(
            {
                "level": level,
                "href": href,
                "text": link.get_text(strip=True),
                "is_category": "menu__link--sublist" in classes,
                "is_current": is_current,
            }
        )

    return items


def get_top_level_items(soup):
    """Get all top-level (level-2) items from the sidebar."""
    return [i for i in parse_sidebar_links(soup) if i["level"] == 2]


def get_children_of_current_page(soup):
    """Get direct children of the current page in the sidebar.

    Uses a sequential scan: finds the current page by aria-current,
    then collects items at level+1 until hitting a same-or-higher level item.
    """
    items = parse_sidebar_links(soup)

    active_idx = None
    active_level = None
    for i, item in enumerate(items):
        if item["is_current"]:
            active_idx = i
            active_level = item["level"]
            break

    if active_idx is None:
        return []

    children = []
    for item in items[active_idx + 1 :]:
        if item["level"] <= active_level:
            break
        if item["level"] == active_level + 1:
            children.append(item)

    return children


def crawl_category(url, client, delay, verbose, seen_urls, depth=0):
    """Recursively crawl a category page to build its URL subtree."""
    if url in seen_urls:
        return None
    seen_urls.add(url)

    if verbose:
        indent = "  " * depth
        slug = url.split("/")[-1][:60]
        print(f"{indent}[cat] {slug}")

    html = fetch_page(url, client, delay)
    if not html:
        return {"url": url, "children": []}

    soup = BeautifulSoup(html, "html.parser")
    child_items = get_children_of_current_page(soup)

    if not child_items and verbose:
        print(f"{'  ' * depth}  (no children found)")

    children = []
    for child in child_items:
        child_url = normalize_url(child["href"])
        if child_url in seen_urls:
            continue

        if child["is_category"]:
            node = crawl_category(
                child_url, client, delay, verbose, seen_urls, depth + 1
            )
        else:
            seen_urls.add(child_url)
            node = {"url": child_url, "children": []}
            if verbose:
                print(f"{'  ' * (depth + 1)}[leaf] {child['text'][:60]}")

        if node:
            children.append(node)

    return {"url": url, "children": children}


def crawl_section(name, root_url, client, delay, verbose):
    """Crawl a complete documentation section starting from its root URL."""
    print(f"\nCrawling section: {name}")

    html = fetch_page(root_url, client, delay)
    if not html:
        print("  ERROR: Could not fetch root page")
        return []

    soup = BeautifulSoup(html, "html.parser")
    top_items = get_top_level_items(soup)
    print(f"  Found {len(top_items)} top-level items")

    seen_urls = set()
    result = []

    for item in top_items:
        url = normalize_url(item["href"])
        print(f"\n  >> {item['text']}")

        if item["is_category"]:
            node = crawl_category(url, client, delay, verbose, seen_urls, depth=1)
        else:
            seen_urls.add(url)
            node = {"url": url, "children": []}

        if node:
            result.append(node)

    return result


def flatten_urls(nodes):
    """Flatten a URL tree into a list of all URLs."""
    urls = []
    for node in nodes:
        urls.append(node["url"])
        urls.extend(flatten_urls(node.get("children", [])))
    return urls


def compute_depths(nodes, depth=1):
    """Compute the depth of every node in the tree."""
    depths = []
    for node in nodes:
        depths.append(depth)
        depths.extend(compute_depths(node.get("children", []), depth + 1))
    return depths


def print_stats(tree):
    """Print summary statistics for the generated tree."""
    all_urls = flatten_urls(tree)
    depths = compute_depths(tree)

    print(f"\n{'=' * 60}")
    print("URL Tree Generated Successfully")
    print(f"{'=' * 60}")
    print(f"Total URLs:         {len(all_urls)}")
    print(f"Top-level sections: {len(tree)}")
    if depths:
        print(f"Max depth:          {max(depths)}")
        print(f"Average depth:      {sum(depths) / len(depths):.1f}")
    print(f"{'=' * 60}")


def validate_sample(tree, client, sample_size=20):
    """Spot-check random URLs from the tree to verify they are live."""
    all_urls = flatten_urls(tree)
    sample = random.sample(all_urls, min(sample_size, len(all_urls)))

    print(f"\nValidating {len(sample)} random URLs...")
    ok, fail, failures = 0, 0, []
    for url in sample:
        time.sleep(0.5)
        try:
            r = client.get(url)
            if r.status_code == 200:
                ok += 1
                print(f"  OK:   ...{url.split('/')[-1][:50]}")
            else:
                fail += 1
                failures.append(f"{r.status_code}: {url}")
                print(f"  FAIL: ({r.status_code}) ...{url.split('/')[-1][:50]}")
        except Exception as e:
            fail += 1
            failures.append(f"ERROR: {url} — {e}")

    total = ok + fail
    print(f"\nValidation: {ok}/{total} passed ({ok / total * 100:.0f}%)")
    if failures:
        print("Failures:")
        for f in failures:
            print(f"  {f}")


def compare_trees(old_path, new_tree):
    """Compare old and new URL trees and print a delta report."""
    if not os.path.exists(old_path):
        return

    with open(old_path) as f:
        old = json.load(f)

    old_urls = set(flatten_urls(old.get("urls", [])))
    new_urls = set(flatten_urls(new_tree))

    added = new_urls - old_urls
    removed = old_urls - new_urls
    unchanged = old_urls & new_urls

    print(f"\nComparison with old config:")
    print(f"  Unchanged: {len(unchanged)}")
    print(f"  Added:     {len(added)}")
    print(f"  Removed:   {len(removed)}")

    if removed:
        print(f"\n  Removed (sample, {len(removed)} total):")
        for u in sorted(removed)[:15]:
            print(f"    - ...{u.split('/')[-1][:60]}")
        if len(removed) > 15:
            print(f"    ... and {len(removed) - 15} more")
    if added:
        print(f"\n  Added (sample, {len(added)} total):")
        for u in sorted(added)[:15]:
            print(f"    + ...{u.split('/')[-1][:60]}")
        if len(added) > 15:
            print(f"    ... and {len(added) - 15} more")


def main():
    parser = argparse.ArgumentParser(
        description="Build Boomi documentation URL tree from sidebar navigation"
    )
    parser.add_argument("--output", default="config.json", help="Output file path")
    parser.add_argument(
        "--sections",
        default="Integration,Connectors,Flow,Event Streams,Boomi for SAP",
        help="Comma-separated doc sections to crawl",
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate a sample of URLs after generation"
    )
    parser.add_argument(
        "--delay", type=float, default=1.0, help="Delay between requests in seconds"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print each URL as it is discovered"
    )
    args = parser.parse_args()

    sections = [s.strip() for s in args.sections.split(",")]

    client = make_client()
    try:
        tree = []
        for section in sections:
            if section not in ROOT_URLS:
                print(f"Unknown section: {section}")
                print(f"Available: {', '.join(ROOT_URLS.keys())}")
                continue
            tree.extend(
                crawl_section(section, ROOT_URLS[section], client, args.delay, args.verbose)
            )

        print_stats(tree)

        # Backup old config before overwriting
        if os.path.exists(args.output):
            bak = args.output + ".bak"
            shutil.copy2(args.output, bak)
            print(f"\nBacked up {args.output} -> {bak}")
            compare_trees(bak, tree)

        # Write new config
        config = {"urls": tree}
        with open(args.output, "w") as f:
            json.dump(config, f, indent=2)
        print(f"\nWrote {args.output}")

        if args.validate:
            validate_sample(tree, client)
    finally:
        client.close()


if __name__ == "__main__":
    main()
