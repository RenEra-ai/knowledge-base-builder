#!/usr/bin/env python3

import argparse
import json
import os
import re
import time
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

DEFAULT_CONFIG = "config.json"
DEFAULT_OUTPUT_DIR = "knowledge_base"
DEFAULT_DELAY = 1.0


def load_urls(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["urls"]


def make_client():
    return httpx.Client(
        follow_redirects=True,
        timeout=30.0,
        headers={
            "User-Agent": "KnowledgeBaseBuilder/2.0 (Boomi Documentation Indexer)"
        },
    )


def count_urls(url_list):
    total = 0
    for obj in url_list:
        total += 1
        total += count_urls(obj.get("children", []))
    return total


def parse_root_indices(raw_value, total_roots):
    if not raw_value:
        return None

    indices = []
    seen = set()
    for part in raw_value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            index = int(part)
        except ValueError as exc:
            raise ValueError(f"Invalid root index: {part}") from exc

        if index < 0 or index >= total_roots:
            raise ValueError(
                f"Root index out of range: {index} (expected 0..{total_roots - 1})"
            )
        if index not in seen:
            seen.add(index)
            indices.append(index)

    return sorted(indices)


def select_roots(urls, root_indices):
    if root_indices is None:
        return urls
    return [urls[index] for index in root_indices]


def fetch_html(url, client, failed_urls, delay, retries=3):
    time.sleep(delay)
    for attempt in range(retries):
        try:
            response = client.get(url)
            if response.status_code == 404:
                print(f"  WARNING: 404 for {url}")
                failed_urls.append(url)
                return ""
            response.raise_for_status()
            return response.text
        except httpx.HTTPError as exc:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  Retry {attempt + 1}/{retries} for {url} (waiting {wait}s): {exc}")
                time.sleep(wait)
            else:
                print(f"  FAILED after {retries} attempts: {url} - {exc}")
                failed_urls.append(url)
                return ""


def extract_article_content(html):
    soup = BeautifulSoup(html, "html.parser")
    content_div = soup.find("div", class_="theme-doc-markdown markdown")
    return str(content_div) if content_div else ""


def fix_relative_urls(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        parsed_href = urlparse(href)
        if not parsed_href.scheme and not parsed_href.netloc:
            anchor["href"] = urljoin(base_url, href)
        else:
            anchor["href"] = href
    for image in soup.find_all("img", src=True):
        src = image["src"]
        if src.startswith("/"):
            image["src"] = urljoin(base_url, src)
    return str(soup)


def remove_class_id_and_svg(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(True):
        tag.attrs.pop("class", None)
        tag.attrs.pop("id", None)
    for svg in soup.find_all("svg"):
        svg.decompose()
    return str(soup)


def build_breadcrumbs(path):
    return " > ".join(f"<a href='{url}'>{title}</a>" for title, url in path)


def sanitize_filename(title):
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", title) + ".html"


def process_url(url_obj, path, indent_level, client, failed_urls, delay, state):
    state["processed"] += 1
    url = url_obj["url"]
    print(f"[{state['processed']}/{state['total']}] Fetching: {url[:80]}...")

    parsed = urlparse(url)
    page_base_url = f"{parsed.scheme}://{parsed.netloc}"

    html = fetch_html(url, client, failed_urls, delay)
    article_content = extract_article_content(html)
    article_content = fix_relative_urls(article_content, page_base_url)
    article_content = remove_class_id_and_svg(article_content)

    soup = BeautifulSoup(article_content, "html.parser")
    first_heading = soup.find("h1")
    title = first_heading.text if first_heading else url
    filename = sanitize_filename(title)

    path.append((title, url))
    breadcrumbs = build_breadcrumbs(path)
    html_output = f"<h{indent_level + 1}>{title}</h{indent_level + 1}>\n\n"
    html_output += f"<p><strong>Path:</strong> {breadcrumbs}</p>\n\n"

    if first_heading:
        first_heading.extract()
    html_output += str(soup) + "\n\n"

    for child in url_obj.get("children", []):
        _, child_content = process_url(
            child,
            path[:],
            indent_level + 1,
            client,
            failed_urls,
            delay,
            state,
        )
        html_output += child_content

    path.pop()
    return filename, html_output


def write_failed_urls(output_dir, failed_urls):
    if not failed_urls:
        return

    print(f"\n{'=' * 60}")
    print(f"WARNING: {len(failed_urls)} URLs failed:")
    for url in failed_urls:
        print(f"  - {url}")

    failed_path = os.path.join(output_dir, "_failed_urls.txt")
    with open(failed_path, "w", encoding="utf-8") as f:
        f.write("\n".join(failed_urls))


def main():
    parser = argparse.ArgumentParser(description="Scrape Boomi documentation pages")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Input config.json path")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where HTML files are written",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help="Delay between requests in seconds",
    )
    parser.add_argument(
        "--root-indices",
        help="Comma-separated top-level root indices to process from config.json",
    )
    args = parser.parse_args()

    urls = load_urls(args.config)
    try:
        root_indices = parse_root_indices(args.root_indices, len(urls))
    except ValueError as exc:
        raise SystemExit(f"FAILED: {exc}") from exc

    selected_roots = select_roots(urls, root_indices)
    total_urls = count_urls(selected_roots)

    if root_indices is None:
        print(f"Processing all {len(selected_roots)} top-level roots ({total_urls} URLs)")
    else:
        print(
            "Processing selected top-level roots "
            f"{root_indices} ({len(selected_roots)} roots, {total_urls} URLs)"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    client = make_client()
    failed_urls = []
    state = {"processed": 0, "total": total_urls}

    try:
        for url_obj in selected_roots:
            filename, content = process_url(
                url_obj,
                [],
                0,
                client,
                failed_urls,
                args.delay,
                state,
            )
            file_path = os.path.join(args.output_dir, filename)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
    finally:
        client.close()
        write_failed_urls(args.output_dir, failed_urls)

    print("HTML files generated successfully")


if __name__ == "__main__":
    main()
