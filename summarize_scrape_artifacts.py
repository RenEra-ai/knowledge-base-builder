#!/usr/bin/env python3

import argparse
import re
import tarfile
import tempfile
from pathlib import Path


ARCHIVE_RE = re.compile(r"knowledge-base-shard-(\d+)\.tar\.gz$")


def find_archives(artifacts_dir):
    archives = []
    for path in Path(artifacts_dir).glob("knowledge-base-shard-*.tar.gz"):
        match = ARCHIVE_RE.match(path.name)
        if match:
            archives.append((int(match.group(1)), path))
    return sorted(archives)


def inspect_archive(archive_path):
    html_count = 0
    failed_urls = []

    with tempfile.TemporaryDirectory(prefix="scrape-report-") as tmpdir:
        tmp_path = Path(tmpdir)
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(tmp_path)

        for html_path in tmp_path.rglob("*.html"):
            if html_path.is_file():
                html_count += 1

        for failed_path in tmp_path.rglob("failed_urls_shard_*.txt"):
            with open(failed_path, "r", encoding="utf-8") as f:
                for line in f:
                    url = line.strip()
                    if url:
                        failed_urls.append(url)

    return html_count, failed_urls


def write_failed_urls(output_path, failed_urls):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    unique_urls = list(dict.fromkeys(failed_urls))
    with open(output, "w", encoding="utf-8") as f:
        if unique_urls:
            f.write("\n".join(unique_urls))


def build_report(artifacts_dir, expected_shards):
    archives = find_archives(artifacts_dir)
    found_shards = {shard_id for shard_id, _ in archives}
    expected = set(range(expected_shards))
    missing = sorted(expected - found_shards)

    lines = [
        "# Scrape Report",
        "",
        f"- Expected shards: {expected_shards}",
        f"- Completed shards: {len(found_shards)}/{expected_shards}",
        f"- Missing shards: {', '.join(map(str, missing)) if missing else 'none'}",
        "",
        "## Shard Details",
    ]

    all_failed_urls = []

    if not archives:
        lines.append("")
        lines.append("No shard artifacts were available.")
        return "\n".join(lines) + "\n", all_failed_urls

    for shard_id, archive_path in archives:
        html_count, failed_urls = inspect_archive(archive_path)
        all_failed_urls.extend(failed_urls)
        lines.append(
            f"- Shard {shard_id}: {html_count} HTML files, {len(failed_urls)} failed URLs"
        )

    unique_failed_urls = list(dict.fromkeys(all_failed_urls))
    lines.extend(
        [
            "",
            "## Failed URL Summary",
            "",
            f"- Total failed URLs: {len(unique_failed_urls)}",
        ]
    )

    return "\n".join(lines) + "\n", unique_failed_urls


def main():
    parser = argparse.ArgumentParser(
        description="Summarize downloaded scrape shard artifacts"
    )
    parser.add_argument("--artifacts-dir", default="artifacts", help="Input artifact directory")
    parser.add_argument(
        "--expected-shards",
        type=int,
        required=True,
        help="Expected number of shard artifacts",
    )
    parser.add_argument("--output", required=True, help="Markdown summary output path")
    parser.add_argument(
        "--failed-urls-output",
        help="Optional path for the aggregated failed URL report",
    )
    args = parser.parse_args()

    report, failed_urls = build_report(args.artifacts_dir, args.expected_shards)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write(report)

    if args.failed_urls_output:
        write_failed_urls(args.failed_urls_output, failed_urls)


if __name__ == "__main__":
    main()
