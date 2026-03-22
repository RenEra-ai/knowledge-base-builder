#!/usr/bin/env python3

import argparse
import os
import re
import shutil
import tarfile
import tempfile
from pathlib import Path


ARCHIVE_RE = re.compile(r"knowledge-base-shard-(\d+)\.tar\.gz$")
FAILED_URLS_RE = re.compile(r"failed_urls_shard_(\d+)\.txt$")


def find_archives(artifacts_dir):
    archives = []
    for path in Path(artifacts_dir).glob("knowledge-base-shard-*.tar.gz"):
        match = ARCHIVE_RE.match(path.name)
        if match:
            archives.append((int(match.group(1)), path))
    return sorted(archives)


def same_file_contents(path_a, path_b):
    with open(path_a, "rb") as f_a, open(path_b, "rb") as f_b:
        return f_a.read() == f_b.read()


def merge_tree(src_dir, dest_dir):
    for src_path in sorted(src_dir.rglob("*")):
        if not src_path.is_file():
            continue

        rel_path = src_path.relative_to(src_dir)
        dest_path = Path(dest_dir) / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if dest_path.exists():
            if same_file_contents(src_path, dest_path):
                continue
            raise RuntimeError(f"Conflicting assembled file: {dest_path}")

        shutil.copy2(src_path, dest_path)


def collect_failed_urls(extract_dir):
    failed_urls = []
    for path in sorted(extract_dir.rglob("failed_urls_shard_*.txt")):
        if not FAILED_URLS_RE.match(path.name):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                url = line.strip()
                if url:
                    failed_urls.append(url)
    return failed_urls


def write_failed_urls(output_path, failed_urls):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    unique_urls = list(dict.fromkeys(failed_urls))
    with open(output, "w", encoding="utf-8") as f:
        if unique_urls:
            f.write("\n".join(unique_urls))


# Use the safer Python 3.12+ tar extraction filter when available, but
# keep working on older interpreters that do not support the filter argument.
def extract_archive(archive, output_path):
    try:
        archive.extractall(path=output_path, filter="data")
    except TypeError:
        archive.extractall(path=output_path)


def assemble_archives(artifacts_dir, output_dir, failed_urls_output):
    archives = find_archives(artifacts_dir)
    if not archives:
        raise RuntimeError(f"No shard archives found in {artifacts_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    failed_urls = []

    for shard_id, archive_path in archives:
        with tempfile.TemporaryDirectory(prefix=f"scrape-shard-{shard_id}-") as tmpdir:
            tmp_path = Path(tmpdir)
            with tarfile.open(archive_path, "r:gz") as archive:
                extract_archive(archive, tmp_path)

            shard_dirs = list((tmp_path / "knowledge_base").glob("shard-*"))
            if len(shard_dirs) != 1:
                raise RuntimeError(
                    f"Expected one shard directory in {archive_path}, found {len(shard_dirs)}"
                )

            merge_tree(shard_dirs[0], output_path)
            failed_urls.extend(collect_failed_urls(tmp_path))

    if failed_urls_output:
        write_failed_urls(failed_urls_output, failed_urls)


def main():
    parser = argparse.ArgumentParser(
        description="Assemble scraped shard archives into a single knowledge_base directory"
    )
    parser.add_argument("--artifacts-dir", default="artifacts", help="Input artifact directory")
    parser.add_argument(
        "--output-dir",
        default="knowledge_base",
        help="Destination directory for assembled HTML files",
    )
    parser.add_argument(
        "--failed-urls-output",
        help="Optional combined failed URL report output path",
    )
    args = parser.parse_args()

    assemble_archives(args.artifacts_dir, args.output_dir, args.failed_urls_output)


if __name__ == "__main__":
    main()
