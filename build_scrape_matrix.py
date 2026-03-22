#!/usr/bin/env python3

import argparse
import heapq
import json


def count_subtree_urls(node):
    return 1 + sum(count_subtree_urls(child) for child in node.get("children", []))


def build_matrix(urls, shard_count):
    shard_count = max(1, min(shard_count, len(urls)))
    rows = [(index, count_subtree_urls(node)) for index, node in enumerate(urls)]

    heap = [(0, shard_id, []) for shard_id in range(shard_count)]
    heapq.heapify(heap)

    for index, url_count in sorted(rows, key=lambda item: item[1], reverse=True):
        total, shard_id, indices = heapq.heappop(heap)
        indices.append(index)
        heapq.heappush(heap, (total + url_count, shard_id, indices))

    include = []
    for total, shard_id, indices in sorted(heap, key=lambda item: item[1]):
        root_indices = sorted(indices)
        include.append(
            {
                "shard": shard_id,
                "root_indices": ",".join(str(index) for index in root_indices),
                "root_count": len(root_indices),
                "url_count": total,
            }
        )

    return {"include": include}


def main():
    parser = argparse.ArgumentParser(
        description="Create a balanced GitHub Actions matrix for scrape shards"
    )
    parser.add_argument("--config", default="config.json", help="Input config.json path")
    parser.add_argument(
        "--shards",
        type=int,
        default=4,
        help="Number of scrape shards to create",
    )
    parser.add_argument(
        "--github-output",
        help="Optional path to the GITHUB_OUTPUT file for Actions step outputs",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        urls = json.load(f)["urls"]

    matrix = build_matrix(urls, args.shards)
    matrix_json = json.dumps(matrix, separators=(",", ":"))

    print(json.dumps(matrix, indent=2))

    if args.github_output:
        with open(args.github_output, "a", encoding="utf-8") as f:
            f.write(f"matrix={matrix_json}\n")
            f.write(f"shard_count={len(matrix['include'])}\n")


if __name__ == "__main__":
    main()
