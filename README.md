# Knowledge Base Builder

This project scrapes Boomi documentation from `help.boomi.com` and `developer.boomi.com`, chunks the content into semantically meaningful pieces, and builds a ChromaDB vector database for semantic search.

## Prerequisites

- Python 3.x
- `pip` (Python package installer)

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/RenEra-ai/knowledge-base-builder.git
    cd knowledge-base-builder
    ```

2. **Create a virtual environment and activate it**:

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3. **Install the dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Generate the URL tree

`build_url_tree.py` crawls the Boomi documentation sidebar to produce an up-to-date `config.json`:

```bash
python build_url_tree.py
```

Options:
- `--output FILE` — output path (default: `config.json`)
- `--sections LIST` — comma-separated sections to crawl (default: all sections)
- `--validate` — spot-check 20 random URLs after generation
- `--delay SECS` — delay between requests (default: `1.0`)
- `--verbose` — print each URL as it is discovered

The old `config.json` is automatically backed up to `config.json.bak`.

### 2. Scrape the documentation

```bash
python main.py
```

Output HTML files are written to the `knowledge_base/` directory. Failed URLs (404s, timeouts) are logged to `knowledge_base/_failed_urls.txt`.

Options:
- `--config FILE` — input config file to scrape (default: `config.json`)
- `--output-dir DIR` — output directory for generated HTML files (default: `knowledge_base/`)
- `--delay SECS` — delay between requests (default: `1.0`)
- `--root-indices LIST` — scrape only the selected top-level root indices (used by GitHub Actions sharding)
- `--failed-urls-file FILE` — write the failed URL report to a specific path

Generated HTML filenames are prefixed with a short SHA-1 hash of the source URL so shard assembly is deterministic and collisions across similarly titled pages are avoided.

### 2b. Fetch the Companion supplemental corpus (optional)

`fetch_companion.py` downloads a small, curated set of Markdown reference docs from a **pinned commit** of a public BSD-2 GitHub repo (the [Boomi Companion](https://github.com/OfficialBoomi/boomi-integration) skill) into a staging directory, alongside a `companion_manifest.json` recording provenance + SHA-256 for each file:

```bash
python fetch_companion.py
```

Only the paths listed in `companion_sources.json` are fetched (non-recursive), and the run **fails closed** if the commit is not a full 40-char sha, a URL template hardcodes a moving ref, a path is disallowed, or any configured source is missing/empty. This content is *supplemental implementation context*, not official Boomi documentation — every chunk it produces is labelled `source_type="companion_reference"` / `verification_status="companion_unverified"`.

Options:
- `--config FILE` — allowlist config (default: `companion_sources.json`)
- `--staging DIR` — staging directory for fetched Markdown (default: `companion_sources/`)
- `--manifest FILE` — output manifest path (default: `companion_manifest.json`)

### 3. Chunk the documentation

`chunk_docs.py` splits HTML files into semantically meaningful chunks on heading boundaries, outputting structured JSONL with metadata:

```bash
python chunk_docs.py --verbose
```

Options:
- `--input DIR` — input directory of HTML files (default: `knowledge_base/`)
- `--output FILE` — output JSONL file (default: `chunks.jsonl`)
- `--min-tokens N` — minimum chunk size in tokens (default: `100`)
- `--max-tokens N` — maximum chunk size in tokens (default: `1200`)
- `--companion-input DIR` — staged companion Markdown directory (default: `companion_sources/`)
- `--companion-manifest FILE` — `companion_manifest.json` from step 2b; **enables** the supplemental corpus when provided (Markdown chunks are appended with stable `companion://…` page keys and provenance metadata)
- `--companion-config FILE` — `companion_sources.json` to check the manifest against (default: `companion_sources.json`; pass `''` to skip)
- `--verbose` — print each chunk as it is created

The companion path is **fail-closed on stale inputs**, because every companion chunk is stamped with the pinned upstream commit and `raw_url` — anything chunked must be exactly what was fetched from that commit, or the chunk asserts a provenance it does not have. The build aborts if:

- the manifest's curation policy has drifted from `companion_sources.json` (you edited the allowlist, a title, or the pinned commit and did not re-run `fetch_companion.py`) — otherwise the edit silently no-ops and the corpus is rebuilt under the old rules;
- a staged file's content does not hash to the SHA-256 the manifest recorded for it (a hand-edit, or a truncated file from an interrupted fetch).

Editing `companion_sources.json` therefore always means re-running `fetch_companion.py` before `chunk_docs.py`.

### 4. Build the semantic index

`build_index.py` embeds chunks and builds a ChromaDB vector database for semantic search:

```bash
python build_index.py --verify
```

Options:
- `--input FILE` — input JSONL chunks file (default: `chunks.jsonl`)
- `--output DIR` — output ChromaDB directory (default: `boomi_knowledge_db/`)
- `--model NAME` — sentence transformer model (default: `all-MiniLM-L6-v2`)
- `--batch-size N` — batch size for indexing (default: `200`)
- `--verify` — run test queries after building
- `--verbose` — print progress details

### Full workflow

```bash
python build_url_tree.py --validate && python main.py && python fetch_companion.py && python chunk_docs.py --verbose --companion-manifest companion_manifest.json && python build_index.py --verify
```

GitHub Actions uses the same flow, but splits the scrape into four balanced shards and runs at most two of them in parallel to stay within workflow time limits and reduce load on Boomi.

## Output

The final artifact is `boomi_knowledge_db/` — a portable ChromaDB directory for downstream consumption (MCP server, RAG pipeline, etc.).
