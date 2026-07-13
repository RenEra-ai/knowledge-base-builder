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

## kb-25 candidate pipeline (S5/S6/S7)

kb-25 replaces the `len(text)//4` token estimate with real WordPiece counting
against the pinned embedding model (`all-MiniLM-L6-v2` @
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, sentence-transformers 5.6.0,
chromadb 1.5.9). Four candidates are built **from one frozen source snapshot**
and compared through the pinned serving stack:

| candidate | chunking | embedding text | manifest |
|---|---|---|---|
| `c0` | legacy splitter | raw document | schema v1 |
| `b5` | S5 tokenizer-aware | raw document | schema v1 |
| `b56` | S5 + S6 header | deduplicated header + raw document | schema v2 |
| `b567` | S5 + S6 + S7 | eval-gated synthetic descriptions for eligible Companion chunks (everything else embeds the B56 input) | schema v2 |

Stages and tools:

1. `snapshot.py freeze` — copy scrape + verified companion inputs into an
   immutable snapshot recording `source_snapshot_sha256`; every candidate
   chunker takes `--snapshot <dir>` and reads inputs ONLY from a snapshot that
   verifies.
2. `s5_pipeline.py` — tokenizer-aware splitting (S5) with exact source-byte
   spans, plus the deduplicated contextual header (S6). Every final embedding
   input is asserted ≤ 256 WordPieces — model-side truncation is forbidden.
3. `s7_cache.py` / `s7_eligibility.py` / `s7_validate.py` — the S7 synthetic
   description machinery: Companion-only eligibility (≥ 0.3 code-WordPiece
   ratio, ≥ 2 structural identifiers), validated 1–3-sentence descriptions,
   an immutable generation cache keyed by content + generator contract, and
   release gates (missing entries fail; ≤ 10% raw_fallback usages). Release
   builds never call an LLM.
4. `build_index.py --candidate <c>` — explicit pinned-revision embeddings,
   raw documents stored in Chroma, schema-v2 manifests with the embedding
   contract, a pre-embedding fixture-marker integrity gate, and the
   reproducibility `semantic_inputs.jsonl` sidecar.
5. `eval_harness.py run` — every fixture query through the pinned Track A
   `KbService.search(top_k=5)` (set `BOOMI_MCP_SERVER_PATH` to the
   boomi-mcp-server checkout); `eval_harness.py compare` hard-gates two runs
   (ids/documents/metadata/contracts/cache/semantic inputs/ranks/recall/
   status; 0.0001 distance tolerance).
6. `eval_gates.py` / `eval_holdout.py` — the B56 fallback gates, B567 publish
   gates, identifier-regression rule, and sealed-holdout evaluation
   (`KB_RELEASE_HOLDOUT_B64`; only its SHA-256 + count are logged before
   evaluation).

**Release decision:** publish B567 only when it passes calibration AND the
sealed holdout. Otherwise publish B56 only when its separate fallback gates
pass — a B56 release fixes truncation and context fidelity but does **not**
claim the original generic-retrieval defect (G01) is solved. kb-24 remains the
rollback artifact.

**Reproducibility:** release builds and release comparisons run ONLY under the
pinned CI environment (requirements.txt pins, same CPU runner class, CPU-only,
single Torch/OMP/MKL thread — `eval_harness.pin_determinism`). Locally built
candidates are never release-grade. Raw float-vector hashes, Chroma database
hashes, and release tarball hashes are diagnostic-only, never gating.

Explicitly out of kb-25's scope: BM25, per-source result quotas, automatic
server-side query rewriting/RRF, dual-vector indexes, and S7 on official
documentation chunks.

## Output

The final artifact is `boomi_knowledge_db/` — a portable ChromaDB directory for downstream consumption (MCP server, RAG pipeline, etc.).
