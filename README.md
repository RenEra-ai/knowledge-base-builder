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
- `--verbose` — print each chunk as it is created

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
python build_url_tree.py --validate && python main.py && python chunk_docs.py --verbose && python build_index.py --verify
```

## Output

The final artifact is `boomi_knowledge_db/` — a portable ChromaDB directory for downstream consumption (MCP server, RAG pipeline, etc.).
