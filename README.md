# Knowledge Base Builder

This project scrapes Boomi documentation from `help.boomi.com` and converts the content into cleaned HTML files, organizing them into a hierarchical knowledge base.

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
- `--sections LIST` — comma-separated sections to crawl (default: `Integration`)
- `--validate` — spot-check 20 random URLs after generation
- `--delay SECS` — delay between requests (default: `1.0`)
- `--verbose` — print each URL as it is discovered

The old `config.json` is automatically backed up to `config.json.bak`.

### 2. Scrape the documentation

```bash
python main.py
```

Output HTML files are written to the `knowledge_base/` directory. Failed URLs (404s, timeouts) are logged to `knowledge_base/_failed_urls.txt`.

### Full workflow

```bash
python build_url_tree.py --validate && python main.py
```
