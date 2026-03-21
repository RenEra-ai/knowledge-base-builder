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

## Configuration

The `config.json` file contains the hierarchical URL tree to scrape:

```json
{
    "urls": [
        {
            "url": "https://help.boomi.com/docs/...",
            "children": [
                {
                    "url": "https://help.boomi.com/docs/...",
                    "children": []
                }
            ]
        }
    ]
}
```

## Running the Script

```bash
python main.py
```

Output HTML files are written to the `knowledge_base/` directory. Failed URLs (404s, timeouts) are logged to `knowledge_base/_failed_urls.txt`.
