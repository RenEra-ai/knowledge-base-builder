"""Tests for the page-identity metadata added to chunk_docs.py."""

from chunk_docs import assign_chunk_indices, chunk_file, derive_page_key


# --- derive_page_key ----------------------------------------------------------

def test_derive_page_key_uses_source_url_when_present():
    assert derive_page_key(
        "https://help.boomi.com/docs/page-one", "abc_page.html"
    ) == "https://help.boomi.com/docs/page-one"


def test_derive_page_key_strips_trailing_slash():
    assert derive_page_key(
        "https://help.boomi.com/docs/page-one/", "abc_page.html"
    ) == "https://help.boomi.com/docs/page-one"


def test_derive_page_key_strips_fragment_and_query():
    assert derive_page_key(
        "https://help.boomi.com/docs/page-one#section-2", "abc_page.html"
    ) == "https://help.boomi.com/docs/page-one"
    assert derive_page_key(
        "https://help.boomi.com/docs/page-one?ver=3", "abc_page.html"
    ) == "https://help.boomi.com/docs/page-one"


def test_derive_page_key_falls_back_to_filename_when_source_url_empty():
    assert derive_page_key("", "abc1234567_my_page.html") == "file:abc1234567_my_page"


def test_derive_page_key_falls_back_when_source_url_normalizes_to_empty():
    # A bare fragment / whitespace normalizes away — must still produce a key.
    assert derive_page_key("#frag", "abc_page.html") == "file:abc_page"
    assert derive_page_key("   ", "abc_page.html") == "file:abc_page"


def test_derive_page_key_is_never_empty():
    for source_url in ("", "   ", "#x", "?q=1", "https://help.boomi.com/docs/x"):
        assert derive_page_key(source_url, "f.html")


# --- chunk_file: one HTML file -> multiple page_keys --------------------------

MULTI_URL_HTML = """
<h1>Page One Title</h1>
<p><strong>Path:</strong> <a href="https://help.boomi.com/docs/page-one">Page One</a></p>
<h2>Section A</h2>
<p>Section A explains the first concept in enough detail to stand alone.</p>
<h2>Section B</h2>
<p>Section B covers the second concept in the same scraped HTML file.</p>
<p><strong>Path:</strong> <a href="https://help.boomi.com/docs/page-two">Page Two</a></p>
<h2>Section C</h2>
<p>Section C belongs to a different documentation page than the sections above.</p>
"""


def test_chunk_file_splits_one_file_into_multiple_page_keys(tmp_path):
    f = tmp_path / "abc1234567_multi.html"
    f.write_text(MULTI_URL_HTML, encoding="utf-8")

    # min_tokens=0 disables merging, large max_tokens disables splitting, so each
    # section maps to exactly one chunk and page identity is easy to assert.
    chunks = chunk_file(str(f), f.name, min_tokens=0, max_tokens=100_000, verbose=False)

    page_keys = {c["page_key"] for c in chunks}
    assert page_keys == {
        "https://help.boomi.com/docs/page-one",
        "https://help.boomi.com/docs/page-two",
    }


def test_chunk_file_chunks_have_page_key_but_not_chunk_index(tmp_path):
    # chunk_index is assigned globally in main(), not inside chunk_file.
    f = tmp_path / "abc1234567_multi.html"
    f.write_text(MULTI_URL_HTML, encoding="utf-8")

    chunks = chunk_file(str(f), f.name, min_tokens=0, max_tokens=100_000, verbose=False)

    assert chunks
    for c in chunks:
        assert c["page_key"]
        assert "chunk_index" not in c


# --- assign_chunk_indices -----------------------------------------------------

def test_assign_chunk_indices_is_zero_based_and_page_local():
    chunks = [
        {"id": "a", "page_key": "P1"},
        {"id": "b", "page_key": "P1"},
        {"id": "c", "page_key": "P2"},
        {"id": "d", "page_key": "P1"},
        {"id": "e", "page_key": "P2"},
    ]
    assign_chunk_indices(chunks)
    assert [c["chunk_index"] for c in chunks] == [0, 1, 0, 2, 1]


def test_assign_chunk_indices_contiguous_per_page_key():
    chunks = [{"id": str(i), "page_key": "P1" if i % 2 == 0 else "P2"} for i in range(6)]
    assign_chunk_indices(chunks)
    p1 = [c["chunk_index"] for c in chunks if c["page_key"] == "P1"]
    p2 = [c["chunk_index"] for c in chunks if c["page_key"] == "P2"]
    assert p1 == [0, 1, 2]
    assert p2 == [0, 1, 2]


def test_assign_chunk_indices_preserves_document_order():
    chunks = [{"id": str(i), "page_key": "P"} for i in range(5)]
    assign_chunk_indices(chunks)
    assert [c["chunk_index"] for c in chunks] == [0, 1, 2, 3, 4]


def test_assign_chunk_indices_empty_list():
    assign_chunk_indices([])  # must not raise


# --- empty-content filtering -------------------------------------------------

CATEGORY_LANDING_HTML = """
<h1>Integration</h1>
<p><strong>Path:</strong> <a href="https://help.boomi.com/docs/integration">Integration</a></p>
<h2>Process Building</h2>
<p><strong>Path:</strong> <a href="https://help.boomi.com/docs/integration/process-building">Process Building</a></p>
<p>Process Building covers how to design integration processes from end to end with enough detail to be useful as a standalone reference.</p>
<h2>Connectors</h2>
<p><strong>Path:</strong> <a href="https://help.boomi.com/docs/integration/connectors">Connectors</a></p>
<p>Connectors describe how Boomi communicates with external systems and what configuration each connector type expects from the integrator.</p>
"""


def test_chunk_file_drops_empty_category_landing_chunk(tmp_path):
    f = tmp_path / "integration_landing.html"
    f.write_text(CATEGORY_LANDING_HTML, encoding="utf-8")

    chunks = chunk_file(str(f), f.name, min_tokens=0, max_tokens=100_000, verbose=False)

    # No chunk may have empty/blank content — the build_index.py validator
    # rejects those and that's what broke the deploy.
    for c in chunks:
        assert c["content"].strip(), f"empty content in chunk {c['id']}: {c!r}"

    # Child pages survive with their own page_key and real content.
    page_keys = {c["page_key"] for c in chunks}
    assert "https://help.boomi.com/docs/integration/process-building" in page_keys
    assert "https://help.boomi.com/docs/integration/connectors" in page_keys

    # The empty landing page (its own page_key) must NOT have been merged into
    # a different page — it should simply be dropped.
    landing_chunks = [
        c for c in chunks
        if c["page_key"] == "https://help.boomi.com/docs/integration"
    ]
    assert landing_chunks == []

    # Titles should reflect each child page, not the parent category.
    titles_by_page = {c["page_key"]: c["title"] for c in chunks}
    assert titles_by_page["https://help.boomi.com/docs/integration/process-building"] == "Process Building"
    assert titles_by_page["https://help.boomi.com/docs/integration/connectors"] == "Connectors"


HEADING_ONLY_HTML = """
<h1>Page Title</h1>
<p><strong>Path:</strong> <a href="https://help.boomi.com/docs/lonely">Page Title</a></p>
<h2>Standalone Heading</h2>
<h2>Real Section</h2>
<p>This section has a meaningful body that should be retained in the chunked output for the test.</p>
"""


def test_chunk_file_returns_only_non_blank_chunks(tmp_path):
    for name, html in (
        ("multi.html", MULTI_URL_HTML),
        ("landing.html", CATEGORY_LANDING_HTML),
        ("heading_only.html", HEADING_ONLY_HTML),
    ):
        f = tmp_path / name
        f.write_text(html, encoding="utf-8")
        chunks = chunk_file(str(f), f.name, min_tokens=0, max_tokens=100_000, verbose=False)
        for c in chunks:
            assert c["content"].strip(), (
                f"{name}: blank content in chunk {c['id']}: {c!r}"
            )
