"""Tests for the WordPiece tokenizer seam."""
import pytest

from kb_tokenizer import (
    EFFECTIVE_BUDGET,
    MAX_SEQ_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    FakeWordPieceTokenizer,
    HFWordPieceTokenizer,
)


def test_constants_pin_the_embedding_contract():
    assert MODEL_ID == "all-MiniLM-L6-v2"
    assert MODEL_REVISION == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    assert MAX_SEQ_TOKENS == 256
    assert EFFECTIVE_BUDGET == 254


def test_fake_tokenizer_counts_ceil_len_over_four_per_token():
    tok = FakeWordPieceTokenizer()
    assert tok.count("") == 0
    assert tok.count("   \n\t ") == 0
    assert tok.count("ab") == 1              # ceil(2/4)
    assert tok.count("abcd") == 1            # ceil(4/4)
    assert tok.count("abcde") == 2           # ceil(5/4)
    assert tok.count("DocumentPropertySet") == 5   # ceil(19/4)
    # Additive across whitespace-separated tokens; whitespace shape is free.
    assert tok.count("ab abcd") == 2
    assert tok.count("ab\n\nabcd") == 2
    assert tok.count("a b c") == 3


def test_fake_tokenizer_is_deterministic():
    tok = FakeWordPieceTokenizer()
    text = "Map functions type=\"DocumentPropertySet\" dynamicdocument"
    assert tok.count(text) == tok.count(text)


def test_hf_tokenizer_counts_without_special_tokens():
    class _StubHF:
        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            return text.split()

    tok = HFWordPieceTokenizer(tokenizer=_StubHF())
    assert tok.count("map function skeleton") == 3


def test_real_hf_tokenizer_matches_known_counts():
    transformers = pytest.importorskip("transformers")  # noqa: F841
    tok = HFWordPieceTokenizer()
    # "hello world" is exactly two WordPieces for the MiniLM vocab.
    assert tok.count("hello world") == 2
    assert tok.count("") == 0
