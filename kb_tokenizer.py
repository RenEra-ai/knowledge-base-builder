#!/usr/bin/env python3
"""WordPiece tokenizer seam for the S5/S6/S7 pipeline.

Every splitting/header/eligibility decision counts REAL WordPieces against the
pinned MiniLM tokenizer — never the legacy ``len(text) // 4`` estimate, which
is retained elsewhere for diagnostics only. Consumers take a ``tokenizer``
argument exposing ``count(text) -> int`` so unit tests can substitute the
deterministic :class:`FakeWordPieceTokenizer` while release builds use
:class:`HFWordPieceTokenizer` (lazy ``transformers`` import).
"""
import math

MODEL_ID = "all-MiniLM-L6-v2"
# Pinned model revision (sentence-transformers/all-MiniLM-L6-v2) — the same
# revision the serving side's embedding contract declares.
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
# The model's hard sequence window, and the share of it available to text once
# the [CLS]/[SEP] specials the encoder always adds are reserved.
MAX_SEQ_TOKENS = 256
SPECIAL_TOKENS = 2
EFFECTIVE_BUDGET = MAX_SEQ_TOKENS - SPECIAL_TOKENS


class FakeWordPieceTokenizer:
    """Deterministic test double.

    Every whitespace-separated token costs ``ceil(len(token) / 4)`` pieces, so
    long identifiers cost more than short prose words — the same pressure
    WordPiece applies — while counts stay hand-computable in tests. Counting is
    additive across whitespace-separated tokens; whitespace itself is free.
    """

    def count(self, text):
        return sum(math.ceil(len(token) / 4) for token in text.split())


class HFWordPieceTokenizer:
    """Real WordPiece counts from the pinned MiniLM tokenizer.

    Counts exclude special tokens (``add_special_tokens=False``) — budgets
    account for the [CLS]/[SEP] pair once via ``EFFECTIVE_BUDGET``.
    """

    def __init__(self, tokenizer=None):
        if tokenizer is None:
            # Lazy: unit tests use the fake; only real builds pay this import.
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                f"sentence-transformers/{MODEL_ID}", revision=MODEL_REVISION
            )
        self._tokenizer = tokenizer

    def count(self, text):
        return len(self._tokenizer.encode(text, add_special_tokens=False))


def load_tokenizer():
    """Load the pinned real tokenizer (network/cache access on first use)."""
    return HFWordPieceTokenizer()
