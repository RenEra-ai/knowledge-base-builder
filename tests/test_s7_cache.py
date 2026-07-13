"""Tests for the S7 generator contract, immutable cache, and usage map."""
import hashlib
import json

import pytest

from s7_cache import (
    S7Cache,
    cache_key,
    canonical_json,
    enforce_release,
    generate_missing,
    generation_input,
    generation_input_sha256,
    generator_contract,
    generator_contract_sha256,
    usage_map,
)


def _contract(**overrides):
    kwargs = {
        "prompt_version": "v1",
        "system_prompt": "You describe Boomi component structures.",
        "user_prompt_template": "Describe:\n{raw_document}",
    }
    kwargs.update(overrides)
    return generator_contract(**kwargs)


def _chunk(chunk_id="c1", content='<FunctionStep type="DocumentPropertySet"/>',
           **overrides):
    chunk = {
        "id": chunk_id,
        "content": content,
        "title": "Map Component Functions",
        "section_heading": "Document property",
        "breadcrumb": "Companion Reference > Components > Map Component Functions",
        "source_type": "companion_reference",
        "source_record_id": "companion://repo/map_component_functions.md",
        "source_start_byte": 100,
        "source_end_byte": 400,
    }
    chunk.update(overrides)
    return chunk


def _identifiers(_chunk_dict):
    return ["DocumentPropertySet", "FunctionStep"]


def _ok_llm(system, user, model_id):
    return model_id, "A Set Document Property function uses FunctionStep."


def _no_errors(description, chunk, identifiers):
    return []


# --- canonical JSON + hashing ---------------------------------------------------

def test_canonical_json_is_byte_stable_and_unicode_unescaped():
    a = canonical_json({"b": 1, "a": "ü"})
    b = canonical_json({"a": "ü", "b": 1})
    assert a == b == b'{"a":"\xc3\xbc","b":1}'


def test_generator_contract_carries_every_canonical_field():
    contract = _contract()
    assert contract["prompt_version"] == "v1"
    assert contract["system_prompt"].startswith("You describe")
    assert contract["user_prompt_template"].startswith("Describe:")
    assert contract["requested_model_id"] == "claude-sonnet-4-6"
    assert contract["temperature"] == 0
    assert contract["max_output_tokens"] > 0
    assert contract["max_description_wordpieces"] == 128
    assert contract["identifier_extractor_version"]
    assert contract["validator_version"]
    # validator_schema is canonical data, so a rule change re-hashes the contract.
    assert contract["validator_schema"]["sentence_range"] == [1, 3]


@pytest.mark.parametrize("field,value", [
    ("prompt_version", "v2"),
    ("system_prompt", "different"),
    ("user_prompt_template", "different {raw_document}"),
])
def test_contract_hash_changes_with_any_canonical_field(field, value):
    base = generator_contract_sha256(_contract())
    changed = generator_contract_sha256(_contract(**{field: value}))
    assert base != changed


def test_contract_hash_changes_with_validator_schema_rule_change():
    contract = _contract()
    base = generator_contract_sha256(contract)
    contract["validator_schema"]["max_wordpieces"] = 64
    assert generator_contract_sha256(contract) != base


def test_cache_key_formula_is_sha256_of_hashes_joined_by_nul():
    gi_sha = generation_input_sha256(generation_input(_chunk(), _identifiers(None)))
    gc_sha = generator_contract_sha256(_contract())
    expected = hashlib.sha256(f"{gi_sha}\x00{gc_sha}".encode()).hexdigest()
    assert cache_key(gi_sha, gc_sha) == expected


def test_generation_input_uses_the_six_canonical_fields_sorted_identifiers():
    gi = generation_input(_chunk(), ["FunctionStep", "DocumentPropertySet"])
    assert sorted(gi) == [
        "breadcrumb", "identifiers", "raw_document", "section_heading",
        "source_type", "title",
    ]
    assert gi["identifiers"] == ["DocumentPropertySet", "FunctionStep"]
    assert gi["raw_document"] == _chunk()["content"]


def test_identical_content_yields_identical_generation_input_hash():
    a = generation_input_sha256(generation_input(_chunk("c1"), _identifiers(None)))
    b = generation_input_sha256(generation_input(_chunk("c2"), _identifiers(None)))
    assert a == b  # chunk id / location are NOT generation inputs


# --- cache immutability -----------------------------------------------------------

def _record(key, status="ok", description="desc"):
    return {
        "cache_key": key,
        "generation_input_sha256": "a" * 64,
        "generator_contract_sha256": "b" * 64,
        "raw_content_sha256": "c" * 64,
        "prompt_version": "v1",
        "requested_model_id": "claude-sonnet-4-6",
        "returned_model_id": "claude-sonnet-4-6",
        "identifiers": ["FunctionStep"],
        "description": description,
        "status": status,
    }


def test_cache_appends_and_round_trips(tmp_path):
    path = str(tmp_path / "s7_cache.jsonl")
    cache = S7Cache(path)
    cache.append(_record("k1"))
    cache.append(_record("k2", status="raw_fallback", description=""))
    reloaded = S7Cache.load(path)
    assert set(reloaded.records) == {"k1", "k2"}
    assert reloaded.get("k2")["status"] == "raw_fallback"


def test_cache_rejects_duplicate_key_append(tmp_path):
    cache = S7Cache(str(tmp_path / "s7_cache.jsonl"))
    cache.append(_record("k1"))
    with pytest.raises(ValueError, match="immutable"):
        cache.append(_record("k1", description="changed"))


def test_cache_rejects_record_missing_required_fields(tmp_path):
    cache = S7Cache(str(tmp_path / "s7_cache.jsonl"))
    bad = _record("k1")
    del bad["raw_content_sha256"]
    with pytest.raises(ValueError, match="raw_content_sha256"):
        cache.append(bad)


def test_verify_immutable_catches_modified_and_removed_records(tmp_path):
    old_path = str(tmp_path / "old.jsonl")
    old = S7Cache(old_path)
    old.append(_record("k1"))
    old.append(_record("k2"))

    modified_path = str(tmp_path / "modified.jsonl")
    modified = S7Cache(modified_path)
    modified.append(_record("k1", description="rewritten"))
    modified.append(_record("k2"))
    with pytest.raises(ValueError, match="k1"):
        S7Cache.verify_immutable(old_path, modified_path)

    removed_path = str(tmp_path / "removed.jsonl")
    removed = S7Cache(removed_path)
    removed.append(_record("k2"))
    with pytest.raises(ValueError, match="k1"):
        S7Cache.verify_immutable(old_path, removed_path)

    extended_path = str(tmp_path / "extended.jsonl")
    extended = S7Cache(extended_path)
    extended.append(_record("k1"))
    extended.append(_record("k2"))
    extended.append(_record("k3"))
    S7Cache.verify_immutable(old_path, extended_path)  # additive is legal


# --- generation --------------------------------------------------------------------

def _generate(cache, chunks, llm, validator=_no_errors, max_attempts=3,
              release=False):
    return generate_missing(
        cache, chunks, _contract(), llm,
        validator=validator, identifiers_fn=_identifiers,
        max_attempts=max_attempts, release=release,
    )


def test_generate_only_missing_keys(tmp_path):
    cache = S7Cache(str(tmp_path / "cache.jsonl"))
    calls = []

    def llm(system, user, model_id):
        calls.append(user)
        return model_id, "A Set Document Property description."

    stats = _generate(cache, [_chunk("c1")], llm)
    assert stats["generated"] == 1 and len(calls) == 1
    # Second run: the key exists — nothing is generated.
    stats = _generate(cache, [_chunk("c1")], llm)
    assert stats["generated"] == 0 and len(calls) == 1


def test_duplicate_content_chunks_share_one_key_and_one_llm_call(tmp_path):
    cache = S7Cache(str(tmp_path / "cache.jsonl"))
    calls = []

    def llm(system, user, model_id):
        calls.append(user)
        return model_id, "One description."

    chunks = [_chunk("c1"), _chunk("c2")]  # identical generation input
    stats = _generate(cache, chunks, llm)
    assert stats["generated"] == 1
    assert len(calls) == 1
    assert len(cache.records) == 1


def test_wrong_returned_model_burns_an_attempt_then_falls_back(tmp_path):
    cache = S7Cache(str(tmp_path / "cache.jsonl"))
    calls = []

    def llm(system, user, model_id):
        calls.append(1)
        return "claude-sonnet-4-5", "text from the wrong model"

    stats = _generate(cache, [_chunk()], llm)
    assert len(calls) == 3
    assert stats["raw_fallback"] == 1
    record = next(iter(cache.records.values()))
    assert record["status"] == "raw_fallback"
    assert record["description"] == ""


def test_validator_rejection_burns_attempts_then_falls_back(tmp_path):
    cache = S7Cache(str(tmp_path / "cache.jsonl"))

    def rejecting_validator(description, chunk, identifiers):
        return ["mandatory topic missing"]

    stats = _generate(cache, [_chunk()], _ok_llm, validator=rejecting_validator)
    assert stats["raw_fallback"] == 1


def test_retry_succeeding_within_attempts_records_ok(tmp_path):
    cache = S7Cache(str(tmp_path / "cache.jsonl"))
    state = {"n": 0}

    def flaky_llm(system, user, model_id):
        state["n"] += 1
        if state["n"] < 3:
            return "claude-other", "wrong model"
        return model_id, "Good description."

    stats = _generate(cache, [_chunk()], flaky_llm)
    assert stats["generated"] == 1 and stats["raw_fallback"] == 0
    record = next(iter(cache.records.values()))
    assert record["status"] == "ok"
    assert record["returned_model_id"] == "claude-sonnet-4-6"


def test_fallback_not_retried_under_same_contract_but_new_contract_retries(tmp_path):
    cache = S7Cache(str(tmp_path / "cache.jsonl"))

    def failing_llm(system, user, model_id):
        return "claude-other", "always wrong"

    _generate(cache, [_chunk()], failing_llm)
    assert len(cache.records) == 1  # raw_fallback under contract v1

    calls = []

    def good_llm(system, user, model_id):
        calls.append(1)
        return model_id, "Recovered description."

    # Same contract: the fallback record satisfies the key — no retry.
    _generate(cache, [_chunk()], good_llm)
    assert calls == []

    # New contract (new prompt_version) -> new cache key -> fresh attempt.
    stats = generate_missing(
        cache, [_chunk()], _contract(prompt_version="v2"), good_llm,
        validator=_no_errors, identifiers_fn=_identifiers,
    )
    assert stats["generated"] == 1
    assert calls == [1]
    assert len(cache.records) == 2


def test_release_mode_never_calls_llm_and_reports_missing(tmp_path):
    cache = S7Cache(str(tmp_path / "cache.jsonl"))
    calls = []

    def llm(system, user, model_id):
        calls.append(1)
        return model_id, "text"

    with pytest.raises(RuntimeError, match="[Rr]elease"):
        _generate(cache, [_chunk()], llm, release=True)
    assert calls == []


# --- usage map + release enforcement -----------------------------------------------

def test_usage_map_lists_every_chunk_usage_per_key(tmp_path):
    cache = S7Cache(str(tmp_path / "cache.jsonl"))
    chunks = [
        _chunk("c1"),
        _chunk("c2", source_start_byte=500, source_end_byte=900),
    ]
    _generate(cache, chunks, _ok_llm)
    mapping = usage_map(_contract(), chunks, identifiers_fn=_identifiers)
    assert len(mapping) == 1
    (key, usages), = mapping.items()
    assert key in cache.records
    assert usages == [
        ["c1", "companion://repo/map_component_functions.md", 100, 400],
        ["c2", "companion://repo/map_component_functions.md", 500, 900],
    ]


def test_enforce_release_fails_on_missing_eligible_entry(tmp_path):
    cache = S7Cache(str(tmp_path / "cache.jsonl"))
    errors = enforce_release(cache, [_chunk()], _contract(),
                             identifiers_fn=_identifiers)
    assert errors and "missing" in errors[0]


def test_enforce_release_fallback_ratio_boundary(tmp_path):
    """<= 10% of eligible chunk USAGES may resolve to raw_fallback: 2 of 20
    usages (10.0%) passes, 3 of 20 fails. Denominator counts usages, not keys."""
    def _build(n_fallback_usages):
        cache = S7Cache(str(tmp_path / f"cache_{n_fallback_usages}.jsonl"))
        chunks = []
        # One fallback KEY shared by n_fallback_usages chunks (duplicate content).
        for i in range(n_fallback_usages):
            chunks.append(_chunk(f"fb{i}", content="<Bad thing='fails'/>"))
        for i in range(20 - n_fallback_usages):
            chunks.append(_chunk(f"ok{i}", content=f"<Good n='{i}'/>"))

        def llm(system, user, model_id):
            if "Bad" in user:
                return "claude-other", "wrong"
            return model_id, f"Description {user[-12:]}"

        generate_missing(cache, chunks, _contract(), llm, validator=_no_errors,
                         identifiers_fn=_identifiers)
        return enforce_release(cache, chunks, _contract(),
                               identifiers_fn=_identifiers)

    assert _build(2) == []          # exactly 10.0% — passes
    errors = _build(3)              # 15% — fails
    assert errors and "raw_fallback" in errors[0]
