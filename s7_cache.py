#!/usr/bin/env python3
"""S7 synthetic-description generator contract, immutable cache, and usage map.

Determinism model: a description is addressed by
``cache_key = sha256(generation_input_sha256 + NUL + generator_contract_sha256)``
— the exact input content AND the exact generation recipe. Identical chunks
share one key (and one LLM call); any change to the prompt bytes, model pin,
or validator rules re-hashes the contract and therefore every key, which is
the ONLY sanctioned way to regenerate an existing description or retry a
``raw_fallback``. Cache records are append-only and immutable.

Release rule: release builds never call an LLM. ``generate_missing`` with
``release=True`` refuses outright when eligible keys are missing;
``enforce_release`` additionally caps ``raw_fallback`` at 10% of eligible
chunk USAGES (not keys — one shared fallback key used by many chunks weighs
by its usages).
"""
import argparse
import hashlib
import json
import os
import sys

from companion import sha256_hex

REQUESTED_MODEL_ID = "claude-sonnet-4-6"
MAX_OUTPUT_TOKENS = 300
MAX_DESCRIPTION_WORDPIECES = 128
IDENTIFIER_EXTRACTOR_VERSION = "1"
VALIDATOR_VERSION = "1"
MAX_ATTEMPTS = 3
MAX_RAW_FALLBACK_USAGE_RATIO = 0.10

STATUS_OK = "ok"
STATUS_RAW_FALLBACK = "raw_fallback"

RECORD_FIELDS = (
    "cache_key", "generation_input_sha256", "generator_contract_sha256",
    "raw_content_sha256", "prompt_version", "requested_model_id",
    "returned_model_id", "identifiers", "description", "status",
)


def canonical_json(obj):
    """Canonical bytes: sorted keys, compact separators, UTF-8, unescaped."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def generator_contract(prompt_version, system_prompt, user_prompt_template,
                       requested_model_id=REQUESTED_MODEL_ID, temperature=0,
                       max_output_tokens=MAX_OUTPUT_TOKENS,
                       max_description_wordpieces=MAX_DESCRIPTION_WORDPIECES,
                       identifier_extractor_version=IDENTIFIER_EXTRACTOR_VERSION,
                       validator_version=VALIDATOR_VERSION):
    """The exact generation recipe. Every field participates in the hash.

    validator_schema spells out the validation constants as data (not just a
    version string), so loosening or tightening a rule cannot silently reuse
    descriptions validated under the old rules.
    """
    return {
        "prompt_version": prompt_version,
        "system_prompt": system_prompt,
        "user_prompt_template": user_prompt_template,
        "requested_model_id": requested_model_id,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "max_description_wordpieces": max_description_wordpieces,
        "identifier_extractor_version": identifier_extractor_version,
        "validator_version": validator_version,
        "validator_schema": {
            "sentence_range": [1, 3],
            "max_wordpieces": max_description_wordpieces,
            "forbid_fenced_code": True,
            "forbid_xml_blocks": True,
            "mandatory_semantics": [
                "topic_in_first_sentence", "root_element_when_present",
                "discriminating_type_when_present", "material_prefix",
            ],
            "introduced_identifier_policing": "case_sensitive",
        },
    }


def generator_contract_sha256(contract):
    return hashlib.sha256(canonical_json(contract)).hexdigest()


def generation_input(chunk, identifiers):
    """The six canonical generation-input fields (chunk identity excluded —
    identical content at different locations shares one description)."""
    return {
        "raw_document": chunk["content"],
        "title": chunk.get("title", ""),
        "section_heading": chunk.get("section_heading", ""),
        "breadcrumb": chunk.get("breadcrumb", ""),
        "source_type": chunk.get("source_type", ""),
        "identifiers": sorted(identifiers),
    }


def generation_input_sha256(gi):
    return hashlib.sha256(canonical_json(gi)).hexdigest()


def cache_key(gi_sha256, contract_sha256):
    return hashlib.sha256(
        f"{gi_sha256}\x00{contract_sha256}".encode("utf-8")
    ).hexdigest()


def chunk_cache_key(chunk, contract, identifiers_fn):
    """The cache key one chunk resolves to under one contract."""
    gi = generation_input(chunk, identifiers_fn(chunk))
    return cache_key(generation_input_sha256(gi), generator_contract_sha256(contract))


class S7Cache:
    """Append-only JSONL description cache.

    Existing records are immutable: appending an existing key is an error, and
    :meth:`verify_immutable` proves a newer cache file is a strict superset of
    an older one (no modified, no removed records).
    """

    def __init__(self, path):
        self.path = path
        self.records = {}

    @classmethod
    def load(cls, path):
        cache = cls(path)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    cache.records[record["cache_key"]] = record
        return cache

    def get(self, key):
        return self.records.get(key)

    def append(self, record):
        missing = [f for f in RECORD_FIELDS if f not in record]
        if missing:
            raise ValueError(f"S7 cache record missing field(s): {missing}")
        if record["status"] not in (STATUS_OK, STATUS_RAW_FALLBACK):
            raise ValueError(f"S7 cache record has unknown status: {record['status']!r}")
        key = record["cache_key"]
        if key in self.records:
            raise ValueError(
                f"S7 cache records are immutable: key {key} already exists "
                "(regenerate only under a NEW generator contract)"
            )
        self.records[key] = record
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def verify_immutable(old_path, new_path):
        """Every record of the old cache must exist byte-identically in the new."""
        old = S7Cache.load(old_path)
        new = S7Cache.load(new_path)
        for key, record in old.records.items():
            if key not in new.records:
                raise ValueError(f"S7 cache record removed: {key}")
            if new.records[key] != record:
                raise ValueError(f"S7 cache record modified: {key}")


def _dedupe_by_key(chunks, contract, identifiers_fn):
    """key -> (representative chunk, identifiers, all usage chunks), stable order."""
    by_key = {}
    for chunk in chunks:
        identifiers = sorted(identifiers_fn(chunk))
        gi = generation_input(chunk, identifiers)
        key = cache_key(
            generation_input_sha256(gi), generator_contract_sha256(contract)
        )
        entry = by_key.setdefault(key, {"chunk": chunk, "identifiers": identifiers,
                                        "usages": []})
        entry["usages"].append(chunk)
    return by_key


def generate_missing(cache, eligible_chunks, contract, llm_call, *, validator,
                     identifiers_fn, max_attempts=MAX_ATTEMPTS, release=False):
    """Generate descriptions for cache keys that do not exist yet.

    ``llm_call(system_prompt, user_prompt, requested_model_id) ->
    (returned_model_id, text)``. An attempt fails when the returned model id
    differs from the requested one or when ``validator(text, chunk,
    identifiers)`` returns errors; after ``max_attempts`` failures a
    ``raw_fallback`` record is appended (the chunk embeds its B56 input).

    ``release=True`` refuses to generate: release builds never call an LLM, so
    missing keys raise instead (the development-only fallback is disabled).
    Returns stats: {"generated", "raw_fallback", "skipped"}.
    """
    by_key = _dedupe_by_key(eligible_chunks, contract, identifiers_fn)
    missing = {k: v for k, v in by_key.items() if k not in cache.records}
    stats = {"generated": 0, "raw_fallback": 0,
             "skipped": len(by_key) - len(missing)}

    if release:
        if missing:
            raise RuntimeError(
                f"Release build with {len(missing)} missing eligible S7 cache "
                "entr(ies): release builds never call an LLM. Generate and "
                "review the cache first."
            )
        return stats

    contract_sha = generator_contract_sha256(contract)
    for key, entry in missing.items():
        chunk, identifiers = entry["chunk"], entry["identifiers"]
        gi = generation_input(chunk, identifiers)
        user_prompt = contract["user_prompt_template"].format(
            raw_document=chunk["content"]
        )
        description = None
        returned_model = None
        for _attempt in range(max_attempts):
            model_id, text = llm_call(
                contract["system_prompt"], user_prompt,
                contract["requested_model_id"],
            )
            if model_id != contract["requested_model_id"]:
                continue
            if validator(text, chunk, identifiers):
                continue
            description = text
            returned_model = model_id
            break

        record = {
            "cache_key": key,
            "generation_input_sha256": generation_input_sha256(gi),
            "generator_contract_sha256": contract_sha,
            "raw_content_sha256": sha256_hex(chunk["content"]),
            "prompt_version": contract["prompt_version"],
            "requested_model_id": contract["requested_model_id"],
            "returned_model_id": returned_model or "",
            "identifiers": identifiers,
            "description": description or "",
            "status": STATUS_OK if description else STATUS_RAW_FALLBACK,
        }
        cache.append(record)
        stats["generated" if description else "raw_fallback"] += 1
    return stats


def usage_map(contract, chunks, *, identifiers_fn):
    """Deterministic cache_key -> [[chunk_id, source_record_id, start, end]…].

    Duplicate-content chunks map to ONE key with every usage listed, so the
    release fallback cap can weigh by usages and reviewers can trace every
    chunk a description serves.
    """
    by_key = _dedupe_by_key(chunks, contract, identifiers_fn)
    return {
        key: sorted(
            [c["id"], c.get("source_record_id", ""),
             c.get("source_start_byte", 0), c.get("source_end_byte", 0)]
            for c in entry["usages"]
        )
        for key, entry in sorted(by_key.items())
    }


def enforce_release(cache, eligible_chunks, contract, *, identifiers_fn):
    """Release gates over the cache. Returns error strings (empty == pass).

    - Every eligible chunk's key must exist (missing entries fail release).
    - At most 10% of eligible chunk USAGES may resolve to raw_fallback.
    """
    errors = []
    by_key = _dedupe_by_key(eligible_chunks, contract, identifiers_fn)

    missing = sorted(k for k in by_key if k not in cache.records)
    for key in missing:
        chunk_ids = [c["id"] for c in by_key[key]["usages"]]
        errors.append(
            f"S7 cache entry missing for eligible chunk(s) {chunk_ids} (key {key})"
        )

    total_usages = sum(len(entry["usages"]) for entry in by_key.values())
    fallback_usages = sum(
        len(entry["usages"]) for key, entry in by_key.items()
        if key in cache.records
        and cache.records[key]["status"] == STATUS_RAW_FALLBACK
    )
    if total_usages and fallback_usages / total_usages > MAX_RAW_FALLBACK_USAGE_RATIO:
        errors.append(
            f"raw_fallback covers {fallback_usages}/{total_usages} eligible chunk "
            f"usages ({fallback_usages / total_usages:.1%}) — release cap is "
            f"{MAX_RAW_FALLBACK_USAGE_RATIO:.0%}"
        )
    return errors


def anthropic_llm_call(system_prompt, user_prompt, model_id):
    """Real generation call (development-time cache building only)."""
    import anthropic  # lazy: never on the release/build path

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model_id,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.model, "".join(
        block.text for block in response.content if block.type == "text"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Verify S7 cache immutability between two cache files"
    )
    parser.add_argument("--old", required=True)
    parser.add_argument("--new", required=True)
    args = parser.parse_args()
    try:
        S7Cache.verify_immutable(args.old, args.new)
    except ValueError as e:
        print(f"FAILED: {e}")
        return 1
    print("cache immutability OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
