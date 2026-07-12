"""Canonical JSON serialization and hashing — the kernel's only hashing code.

Rules:

- keys sorted, compact separators, UTF-8, ``ensure_ascii=False``;
- floats serialize via CPython ``repr`` (shortest round-trip); non-finite
  floats are rejected;
- strings pass through byte-for-byte — deliberately **no** Unicode
  normalization (mojibake fixtures carry non-NFC sequences on purpose);
- artifact file blobs are hashed as raw bytes elsewhere; this module never
  re-serializes agent-visible file content.

Never hash pydantic objects directly: always
``content_hash(model.model_dump(mode="json"))`` so the hash contract belongs
to this module, not the library.
"""

import hashlib
import json
import math

from ptaie.kernel.errors import CanonicalizationError

type JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


def _validate(value: object, path: str) -> None:
    if value is None or isinstance(value, bool | int | str):
        return
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise CanonicalizationError(f"non-finite float at {path}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError(f"non-string key {key!r} at {path}")
            _validate(item, f"{path}.{key}")
        return
    raise CanonicalizationError(f"non-JSON type {type(value).__name__} at {path}")


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a JSON-shaped value to canonical UTF-8 bytes.

    Rejects anything that is not plain JSON data (tuples included — callers
    convert via ``model_dump(mode="json")`` or explicit construction).
    """
    _validate(value, "$")
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_hash(value: object) -> str:
    """Canonical hash of a JSON-shaped value."""
    return sha256_hex(canonical_json_bytes(value))
