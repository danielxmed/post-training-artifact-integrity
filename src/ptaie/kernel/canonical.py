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
from typing import Annotated

from pydantic import AllowInfNan

from ptaie.kernel.errors import CanonicalizationError

# The pydantic-facing alias rejects non-finite floats at validation time, so a
# wire-valid value is always canonicalizable — the two contracts must agree.
type JsonValue = (
    None
    | bool
    | int
    | Annotated[float, AllowInfNan(False)]
    | str
    | list["JsonValue"]
    | dict[str, "JsonValue"]
)


def _require_encodable(value: str, path: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        # Lone UTF-16 surrogates are legal JSON escapes (json.loads accepts
        # "\ud800") but are not encodable text; reject them with the typed
        # error instead of crashing at the encode step.
        raise CanonicalizationError(f"unpaired surrogate in string at {path}") from None


def _validate(value: object, path: str) -> None:
    if value is None or isinstance(value, bool | int):
        return
    if isinstance(value, str):
        _require_encodable(value, path)
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
            _require_encodable(key, f"{path}.<key>")
            _validate(item, f"{path}.{key}")
        return
    raise CanonicalizationError(f"non-JSON type {type(value).__name__} at {path}")


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a JSON-shaped value to canonical UTF-8 bytes.

    Rejects anything that is not plain JSON data (tuples included — callers
    convert via ``model_dump(mode="json")`` or explicit construction).
    """
    _validate(value, "$")
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except UnicodeEncodeError as exc:  # backstop: _validate should have caught it
        raise CanonicalizationError(f"unencodable string: {exc}") from exc


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_hash(value: object) -> str:
    """Canonical hash of a JSON-shaped value."""
    return sha256_hex(canonical_json_bytes(value))
