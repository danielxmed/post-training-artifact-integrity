import math

import pytest

from ptaie.kernel.canonical import canonical_json_bytes, content_hash, sha256_hex
from ptaie.kernel.errors import CanonicalizationError


def test_key_order_does_not_matter() -> None:
    assert content_hash({"a": 1, "b": [2, 3]}) == content_hash({"b": [2, 3], "a": 1})


def test_compact_separators_and_sorted_keys() -> None:
    assert canonical_json_bytes({"b": 1, "a": [1, 2]}) == b'{"a":[1,2],"b":1}'


def test_floats_use_shortest_roundtrip_repr() -> None:
    assert canonical_json_bytes(0.1) == b"0.1"
    assert canonical_json_bytes([1e300]) == b"[1e+300]"


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_floats_rejected(value: float) -> None:
    with pytest.raises(CanonicalizationError):
        canonical_json_bytes({"x": value})


def test_non_string_keys_rejected() -> None:
    with pytest.raises(CanonicalizationError):
        canonical_json_bytes({1: "a"})


@pytest.mark.parametrize("value", [(1, 2), {1, 2}, object(), b"bytes"])
def test_non_json_types_rejected(value: object) -> None:
    with pytest.raises(CanonicalizationError):
        canonical_json_bytes(value)


def test_no_unicode_normalization() -> None:
    composed = "é"  # é as a single code point
    decomposed = "é"  # e + combining acute
    assert content_hash(composed) != content_hash(decomposed)


def test_utf8_not_ascii_escaped() -> None:
    assert canonical_json_bytes("café") == b'"caf\xc3\xa9"'


def test_sha256_hex_known_vector() -> None:
    assert sha256_hex(b"") == ("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
