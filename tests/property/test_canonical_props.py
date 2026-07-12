import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ptaie.kernel.canonical import canonical_json_bytes, content_hash

# JSON-shaped values; floats restricted to finite (non-finite is rejected by design).
_json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(min_value=-(2**53), max_value=2**53)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(max_size=40),
    lambda children: (
        st.lists(children, max_size=5) | st.dictionaries(st.text(max_size=20), children, max_size=5)
    ),
    max_leaves=25,
)


@pytest.mark.property
@given(pairs=st.lists(st.tuples(st.text(max_size=20), _json_values), max_size=8), data=st.data())
def test_hash_invariant_under_dict_insertion_order(
    pairs: list[tuple[str, object]], data: st.DataObject
) -> None:
    permutation = data.draw(st.permutations(pairs))
    original = dict(pairs)
    shuffled = dict(permutation)
    if original == shuffled:  # same mapping regardless of insertion order
        assert content_hash(original) == content_hash(shuffled)


@pytest.mark.property
@given(value=_json_values)
def test_canonical_roundtrip(value: object) -> None:
    assert json.loads(canonical_json_bytes(value)) == value


@pytest.mark.property
@given(value=_json_values)
def test_canonicalization_is_stable(value: object) -> None:
    once = canonical_json_bytes(value)
    again = canonical_json_bytes(json.loads(once))
    assert once == again
