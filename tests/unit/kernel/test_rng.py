import pytest

from ptaie.kernel.rng import DerivedRng, derive_seed


def test_derive_seed_deterministic_and_label_sensitive() -> None:
    assert derive_seed(42, "taskgen") == derive_seed(42, "taskgen")
    assert derive_seed(42, "taskgen") != derive_seed(42, "contract")
    assert derive_seed(42, "taskgen") != derive_seed(43, "taskgen")


def test_label_paths_cannot_collide() -> None:
    assert derive_seed(1, "a", "b") != derive_seed(1, "ab")
    with pytest.raises(ValueError, match="label"):
        derive_seed(1, "a/b")
    with pytest.raises(ValueError, match="label"):
        derive_seed(1, "")


def test_same_seed_same_sequence() -> None:
    a = DerivedRng(7)
    b = DerivedRng(7)
    assert [a.randint(0, 1000) for _ in range(20)] == [b.randint(0, 1000) for _ in range(20)]


def test_substreams_are_isolated_from_parent_consumption() -> None:
    fresh = DerivedRng(99)
    child_of_fresh = fresh.substream("content")
    expected = [child_of_fresh.random() for _ in range(10)]

    consumed = DerivedRng(99)
    for _ in range(50):
        consumed.random()  # heavy parent consumption
    child_after_consumption = consumed.substream("content")
    assert [child_after_consumption.random() for _ in range(10)] == expected


def test_sibling_substreams_differ() -> None:
    root = DerivedRng(5)
    a = root.substream("a")
    b = root.substream("b")
    assert [a.random() for _ in range(5)] != [b.random() for _ in range(5)]
