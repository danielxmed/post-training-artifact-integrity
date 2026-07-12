"""Deterministic seed derivation and labeled RNG substreams.

This is the only randomness source permitted in the kernel and in artifact
class plugins (guardrail-tested). Substreams isolate consumption: components
drawing from sibling substreams cannot perturb each other's sequences, so an
extra draw in one tool call never shifts every later sample in the episode.
"""

import hashlib
import random
from collections.abc import MutableSequence, Sequence

_NAMESPACE = "ptaie"


def derive_seed(root: int, *labels: str) -> int:
    """Derive a 64-bit seed from a root seed and a label path.

    Labels must be non-empty and must not contain ``/`` (the join separator),
    so distinct label paths can never collide.
    """
    for label in labels:
        if not label or "/" in label:
            raise ValueError(f"invalid rng label {label!r}: must be non-empty and contain no '/'")
    material = f"{_NAMESPACE}:{root}:{'/'.join(labels)}"
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


class DerivedRng:
    """A seeded RNG that can fork labeled, independent substreams."""

    __slots__ = ("_random", "_seed")

    def __init__(self, seed: int) -> None:
        self._seed = seed
        self._random = random.Random(seed)

    @property
    def seed(self) -> int:
        return self._seed

    def substream(self, label: str) -> "DerivedRng":
        """Fork an independent substream; forking does not consume from this stream."""
        return DerivedRng(derive_seed(self._seed, label))

    def random(self) -> float:
        return self._random.random()

    def randint(self, low: int, high: int) -> int:
        return self._random.randint(low, high)

    def uniform(self, low: float, high: float) -> float:
        return self._random.uniform(low, high)

    def choice[T](self, seq: Sequence[T]) -> T:
        return self._random.choice(seq)

    def shuffle(self, seq: MutableSequence[object]) -> None:
        self._random.shuffle(seq)

    def sample[T](self, population: Sequence[T], k: int) -> list[T]:
        return self._random.sample(population, k)

    def __repr__(self) -> str:
        return f"DerivedRng(seed={self._seed})"
