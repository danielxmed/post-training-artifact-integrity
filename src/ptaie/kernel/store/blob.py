"""Content-addressed immutable byte store.

Blobs are immutable by construction: there is no update or delete API, so
every workspace state ever staged remains addressable for the whole episode
(reversibility until an explicit boundary) and rollback is O(1).
"""

from ptaie.kernel.canonical import sha256_hex
from ptaie.kernel.errors import MissingBlobError


class BlobStore:
    """In-memory content-addressed blob store (fresh per episode)."""

    __slots__ = ("_blobs",)

    def __init__(self) -> None:
        self._blobs: dict[str, bytes] = {}

    def put(self, data: bytes | bytearray) -> str:
        """Store bytes; returns the sha256 hex hash. Idempotent."""
        blob = bytes(data)
        blob_hash = sha256_hex(blob)
        self._blobs.setdefault(blob_hash, blob)
        return blob_hash

    def get(self, blob_hash: str) -> bytes:
        try:
            return self._blobs[blob_hash]
        except KeyError:
            raise MissingBlobError(f"blob {blob_hash} not in store") from None

    def has(self, blob_hash: str) -> bool:
        return blob_hash in self._blobs

    def __len__(self) -> int:
        return len(self._blobs)
