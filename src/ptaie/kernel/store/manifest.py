"""Flat, sorted path→blob manifests.

Paths are workspace-relative POSIX strings: no ``..``, no leading ``/``, no
backslashes, no drive letters — and no symlink concept exists at all, which
structurally eliminates traversal exploits inside the store. Entries are
kept sorted by path so the manifest hash is independent of insertion order.
"""

from typing import Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ptaie.kernel.canonical import content_hash
from ptaie.kernel.errors import ManifestError, PathViolationError
from ptaie.kernel.ids import require_hex256

DEFAULT_MEDIA_TYPE = "application/octet-stream"

_MANIFEST_SCHEMA_VERSION = 1


def validate_workspace_path(path: str) -> str:
    """Validate a workspace-relative POSIX path; returns it unchanged.

    Raises :class:`PathViolationError` on anything absolute, traversing,
    Windows-flavored, or malformed.
    """
    if not path or len(path) > 1024:
        raise PathViolationError(f"invalid path: empty or too long: {path[:64]!r}")
    if "\\" in path:
        raise PathViolationError(f"invalid path (backslash): {path!r}")
    if path.startswith("/"):
        raise PathViolationError(f"invalid path (absolute): {path!r}")
    if len(path) >= 2 and path[1] == ":":
        raise PathViolationError(f"invalid path (drive letter): {path!r}")
    if any(ord(ch) < 0x20 for ch in path):
        raise PathViolationError("invalid path (control characters)")
    segments = path.split("/")
    for segment in segments:
        if segment in ("", ".", ".."):
            raise PathViolationError(f"invalid path segment {segment!r} in {path!r}")
    return path


class ManifestEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    blob: str
    media_type: str = DEFAULT_MEDIA_TYPE

    @field_validator("path")
    @classmethod
    def _valid_path(cls, value: str) -> str:
        return validate_workspace_path(value)

    @field_validator("blob")
    @classmethod
    def _valid_blob(cls, value: str) -> str:
        return require_hex256(value, what="blob hash")


class ManifestDiff(BaseModel):
    """Paths added / removed / changed going from one manifest to another."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    added: tuple[str, ...]
    removed: tuple[str, ...]
    changed: tuple[str, ...]

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.removed or self.changed)


class Manifest(BaseModel):
    """An immutable, sorted set of manifest entries with a canonical hash."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entries: tuple[ManifestEntry, ...] = ()

    @field_validator("entries")
    @classmethod
    def _sort_entries(cls, value: tuple[ManifestEntry, ...]) -> tuple[ManifestEntry, ...]:
        return tuple(sorted(value, key=lambda entry: entry.path))

    @model_validator(mode="after")
    def _unique_paths(self) -> Self:
        paths = [entry.path for entry in self.entries]
        if len(set(paths)) != len(paths):
            raise ValueError("manifest paths must be unique")
        return self

    @property
    def manifest_hash(self) -> str:
        return content_hash(
            {
                "v": _MANIFEST_SCHEMA_VERSION,
                "entries": [
                    {"path": entry.path, "blob": entry.blob, "media_type": entry.media_type}
                    for entry in self.entries
                ],
            }
        )

    def paths(self) -> tuple[str, ...]:
        return tuple(entry.path for entry in self.entries)

    def get(self, path: str) -> ManifestEntry | None:
        for entry in self.entries:
            if entry.path == path:
                return entry
        return None

    def with_entry(self, path: str, blob: str, media_type: str = DEFAULT_MEDIA_TYPE) -> "Manifest":
        """Return a manifest with ``path`` added or replaced."""
        validate_workspace_path(path)
        kept = tuple(entry for entry in self.entries if entry.path != path)
        new_entry = ManifestEntry(path=path, blob=blob, media_type=media_type)
        return Manifest(entries=(*kept, new_entry))

    def without(self, path: str) -> "Manifest":
        """Return a manifest with ``path`` removed; error if absent."""
        if self.get(path) is None:
            raise ManifestError(f"cannot remove absent path {path!r}")
        return Manifest(entries=tuple(entry for entry in self.entries if entry.path != path))

    def diff(self, other: "Manifest") -> ManifestDiff:
        """Diff from ``self`` to ``other``."""
        mine = {entry.path: entry.blob for entry in self.entries}
        theirs = {entry.path: entry.blob for entry in other.entries}
        added = tuple(sorted(path for path in theirs if path not in mine))
        removed = tuple(sorted(path for path in mine if path not in theirs))
        changed = tuple(
            sorted(path for path in mine if path in theirs and mine[path] != theirs[path])
        )
        return ManifestDiff(added=added, removed=removed, changed=changed)
