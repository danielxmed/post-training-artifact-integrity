"""Copy-on-write workspace over the blob store.

Every state the workspace has ever been in remains addressable during the
episode: blobs and manifests are never deleted, so rollback is O(1) and
nothing is destroyed — matching the "reversible until an explicit boundary"
invariant. ``lineage()`` (every staged root ever produced, in order) is the
M1 provenance record.
"""

from pydantic import BaseModel, ConfigDict

from ptaie.kernel.errors import (
    MissingBlobError,
    UnknownSnapshotError,
    WorkspacePathNotFoundError,
)
from ptaie.kernel.store.blob import BlobStore
from ptaie.kernel.store.manifest import (
    DEFAULT_MEDIA_TYPE,
    Manifest,
    ManifestDiff,
    validate_workspace_path,
)


class Snapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    snapshot_id: str
    label: str
    root: str


class WorkspaceReadView:
    """A read-only capability over one fixed manifest.

    Handed to tools, validators, and verifiers. Holds no reference to the
    workspace, the engine, or any hidden state — and the manifest it wraps is
    frozen, so the view is a stable snapshot of the state it was built from.
    """

    __slots__ = ("_manifest", "_store")

    def __init__(self, store: BlobStore, manifest: Manifest) -> None:
        self._store = store
        self._manifest = manifest

    @property
    def staged_root(self) -> str:
        return self._manifest.manifest_hash

    @property
    def manifest(self) -> Manifest:
        return self._manifest

    def paths(self) -> tuple[str, ...]:
        return self._manifest.paths()

    def has(self, path: str) -> bool:
        validate_workspace_path(path)
        return self._manifest.get(path) is not None

    def read(self, path: str) -> bytes:
        validate_workspace_path(path)
        entry = self._manifest.get(path)
        if entry is None:
            raise WorkspacePathNotFoundError(f"path not in workspace: {path!r}")
        return self._store.get(entry.blob)

    def __repr__(self) -> str:
        return f"WorkspaceReadView(root={self.staged_root[:12]}..., files={len(self.paths())})"


class Workspace:
    """The mutable staging area: CoW writes, snapshots, rollback, lineage."""

    __slots__ = (
        "_lineage",
        "_manifests",
        "_snap_counter",
        "_snapshot_order",
        "_snapshots",
        "_staged",
        "_store",
    )

    def __init__(self, store: BlobStore, initial: Manifest) -> None:
        for entry in initial.entries:
            if not store.has(entry.blob):
                raise MissingBlobError(
                    f"initial manifest references blob {entry.blob} missing from store"
                )
        self._store = store
        self._staged = initial
        self._manifests: dict[str, Manifest] = {initial.manifest_hash: initial}
        self._snapshots: dict[str, Snapshot] = {}
        self._snapshot_order: list[str] = []
        self._snap_counter = 0
        self._lineage: list[str] = [initial.manifest_hash]

    @property
    def staged(self) -> str:
        return self._staged.manifest_hash

    @property
    def manifest(self) -> Manifest:
        return self._staged

    def read(self, path: str) -> bytes:
        return self.read_view().read(path)

    def write(self, path: str, data: bytes, media_type: str = DEFAULT_MEDIA_TYPE) -> str:
        """CoW write: new blob + new manifest; returns the new staged root."""
        validate_workspace_path(path)
        blob_hash = self._store.put(data)
        new_manifest = self._staged.with_entry(path, blob_hash, media_type)
        return self._stage(new_manifest)

    def delete(self, path: str) -> str:
        validate_workspace_path(path)
        if self._staged.get(path) is None:
            raise WorkspacePathNotFoundError(f"path not in workspace: {path!r}")
        return self._stage(self._staged.without(path))

    def snapshot(self, label: str) -> str:
        """Record the current staged root under a deterministic snapshot id."""
        snapshot_id = f"snap-{self._snap_counter}"
        self._snap_counter += 1
        snap = Snapshot(snapshot_id=snapshot_id, label=label, root=self.staged)
        self._snapshots[snapshot_id] = snap
        self._snapshot_order.append(snapshot_id)
        return snapshot_id

    def rollback(self, snapshot_id: str) -> str:
        """Restore the staged manifest recorded by a snapshot. The rollback
        itself extends lineage — history is append-only, never rewritten."""
        snap = self._snapshots.get(snapshot_id)
        if snap is None:
            raise UnknownSnapshotError(f"unknown snapshot {snapshot_id!r}")
        manifest = self._manifests[snap.root]
        return self._stage(manifest)

    def diff_roots(self, root_a: str, root_b: str) -> ManifestDiff:
        """Diff two roots this workspace has staged (or snapshotted)."""
        manifest_a = self._manifests.get(root_a)
        manifest_b = self._manifests.get(root_b)
        if manifest_a is None or manifest_b is None:
            missing = root_a if manifest_a is None else root_b
            raise UnknownSnapshotError(f"root {missing} was never staged in this workspace")
        return manifest_a.diff(manifest_b)

    def lineage(self) -> tuple[str, ...]:
        """Every staged root ever produced, in order (the provenance record)."""
        return tuple(self._lineage)

    def snapshots(self) -> tuple[Snapshot, ...]:
        return tuple(self._snapshots[snapshot_id] for snapshot_id in self._snapshot_order)

    def manifest_for(self, root: str) -> Manifest:
        manifest = self._manifests.get(root)
        if manifest is None:
            raise UnknownSnapshotError(f"root {root} was never staged in this workspace")
        return manifest

    def read_view(self) -> WorkspaceReadView:
        """A stable read-only view of the *current* staged manifest."""
        return WorkspaceReadView(self._store, self._staged)

    def _stage(self, manifest: Manifest) -> str:
        self._staged = manifest
        self._manifests.setdefault(manifest.manifest_hash, manifest)
        self._lineage.append(manifest.manifest_hash)
        return manifest.manifest_hash
