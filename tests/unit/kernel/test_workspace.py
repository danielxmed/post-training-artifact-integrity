import pytest

from ptaie.kernel.errors import (
    MissingBlobError,
    UnknownSnapshotError,
    WorkspacePathNotFoundError,
)
from ptaie.kernel.store import BlobStore, Manifest, ManifestEntry, Workspace


def _workspace() -> Workspace:
    store = BlobStore()
    blob = store.put(b'{"id": "rec-1"}\n')
    manifest = Manifest(entries=(ManifestEntry(path="data/train.jsonl", blob=blob),))
    return Workspace(store, manifest)


def test_init_requires_blobs_present() -> None:
    store = BlobStore()
    manifest = Manifest(entries=(ManifestEntry(path="a.txt", blob="0" * 64),))
    with pytest.raises(MissingBlobError):
        Workspace(store, manifest)


def test_write_is_cow_and_extends_lineage() -> None:
    ws = _workspace()
    initial_root = ws.staged
    new_root = ws.write("data/train.jsonl", b"edited\n")
    assert new_root != initial_root
    assert ws.staged == new_root
    assert ws.read("data/train.jsonl") == b"edited\n"
    assert ws.lineage() == (initial_root, new_root)


def test_delete_and_missing_path() -> None:
    ws = _workspace()
    ws.delete("data/train.jsonl")
    with pytest.raises(WorkspacePathNotFoundError):
        ws.read("data/train.jsonl")
    with pytest.raises(WorkspacePathNotFoundError):
        ws.delete("data/train.jsonl")


def test_snapshot_rollback_restores_bytes() -> None:
    ws = _workspace()
    original = ws.read("data/train.jsonl")
    snap = ws.snapshot("before-edit")
    ws.write("data/train.jsonl", b"broken")
    ws.write("extra.txt", b"noise")
    restored_root = ws.rollback(snap)
    assert ws.staged == restored_root
    assert ws.read("data/train.jsonl") == original
    with pytest.raises(WorkspacePathNotFoundError):
        ws.read("extra.txt")
    # rollback extends history, never rewrites it
    assert len(ws.lineage()) == 4
    assert ws.lineage()[-1] == ws.lineage()[0]


def test_unknown_snapshot_raises() -> None:
    ws = _workspace()
    with pytest.raises(UnknownSnapshotError):
        ws.rollback("snap-999")


def test_read_view_is_a_stable_snapshot() -> None:
    ws = _workspace()
    view = ws.read_view()
    before = view.read("data/train.jsonl")
    ws.write("data/train.jsonl", b"changed")
    assert view.read("data/train.jsonl") == before
    assert ws.read("data/train.jsonl") == b"changed"
    assert view.staged_root != ws.staged


def test_diff_roots() -> None:
    ws = _workspace()
    root_a = ws.staged
    root_b = ws.write("new.txt", b"n")
    diff = ws.diff_roots(root_a, root_b)
    assert diff.added == ("new.txt",)
    with pytest.raises(UnknownSnapshotError):
        ws.diff_roots(root_a, "f" * 64)
