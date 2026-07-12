import pytest

from ptaie.kernel.canonical import sha256_hex
from ptaie.kernel.errors import ManifestError, MissingBlobError, PathViolationError
from ptaie.kernel.store import BlobStore, Manifest, ManifestEntry, validate_workspace_path


def test_blob_roundtrip_and_idempotence() -> None:
    store = BlobStore()
    h1 = store.put(b"hello")
    h2 = store.put(b"hello")
    assert h1 == h2 == sha256_hex(b"hello")
    assert store.get(h1) == b"hello"
    assert store.has(h1)
    assert len(store) == 1


def test_missing_blob_raises() -> None:
    store = BlobStore()
    with pytest.raises(MissingBlobError):
        store.get("0" * 64)


def test_bytearray_input_is_copied() -> None:
    store = BlobStore()
    data = bytearray(b"mutable")
    blob_hash = store.put(data)
    data[0] = 0
    assert store.get(blob_hash) == b"mutable"


@pytest.mark.parametrize(
    "path",
    ["data/train.jsonl", "dataset_card.json", "checks/format_check.py", "a/b/c.txt"],
)
def test_valid_workspace_paths(path: str) -> None:
    assert validate_workspace_path(path) == path


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/abs/path",
        "a\\b",
        "C:/windows",
        "c:sneaky",
        "x/C:/y",
        "file.txt:stream",
        "../escape",
        "a/../b",
        "a//b",
        "a/./b",
        ".",
        "..",
        "a/\x00b",
        "a/\x7fb",
        "a/\x85b",
        "a/‮b",
        "a/​b",
    ],
)
def test_invalid_workspace_paths(path: str) -> None:
    with pytest.raises(PathViolationError):
        validate_workspace_path(path)


def _entry(path: str, data: bytes) -> ManifestEntry:
    return ManifestEntry(path=path, blob=sha256_hex(data))


def test_manifest_hash_independent_of_insertion_order() -> None:
    a = _entry("a.txt", b"a")
    b = _entry("b.txt", b"b")
    assert Manifest(entries=(a, b)).manifest_hash == Manifest(entries=(b, a)).manifest_hash
    assert Manifest(entries=(b, a)).paths() == ("a.txt", "b.txt")


def test_manifest_rejects_duplicate_paths() -> None:
    a1 = _entry("a.txt", b"one")
    a2 = _entry("a.txt", b"two")
    with pytest.raises(ValueError, match="unique"):
        Manifest(entries=(a1, a2))


def test_manifest_entry_validates_blob_hash() -> None:
    with pytest.raises(ValueError, match="sha256"):
        ManifestEntry(path="a.txt", blob="nothex")


def test_with_entry_add_and_replace() -> None:
    manifest = Manifest(entries=(_entry("a.txt", b"one"),))
    grown = manifest.with_entry("b.txt", sha256_hex(b"b"))
    assert grown.paths() == ("a.txt", "b.txt")
    replaced = grown.with_entry("a.txt", sha256_hex(b"new"))
    entry = replaced.get("a.txt")
    assert entry is not None and entry.blob == sha256_hex(b"new")
    # originals untouched (frozen, CoW)
    original = manifest.get("a.txt")
    assert original is not None and original.blob == sha256_hex(b"one")


def test_without_absent_path_raises() -> None:
    manifest = Manifest(entries=(_entry("a.txt", b"a"),))
    with pytest.raises(ManifestError):
        manifest.without("missing.txt")


def test_diff_agrees_with_manifest_hash_on_media_type_changes() -> None:
    blob = sha256_hex(b"same-bytes")
    plain = Manifest(entries=(ManifestEntry(path="a.txt", blob=blob, media_type="text/plain"),))
    json_typed = Manifest(
        entries=(ManifestEntry(path="a.txt", blob=blob, media_type="application/json"),)
    )
    assert plain.manifest_hash != json_typed.manifest_hash
    diff = plain.diff(json_typed)
    assert diff.changed == ("a.txt",)
    assert not diff.is_empty  # diff emptiness must track manifest identity


def test_diff() -> None:
    base = Manifest(entries=(_entry("keep.txt", b"k"), _entry("edit.txt", b"v1")))
    target = Manifest(
        entries=(_entry("keep.txt", b"k"), _entry("edit.txt", b"v2"), _entry("new.txt", b"n"))
    )
    diff = base.diff(target)
    assert diff.added == ("new.txt",)
    assert diff.removed == ()
    assert diff.changed == ("edit.txt",)
    assert base.diff(base).is_empty
