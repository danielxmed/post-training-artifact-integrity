from dataclasses import dataclass

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ptaie.kernel.store import BlobStore, Manifest, ManifestEntry, Workspace


@dataclass(frozen=True)
class WriteOp:
    path: str
    data: bytes


@dataclass(frozen=True)
class SnapshotOp:
    pass


@dataclass(frozen=True)
class RollbackOp:
    which: int  # index into existing snapshots, modulo


_paths = st.sampled_from(["data/train.jsonl", "dataset_card.json", "notes/a.txt"])
_ops = st.one_of(
    st.builds(WriteOp, path=_paths, data=st.binary(max_size=32)),
    st.builds(SnapshotOp),
    st.builds(RollbackOp, which=st.integers(min_value=0, max_value=10)),
)


@pytest.mark.property
@given(ops=st.lists(_ops, max_size=30))
def test_workspace_invariants_under_random_op_sequences(
    ops: list[WriteOp | SnapshotOp | RollbackOp],
) -> None:
    store = BlobStore()
    blob = store.put(b"initial")
    ws = Workspace(store, Manifest(entries=(ManifestEntry(path="data/train.jsonl", blob=blob),)))

    snapshot_ids: list[str] = []
    snapshot_contents: dict[str, dict[str, bytes]] = {}

    def current_contents() -> dict[str, bytes]:
        return {path: ws.read(path) for path in ws.manifest.paths()}

    for op in ops:
        match op:
            case WriteOp(path=path, data=data):
                ws.write(path, data)
            case SnapshotOp():
                snapshot_id = ws.snapshot("s")
                snapshot_ids.append(snapshot_id)
                snapshot_contents[snapshot_id] = current_contents()
            case RollbackOp(which=which):
                if snapshot_ids:
                    snapshot_id = snapshot_ids[which % len(snapshot_ids)]
                    ws.rollback(snapshot_id)
                    # rollback restores exact byte contents recorded at snapshot time
                    assert current_contents() == snapshot_contents[snapshot_id]

        # invariants after every op:
        assert ws.staged == ws.lineage()[-1]  # staged is always the lineage head
        assert ws.staged == ws.manifest.manifest_hash
        # every snapshot root stays addressable (nothing is ever destroyed)
        for sid in snapshot_ids:
            root = next(s.root for s in ws.snapshots() if s.snapshot_id == sid)
            ws.manifest_for(root)
