"""The audit log is environment-owned and append-only (binding invariant)."""

from ptaie.kernel.audit import AuditEventKind, AuditLog

EXPECTED_PUBLIC_API = {
    "append",
    "seal_claim",
    "events",
    "get",
    "head_hash",
    "verify_chain",
    "verify_ref",
    "episode_id",
}


def test_public_api_has_no_mutators_beyond_append() -> None:
    public = {name for name in dir(AuditLog) if not name.startswith("_")}
    assert public == EXPECTED_PUBLIC_API
    for forbidden in ("remove", "pop", "clear", "delete", "update", "__setitem__", "__delitem__"):
        assert not hasattr(AuditLog, forbidden)


def test_events_returns_a_copy_not_the_internal_list() -> None:
    log = AuditLog("ep-guard")
    log.append(AuditEventKind.ACTION, "agent", {}, step_index=0)
    events = log.events()
    assert isinstance(events, tuple)
    assert len(events) == 1
    assert len(log.events()) == 1
