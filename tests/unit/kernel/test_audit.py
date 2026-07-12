import pytest

from ptaie.kernel.audit import AuditEventKind, AuditLog
from ptaie.kernel.canonical import JsonValue, content_hash
from ptaie.kernel.claims import ClaimBundle
from ptaie.kernel.errors import CanonicalizationError
from ptaie.kernel.ids import GENESIS_HASH, AuditRef
from ptaie.kernel.rewards import TerminalDisposition


def _log() -> AuditLog:
    return AuditLog("ep-test")


def test_chain_appends_and_verifies() -> None:
    log = _log()
    assert log.head_hash() == GENESIS_HASH
    ref0 = log.append(AuditEventKind.EPISODE_START, "environment", {"seed": 1}, step_index=0)
    ref1 = log.append(AuditEventKind.ACTION, "agent", {"kind": "tool"}, step_index=1)
    assert (ref0.event_index, ref1.event_index) == (0, 1)
    assert log.verify_chain()
    assert log.head_hash() == ref1.event_hash
    assert log.get(1).prev_hash == ref0.event_hash


def test_verify_ref() -> None:
    log = _log()
    ref = log.append(AuditEventKind.ACTION, "agent", {}, step_index=0)
    assert log.verify_ref(ref)
    assert not log.verify_ref(AuditRef(episode_id="ep-test", event_index=0, event_hash="0" * 64))
    assert not log.verify_ref(
        AuditRef(episode_id="other", event_index=0, event_hash=ref.event_hash)
    )
    assert not log.verify_ref(
        AuditRef(episode_id="ep-test", event_index=5, event_hash=ref.event_hash)
    )


def test_payload_normalized_at_append_time() -> None:
    log = _log()
    inner: list[JsonValue] = ["a", "b"]
    payload: dict[str, JsonValue] = {"key": inner}
    log.append(AuditEventKind.TOOL_RESULT, "environment", payload, step_index=0)
    inner.append("mutated-later")
    assert log.verify_chain()
    assert log.get(0).payload == {"key": ["a", "b"]}


def test_non_canonical_payload_rejected() -> None:
    log = _log()
    with pytest.raises(CanonicalizationError):
        log.append(AuditEventKind.ACTION, "agent", {"bad": float("nan")}, step_index=0)


def test_seal_claim_appends_event_first_and_binds_hash() -> None:
    log = _log()
    bundle = ClaimBundle(
        disposition=TerminalDisposition.COMMIT,
        artifact_root_hash="a" * 64,
        confidence=0.8,
    )
    sealed = log.seal_claim(bundle, step_index=3)
    assert len(log) == 1
    event = log.get(0)
    assert event.kind is AuditEventKind.CLAIM_SEALED
    assert sealed.bundle_hash == content_hash(bundle.model_dump(mode="json"))
    assert event.payload["bundle_hash"] == sealed.bundle_hash
    assert sealed.sealed_at_step == 3
    assert log.verify_ref(sealed.seal_event)


def test_tamper_detection() -> None:
    log = _log()
    log.append(AuditEventKind.ACTION, "agent", {"n": 1}, step_index=0)
    log.append(AuditEventKind.ACTION, "agent", {"n": 2}, step_index=1)
    assert log.verify_chain()
    honest = log._events[0]
    forged = honest.model_copy(update={"payload": {"n": 999}})
    log._events[0] = forged
    assert not log.verify_chain()
