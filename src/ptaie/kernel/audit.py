"""The environment-owned, append-only, hash-chained audit log.

``event_index`` is the episode's logical clock — there is no wall-clock time
anywhere in the kernel. The only mutator is :meth:`AuditLog.append`;
tampering with recorded events is detectable because every event hash chains
over its predecessor.
"""

import json
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from ptaie.kernel.canonical import JsonValue, canonical_json_bytes, content_hash
from ptaie.kernel.claims import ClaimBundle, SealedClaimBundle
from ptaie.kernel.ids import GENESIS_HASH, AuditRef

type Actor = Literal["agent", "environment"]


class AuditEventKind(StrEnum):
    EPISODE_START = "episode_start"
    ACTION = "action"
    TOOL_RESULT = "tool_result"
    SNAPSHOT = "snapshot"
    ROLLBACK = "rollback"
    CLARIFICATION = "clarification"
    PROGRESS_EVENT = "progress_event"
    CONSTRAINT_FLAG = "constraint_flag"
    CLAIM_SEALED = "claim_sealed"
    VERIFICATION = "verification"
    EPISODE_END = "episode_end"


class AuditEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event_index: Annotated[int, Field(ge=0)]
    step_index: Annotated[int, Field(ge=0)]
    kind: AuditEventKind
    actor: Actor
    payload: dict[str, JsonValue]
    prev_hash: str
    event_hash: str


def _event_hash(
    *,
    prev_hash: str,
    event_index: int,
    step_index: int,
    kind: AuditEventKind,
    actor: Actor,
    payload: dict[str, JsonValue],
) -> str:
    return content_hash(
        {
            "prev_hash": prev_hash,
            "event_index": event_index,
            "step_index": step_index,
            "kind": kind.value,
            "actor": actor,
            "payload": payload,
        }
    )


class AuditLog:
    """Append-only event log for one episode.

    Payloads are normalized through canonical JSON at append time, so a
    caller mutating a dict it passed in cannot silently change recorded
    history.
    """

    __slots__ = ("_episode_id", "_events")

    def __init__(self, episode_id: str) -> None:
        self._episode_id = episode_id
        self._events: list[AuditEvent] = []

    @property
    def episode_id(self) -> str:
        return self._episode_id

    def __len__(self) -> int:
        return len(self._events)

    def append(
        self,
        kind: AuditEventKind,
        actor: Actor,
        payload: dict[str, JsonValue],
        *,
        step_index: int,
    ) -> AuditRef:
        normalized: dict[str, JsonValue] = json.loads(canonical_json_bytes(payload))
        event_index = len(self._events)
        prev_hash = self._events[-1].event_hash if self._events else GENESIS_HASH
        event_hash = _event_hash(
            prev_hash=prev_hash,
            event_index=event_index,
            step_index=step_index,
            kind=kind,
            actor=actor,
            payload=normalized,
        )
        event = AuditEvent(
            event_index=event_index,
            step_index=step_index,
            kind=kind,
            actor=actor,
            payload=normalized,
            prev_hash=prev_hash,
            event_hash=event_hash,
        )
        self._events.append(event)
        return AuditRef(episode_id=self._episode_id, event_index=event_index, event_hash=event_hash)

    def seal_claim(self, bundle: ClaimBundle, *, step_index: int) -> SealedClaimBundle:
        """Seal a claim bundle: the CLAIM_SEALED event is appended *before*
        the sealed bundle exists, so hidden verification (which requires a
        ``SealedClaimBundle``) can never precede sealing."""
        bundle_hash = content_hash(bundle.model_dump(mode="json"))
        ref = self.append(
            AuditEventKind.CLAIM_SEALED,
            "environment",
            {
                "bundle_hash": bundle_hash,
                "disposition": bundle.disposition.value,
                "artifact_root_hash": bundle.artifact_root_hash,
                "confidence": bundle.confidence,
            },
            step_index=step_index,
        )
        return SealedClaimBundle(
            bundle=bundle, bundle_hash=bundle_hash, sealed_at_step=step_index, seal_event=ref
        )

    def events(self) -> tuple[AuditEvent, ...]:
        """Environment-side read-only view. Agents only ever see the
        ``AuditRef`` values they were handed."""
        return tuple(self._events)

    def get(self, event_index: int) -> AuditEvent:
        return self._events[event_index]

    def head_hash(self) -> str:
        return self._events[-1].event_hash if self._events else GENESIS_HASH

    def verify_chain(self) -> bool:
        """Recompute the full hash chain; ``False`` means recorded history
        was tampered with."""
        prev_hash = GENESIS_HASH
        for expected_index, event in enumerate(self._events):
            if event.event_index != expected_index or event.prev_hash != prev_hash:
                return False
            recomputed = _event_hash(
                prev_hash=prev_hash,
                event_index=event.event_index,
                step_index=event.step_index,
                kind=event.kind,
                actor=event.actor,
                payload=event.payload,
            )
            if recomputed != event.event_hash:
                return False
            prev_hash = event.event_hash
        return True

    def verify_ref(self, ref: AuditRef) -> bool:
        """True iff the reference points at a real event in this log."""
        if ref.episode_id != self._episode_id:
            return False
        if not 0 <= ref.event_index < len(self._events):
            return False
        return self._events[ref.event_index].event_hash == ref.event_hash
