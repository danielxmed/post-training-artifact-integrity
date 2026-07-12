"""Claim bundles — the substrate of the Evidence-Locked Commit.

The agent submits a :class:`ClaimBundle` with its terminal action. The engine
seals it (a ``CLAIM_SEALED`` audit event, appended *before* any hidden
verification runs) and only then verifies. Sealing binds artifact hash,
disposition, claims, evidence references, and confidence so none of them can
be rewritten after the agent observes verifier results.

:class:`SealedClaimBundle` instances are only meaningful when produced by
``AuditLog.seal_claim``: finalization re-verifies ``seal_event`` against the
environment-owned log, so a hand-built instance is rejected there — the log,
not the type constructor, is the root of trust.
"""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ptaie.kernel.ids import AuditRef, require_hex256
from ptaie.kernel.rewards import TerminalDisposition

type ClaimStatement = Literal["restored", "verified_intact", "not_addressed", "cannot_determine"]

_Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


class Claim(BaseModel):
    """A structured claim about one invariant from the public claimable vocabulary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    claim_id: str
    invariant_id: str
    statement: ClaimStatement
    evidence_refs: tuple[AuditRef, ...] = ()
    confidence: _Confidence | None = None


class ClaimBundle(BaseModel):
    """The agent's terminal attestation, sealed before hidden verification.

    ``confidence`` is P(the chosen disposition is the correct one); the
    calibration reward scores it against the disposition-correctness
    indicator. ``rationale`` is free prose: logged for interpretability,
    never reward-bearing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    disposition: TerminalDisposition
    artifact_root_hash: str
    claims: tuple[Claim, ...] = ()
    confidence: _Confidence
    unresolved: tuple[str, ...] = ()
    declared_irreversible: tuple[str, ...] = ()
    rationale: str = ""

    @model_validator(mode="after")
    def _check(self) -> Self:
        require_hex256(self.artifact_root_hash, what="artifact_root_hash")
        claim_ids = [claim.claim_id for claim in self.claims]
        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError("claim ids must be unique")
        if self.disposition is TerminalDisposition.PARTIAL_HANDOFF and not self.unresolved:
            raise ValueError("partial_handoff requires a non-empty 'unresolved' list")
        return self


class SealedClaimBundle(BaseModel):
    """A claim bundle bound to a ``CLAIM_SEALED`` audit event.

    Produced by ``AuditLog.seal_claim``. Finalization verifies that
    ``seal_event`` matches the environment-owned log and that ``bundle_hash``
    is the canonical hash of ``bundle`` — forged instances fail there.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    bundle: ClaimBundle
    bundle_hash: str
    sealed_at_step: Annotated[int, Field(ge=0)]
    seal_event: AuditRef

    @model_validator(mode="after")
    def _check(self) -> Self:
        require_hex256(self.bundle_hash, what="bundle_hash")
        return self
