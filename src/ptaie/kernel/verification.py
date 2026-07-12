"""Verification data models and the validator/verifier protocols.

Layered verification: all applicable layers always run at finalization and
every result is recorded — passing an earlier layer never conceals failure at
a later one. A check that cannot run because an earlier defect blocks it
reports ``BLOCKED``, which counts as not-passed, never as passed.

Visible validators are agent-runnable and deliberately weaker than hidden
verifiers. Hidden verifiers run only after the claim bundle is sealed.
"""

from enum import IntEnum, StrEnum
from typing import Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, model_validator

from ptaie.kernel.canonical import JsonValue
from ptaie.kernel.claims import SealedClaimBundle
from ptaie.kernel.contract import HiddenBaseModel
from ptaie.kernel.rewards import ConstraintVector, OutcomeClass, RewardVector
from ptaie.kernel.store.workspace import WorkspaceReadView
from ptaie.kernel.task import TaskHidden


class VerifierLayer(IntEnum):
    """Verification layers. M1 ships 0-4; 5-8 are reserved slots."""

    INTEGRITY = 0
    REPRESENTATION = 1
    SCHEMA = 2
    LOCAL_SEMANTICS = 3
    RELATIONAL = 4
    PROVENANCE = 5
    METAMORPHIC = 6
    DOWNSTREAM = 7
    ADVERSARIAL = 8


class CheckStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"


class VerifierResult(BaseModel):
    """The outcome of one check within one layer.

    Dual-use (visible validators and hidden verifiers both emit it), so it
    keeps a plain repr — hidden-layer results must only be rendered through
    the enclosing, redacted :class:`VerificationReport`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    check_id: str
    layer: VerifierLayer
    status: CheckStatus
    failure_codes: tuple[str, ...] = ()
    details: dict[str, JsonValue] = {}

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.status is CheckStatus.PASSED and self.failure_codes:
            raise ValueError("passed checks must not carry failure codes")
        return self


class ClaimAssessment(HiddenBaseModel):
    """Whether the audit events a claim cites actually evidence the claim."""

    claim_id: str
    supported: bool
    reason: str


class VerificationReport(HiddenBaseModel):
    """The complete hidden-verification outcome for one sealed claim bundle.

    Environment-side only (hence the redacted repr): the agent observes the
    terminal code, never this report.
    """

    sealed_bundle_hash: str
    results: tuple[VerifierResult, ...]
    claim_assessments: tuple[ClaimAssessment, ...] = ()
    semantic_pass: bool
    outcome_class: OutcomeClass
    constraint: ConstraintVector
    reward: RewardVector


@runtime_checkable
class Validator(Protocol):
    """An agent-runnable visible check (weaker than hidden verifiers).

    Must be pure and deterministic in the read view it is given, and its
    result messages must come from fixed templates (leakage-tested).
    """

    @property
    def check_id(self) -> str: ...

    @property
    def layer(self) -> VerifierLayer: ...

    def check(self, view: WorkspaceReadView) -> tuple[VerifierResult, ...]: ...


@runtime_checkable
class Verifier(Protocol):
    """A hidden check, run only at finalization after the bundle is sealed.

    Takes read-only views plus hidden task state; pure and deterministic.
    The signature deliberately admits moving verifier execution behind a
    process boundary later without interface change.
    """

    @property
    def check_id(self) -> str: ...

    @property
    def layer(self) -> VerifierLayer: ...

    def check(
        self,
        view: WorkspaceReadView,
        sealed: SealedClaimBundle,
        hidden: TaskHidden,
    ) -> tuple[VerifierResult, ...]: ...
