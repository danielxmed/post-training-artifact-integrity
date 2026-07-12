"""A deterministic stub Finalizer for PR4 lifecycle tests.

PR4 exercises the episode runtime (tools, budgets, seal-then-verify ordering,
observations, replay) without the real scorer. The stub runs no hidden
verification: it echoes the latched constraint, classifies purely from the
chosen disposition (never from hidden latent state), and returns a fixed
reward vector. The real sft_chat finalizer arrives in PR5.
"""

from ptaie.kernel.audit import AuditLog
from ptaie.kernel.claims import SealedClaimBundle
from ptaie.kernel.rewards import ConstraintVector, OutcomeClass, RewardVector, TerminalDisposition
from ptaie.kernel.store.workspace import WorkspaceReadView
from ptaie.kernel.task import TaskRecord
from ptaie.kernel.verification import VerificationReport

_OUTCOME = {
    TerminalDisposition.COMMIT: OutcomeClass.CORRECT_COMMIT,
    TerminalDisposition.VERIFIED_NOOP: OutcomeClass.CORRECT_NOOP,
    TerminalDisposition.ABSTAIN: OutcomeClass.JUSTIFIED_ABSTAIN,
    TerminalDisposition.DEFER_ESCALATE: OutcomeClass.JUSTIFIED_DEFER,
    TerminalDisposition.PARTIAL_HANDOFF: OutcomeClass.SAFE_PARTIAL_HANDOFF,
}


class StubFinalizer:
    def finalize(
        self,
        *,
        task: TaskRecord,
        initial_view: WorkspaceReadView,
        final_view: WorkspaceReadView,
        sealed: SealedClaimBundle,
        constraint: ConstraintVector,
        audit: AuditLog,
        resource_cost: float,
    ) -> VerificationReport:
        outcome = (
            OutcomeClass.INTEGRITY_VIOLATION
            if constraint.any_violation
            else _OUTCOME[sealed.bundle.disposition]
        )
        return VerificationReport(
            sealed_bundle_hash=sealed.bundle_hash,
            results=(),
            claim_assessments=(),
            semantic_pass=not constraint.any_violation,
            outcome_class=outcome,
            constraint=constraint,
            reward=RewardVector.zero(),
        )
