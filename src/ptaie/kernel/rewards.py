"""Reward vector, constraint vector, and terminal semantics.

Design invariants (binding, see CLAUDE.md):

- The environment emits a reward **vector** plus a constraint **vector**,
  never a bare scalar. ``RewardVector`` deliberately has no ``total``,
  ``scalar``, or ``__float__`` surface (guardrail-tested); scalarization
  belongs to trainer adapters.
- Exactly five agent-selectable terminal dispositions. ``TerminalCode`` is
  the superset the engine reports, adding engine-imposed endings that are not
  dispositions (budget truncation, constraint halt).
- Constraint violations are non-compensable: they cannot be traded off
  against semantic success. They latch — once true, they stay true for the
  rest of the episode.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class TerminalDisposition(StrEnum):
    """The exactly-five agent-selectable terminal dispositions."""

    COMMIT = "commit"
    VERIFIED_NOOP = "verified_noop"
    ABSTAIN = "abstain"
    DEFER_ESCALATE = "defer_escalate"
    PARTIAL_HANDOFF = "partial_handoff"


class TerminalCode(StrEnum):
    """What ``step()`` reports at episode end.

    A superset of :class:`TerminalDisposition`: the two extra members are
    engine-imposed endings, not dispositions the agent can select.
    """

    COMMIT = "commit"
    VERIFIED_NOOP = "verified_noop"
    ABSTAIN = "abstain"
    DEFER_ESCALATE = "defer_escalate"
    PARTIAL_HANDOFF = "partial_handoff"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CONSTRAINT_HALT = "constraint_halt"


class OutcomeClass(StrEnum):
    """Hidden-side outcome classification (metrics/eval; never observed by the agent)."""

    CORRECT_COMMIT = "correct_commit"
    FALSE_COMMIT = "false_commit"
    CORRECT_NOOP = "correct_noop"
    FALSE_NOOP = "false_noop"
    JUSTIFIED_ABSTAIN = "justified_abstain"
    BLANKET_REFUSAL = "blanket_refusal"
    JUSTIFIED_DEFER = "justified_defer"
    UNJUSTIFIED_DEFER = "unjustified_defer"
    SAFE_PARTIAL_HANDOFF = "safe_partial_handoff"
    INVALID_PARTIAL_HANDOFF = "invalid_partial_handoff"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    INTEGRITY_VIOLATION = "integrity_violation"


class RewardVector(BaseModel):
    """Per-step and terminal reward components.

    Per-step vectors are zero except ``progress`` and ``resource_cost``; the
    full vector is emitted on the terminal step. ``resource_cost`` is >= 0 and
    represents consumption (trainer adapters subtract it).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    semantic: float = 0.0
    disposition: float = 0.0
    provenance: float = 0.0
    evidence: float = 0.0
    calibration: float = 0.0
    progress: float = 0.0
    resource_cost: float = 0.0

    @classmethod
    def zero(cls) -> "RewardVector":
        return cls()


class ConstraintVector(BaseModel):
    """Latched, non-compensable constraint flags. ``True`` means violated.

    ``hard_policy`` covers declared per-episode hard invariants;
    ``prohibited_info_loss`` corresponds to NORTH_STAR's ``c_secret``
    (protected records/fields destroyed).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tamper: bool = False
    escape: bool = False
    prohibited_info_loss: bool = False
    irreversible: bool = False
    hard_policy: bool = False

    @property
    def any_violation(self) -> bool:
        return (
            self.tamper
            or self.escape
            or self.prohibited_info_loss
            or self.irreversible
            or self.hard_policy
        )

    def latch(self, **flags: bool) -> "ConstraintVector":
        """Return a copy with the given flags set. Latched flags never clear:
        attempting to lower a set flag raises ``ValueError``."""
        for name, value in flags.items():
            if name not in type(self).model_fields:
                raise ValueError(f"unknown constraint flag {name!r}")
            if not value and getattr(self, name):
                raise ValueError(f"constraint flag {name!r} is latched and cannot be cleared")
        updates = {name: value for name, value in flags.items() if value}
        return self.model_copy(update=updates) if updates else self
