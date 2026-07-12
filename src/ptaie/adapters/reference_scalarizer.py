"""A reference vector->scalar scalarizer, proving the trainer-side seam.

Scalarization belongs to the trainer, never the kernel — this is a *reference*
only. It is lexicographic-flavored: any latched constraint is non-compensable
and forces a large negative floor no positive reward can offset, so a
semantically-correct-but-tampering episode can never outrank an honest one.
"""

from dataclasses import dataclass

from ptaie.kernel.rewards import ConstraintVector, RewardVector

_CONSTRAINT_FLOOR = -1000.0


@dataclass(frozen=True)
class ScalarizerWeights:
    semantic: float = 1.0
    disposition: float = 0.5
    provenance: float = 0.2
    evidence: float = 0.2
    calibration: float = 0.3
    progress: float = 0.1
    resource: float = 0.05


def scalarize(
    reward: RewardVector,
    constraint: ConstraintVector,
    weights: ScalarizerWeights | None = None,
) -> float:
    """Collapse the reward vector to a single scalar for a trainer.

    Non-compensable: if any constraint is latched the result is at most
    ``_CONSTRAINT_FLOOR`` plus the (non-positive) calibration/cost terms, so it
    cannot be raised above an honest episode by any amount of semantic reward.
    """
    w = weights or ScalarizerWeights()
    value = (
        w.semantic * reward.semantic
        + w.disposition * reward.disposition
        + w.provenance * reward.provenance
        + w.evidence * reward.evidence
        + w.calibration * reward.calibration
        + w.progress * reward.progress
        - w.resource * reward.resource_cost
    )
    if constraint.any_violation:
        return _CONSTRAINT_FLOOR + min(0.0, value)
    return value
