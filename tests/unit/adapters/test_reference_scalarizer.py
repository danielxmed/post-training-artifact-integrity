"""The reference scalarizer proves the vector->scalar seam and enforces
non-compensability: a latched constraint cannot be outweighed by reward."""

from ptaie.adapters.reference_scalarizer import scalarize
from ptaie.kernel.rewards import ConstraintVector, RewardVector


def test_clean_reward_scalarizes_positive() -> None:
    reward = RewardVector(semantic=1.0, disposition=1.0, provenance=1.0, evidence=1.0)
    assert scalarize(reward, ConstraintVector()) > 0


def test_constraint_floors_below_any_clean_episode() -> None:
    perfect = RewardVector(
        semantic=1.0, disposition=1.0, provenance=1.0, evidence=1.0, calibration=0.0
    )
    tampering = scalarize(perfect, ConstraintVector(tamper=True))
    honest_but_mediocre = scalarize(RewardVector(semantic=0.0), ConstraintVector())
    # a tampering episode, however semantically perfect, ranks below an honest one
    assert tampering < honest_but_mediocre
    assert tampering < 0


def test_every_constraint_flag_is_non_compensable() -> None:
    perfect = RewardVector(semantic=1.0, disposition=1.0, provenance=1.0, evidence=1.0)
    for flag in ("tamper", "escape", "prohibited_info_loss", "irreversible", "hard_policy"):
        constraint = ConstraintVector(**{flag: True})
        assert scalarize(perfect, constraint) < 0, flag


def test_resource_cost_is_subtracted() -> None:
    base = RewardVector(semantic=1.0)
    costly = RewardVector(semantic=1.0, resource_cost=1.0)
    assert scalarize(costly, ConstraintVector()) < scalarize(base, ConstraintVector())
