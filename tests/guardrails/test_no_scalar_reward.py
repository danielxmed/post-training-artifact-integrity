"""The reward vector must never grow a scalar surface (binding invariant)."""

from ptaie.kernel.rewards import ConstraintVector, RewardVector


def test_reward_vector_has_no_scalar_surface() -> None:
    for forbidden in ("total", "scalar", "sum", "__float__", "__int__"):
        assert not hasattr(RewardVector, forbidden), (
            f"RewardVector must not expose {forbidden!r}: scalarization belongs to trainer "
            "adapters, never the kernel"
        )
    instance = RewardVector.zero()
    for forbidden in ("total", "scalar"):
        assert not hasattr(instance, forbidden)


def test_reward_vector_fields_are_exactly_the_seven_components() -> None:
    assert set(RewardVector.model_fields) == {
        "semantic",
        "disposition",
        "provenance",
        "evidence",
        "calibration",
        "progress",
        "resource_cost",
    }
    assert all(field.annotation is float for field in RewardVector.model_fields.values())


def test_constraint_vector_fields_are_exactly_the_five_flags() -> None:
    assert set(ConstraintVector.model_fields) == {
        "tamper",
        "escape",
        "prohibited_info_loss",
        "irreversible",
        "hard_policy",
    }
    assert all(field.annotation is bool for field in ConstraintVector.model_fields.values())


def test_constraint_latching_never_clears() -> None:
    vector = ConstraintVector().latch(tamper=True)
    assert vector.tamper and vector.any_violation
    try:
        vector.latch(tamper=False)
    except ValueError:
        pass
    else:  # pragma: no cover - failure branch
        raise AssertionError("lowering a latched constraint flag must raise")
