from ptaie.kernel.budgets import BudgetCharge, BudgetLedger, Budgets


def _ledger() -> BudgetLedger:
    return BudgetLedger(Budgets(max_steps=3, max_cost_units=10, max_questions=1, max_mutations=2))


def test_charge_applies_and_view_updates() -> None:
    ledger = _ledger()
    result = ledger.charge(BudgetCharge(steps=1, cost_units=4, mutations=1))
    assert result.applied and result.exhausted == ()
    view = ledger.view()
    assert view.remaining_steps == 2
    assert view.remaining_cost_units == 6
    assert view.remaining_questions == 1
    assert view.remaining_mutations == 1


def test_refusal_is_atomic_and_names_all_insufficient_components() -> None:
    ledger = _ledger()
    before = ledger.view()
    result = ledger.charge(BudgetCharge(steps=5, cost_units=11, questions=2, mutations=0))
    assert not result.applied
    assert result.exhausted == ("steps", "cost_units", "questions")
    assert ledger.view() == before  # nothing deducted


def test_exhaustion_boundary() -> None:
    ledger = _ledger()
    for _ in range(3):
        assert ledger.charge(BudgetCharge(steps=1)).applied
    result = ledger.charge(BudgetCharge(steps=1))
    assert not result.applied and result.exhausted == ("steps",)
