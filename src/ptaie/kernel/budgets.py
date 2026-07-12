"""Budgets and per-episode budget accounting.

Budget exhaustion never raises to the agent: an unaffordable charge is
refused deterministically (nothing is deducted) and the engine truncates the
episode with ``TerminalCode.BUDGET_EXHAUSTED``.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

_NonNegative = Annotated[int, Field(ge=0)]


class Budgets(BaseModel):
    """Immutable per-task budget limits (part of the public task record)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_steps: Annotated[int, Field(ge=1)]
    max_cost_units: _NonNegative
    max_questions: _NonNegative
    max_mutations: _NonNegative


class BudgetView(BaseModel):
    """Agent-visible remaining budgets (embedded in every observation)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    remaining_steps: _NonNegative
    remaining_cost_units: _NonNegative
    remaining_questions: _NonNegative
    remaining_mutations: _NonNegative


class BudgetCharge(BaseModel):
    """The cost of one action, per budget component."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    steps: _NonNegative = 1
    cost_units: _NonNegative = 0
    questions: _NonNegative = 0
    mutations: _NonNegative = 0


class BudgetChargeResult(BaseModel):
    """Outcome of attempting a charge. ``exhausted`` is empty iff ``applied``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    applied: bool
    exhausted: tuple[str, ...] = ()


class BudgetLedger:
    """Mutable per-episode budget accounting.

    Charging is atomic: if any component is insufficient, nothing is deducted
    and the result names every insufficient component in a fixed order.
    """

    __slots__ = ("_cost_units", "_mutations", "_questions", "_steps")

    def __init__(self, budgets: Budgets) -> None:
        self._steps = budgets.max_steps
        self._cost_units = budgets.max_cost_units
        self._questions = budgets.max_questions
        self._mutations = budgets.max_mutations

    def charge(self, charge: BudgetCharge) -> BudgetChargeResult:
        exhausted: list[str] = []
        if charge.steps > self._steps:
            exhausted.append("steps")
        if charge.cost_units > self._cost_units:
            exhausted.append("cost_units")
        if charge.questions > self._questions:
            exhausted.append("questions")
        if charge.mutations > self._mutations:
            exhausted.append("mutations")
        if exhausted:
            return BudgetChargeResult(applied=False, exhausted=tuple(exhausted))
        self._steps -= charge.steps
        self._cost_units -= charge.cost_units
        self._questions -= charge.questions
        self._mutations -= charge.mutations
        return BudgetChargeResult(applied=True)

    def view(self) -> BudgetView:
        return BudgetView(
            remaining_steps=self._steps,
            remaining_cost_units=self._cost_units,
            remaining_questions=self._questions,
            remaining_mutations=self._mutations,
        )
