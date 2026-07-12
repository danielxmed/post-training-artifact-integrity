"""The clarification oracle: deterministic structured answers.

M1 restricts clarification to structured dimension queries keyed by name; the
oracle answers by table lookup on the task's scripted entries. Free-text
question text is logged for interpretability but never parsed, so there is no
NLU and no fingerprintable phrasing.
"""

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from ptaie.kernel.task import ClarificationEntry

type AnswerKind = Literal["value", "unknown", "already_stated", "budget_exhausted", "no_such_key"]


class ClarificationResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    question_key: str | None
    answer_kind: AnswerKind
    answer_value: str | None = None


@runtime_checkable
class ClarificationOracle(Protocol):
    def answer(
        self, question_key: str | None, asked_before: frozenset[str]
    ) -> ClarificationResponse: ...


class ScriptedOracle:
    """Answers from the task's scripted clarification entries.

    A repeated question (already in ``asked_before``) returns
    ``already_stated`` — the oracle never does the agent's inspection twice.
    An unknown key returns ``no_such_key``.
    """

    __slots__ = ("_by_key",)

    def __init__(self, script: tuple[ClarificationEntry, ...]) -> None:
        self._by_key = {entry.question_key: entry for entry in script}

    def answer(
        self, question_key: str | None, asked_before: frozenset[str]
    ) -> ClarificationResponse:
        if question_key is None or question_key not in self._by_key:
            return ClarificationResponse(question_key=question_key, answer_kind="no_such_key")
        if question_key in asked_before:
            return ClarificationResponse(question_key=question_key, answer_kind="already_stated")
        entry = self._by_key[question_key]
        return ClarificationResponse(
            question_key=question_key,
            answer_kind=entry.answer_kind,
            answer_value=entry.answer_value,
        )
