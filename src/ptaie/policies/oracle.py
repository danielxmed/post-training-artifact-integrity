"""The oracle policy: uses hidden task info to reach the correct disposition.

For each latent state it plays the mode the environment considers correct —
repair-and-commit, verified no-op, ask-then-commit, abstain, defer — using the
certified repair (computed from the hidden defect DAG) driven through the
public line-level tools. It is the M1 go/no-go instrument: it must reach the
expected disposition with no constraint violation, within budget, on every
accepted seed.
"""

from ptaie.kernel.actions import (
    Action,
    AskClarification,
    ReportInconsistency,
    TerminalAction,
    ToolAction,
)
from ptaie.kernel.claims import ClaimBundle
from ptaie.kernel.observation import Observation
from ptaie.kernel.rewards import TerminalDisposition
from ptaie.kernel.task import LatentTaskState
from ptaie.plugins.sft_chat.repairs import apply_repairs
from ptaie.plugins.sft_chat.taskgen import SftTask
from ptaie.policies.planning import plan_rewrite

_AMBIGUOUS_DIM = "dedup_policy"


class OraclePolicy:
    __slots__ = ("_disposition", "_index", "_prefix")

    def __init__(self, task: SftTask) -> None:
        self._prefix, self._disposition = _plan(task)
        self._index = 0

    def act(self, observation: Observation) -> Action:
        if self._index < len(self._prefix):
            action = self._prefix[self._index]
            self._index += 1
            return action
        return _terminal(self._disposition, observation.staged_root)


def _plan(task: SftTask) -> tuple[list[Action], TerminalDisposition]:
    state = task.latent_state
    if state is LatentTaskState.ALREADY_CORRECT:
        return [], TerminalDisposition.VERIFIED_NOOP
    if state is LatentTaskState.UNDERDETERMINED:
        return (
            [AskClarification(question_key=_AMBIGUOUS_DIM, question_text="dedup policy?")],
            TerminalDisposition.ABSTAIN,
        )
    if state is LatentTaskState.INCONSISTENT:
        return (
            [ReportInconsistency(description="dataset card contradicts its own notes")],
            TerminalDisposition.DEFER_ESCALATE,
        )

    prefix: list[Action] = []
    if state is LatentTaskState.AMBIGUOUS_RESOLVABLE:
        prefix.append(AskClarification(question_key=_AMBIGUOUS_DIM, question_text="dedup policy?"))

    target_data, _ = apply_repairs(
        task.corrupted_data, task.corrupted_card, list(task.nodes), task.contract
    )
    prefix.extend(plan_rewrite(task.corrupted_data, target_data))
    prefix.append(ToolAction(tool_name="sft_chat.regenerate_card"))
    return prefix, TerminalDisposition.COMMIT


def _terminal(disposition: TerminalDisposition, staged_root: str) -> TerminalAction:
    unresolved = (
        ("dedup policy unresolved",) if disposition is TerminalDisposition.PARTIAL_HANDOFF else ()
    )
    return TerminalAction(
        claim_bundle=ClaimBundle(
            disposition=disposition,
            artifact_root_hash=staged_root,
            confidence=0.95,
            unresolved=unresolved,
        )
    )
