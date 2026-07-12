"""The Milestone 1 go/no-go: the oracle reaches the correct disposition on
every generated task, and the degenerate baselines are measurably worse.

This is the Phase-0 gate at vertical-slice scale — if it fails, the generator,
verifiers, or scorer disagree about what "correct" means.
"""

from collections import Counter
from collections.abc import Callable

import pytest
from tests.support.rollout import run_episode

from ptaie.kernel.engine import StepResult
from ptaie.kernel.replay import Policy
from ptaie.kernel.rewards import OutcomeClass
from ptaie.kernel.scoring import is_correct_outcome
from ptaie.kernel.task import LatentTaskState
from ptaie.plugins.sft_chat.taskgen import SftTask, generate_sft_task
from ptaie.policies.baselines import AlwaysAbstainPolicy, AlwaysRepairPolicy
from ptaie.policies.oracle import OraclePolicy
from ptaie.version import ENV_VERSION

pytestmark = pytest.mark.acceptance

_CI_SWEEP = 300

_CORRECT_OUTCOME = {
    LatentTaskState.REPAIRABLE: OutcomeClass.CORRECT_COMMIT,
    LatentTaskState.ALREADY_CORRECT: OutcomeClass.CORRECT_NOOP,
    LatentTaskState.AMBIGUOUS_RESOLVABLE: OutcomeClass.CORRECT_COMMIT,
    LatentTaskState.UNDERDETERMINED: OutcomeClass.JUSTIFIED_ABSTAIN,
    LatentTaskState.INCONSISTENT: OutcomeClass.JUSTIFIED_DEFER,
    LatentTaskState.ADVERSARIAL_TRAP: OutcomeClass.CORRECT_COMMIT,
}


def _outcome(result: StepResult) -> OutcomeClass:
    value = result.metrics["outcome_class"]
    assert isinstance(value, str)
    return OutcomeClass(value)


def _sweep(sweep: int) -> None:
    mismatches: list[tuple[int, str, str]] = []
    for seed in range(sweep):
        task = generate_sft_task(seed, ENV_VERSION)
        result = run_episode(OraclePolicy(task), task_seed=seed)
        want = _CORRECT_OUTCOME[task.latent_state]
        got = _outcome(result)
        if got is not want or result.constraint.any_violation:
            mismatches.append((seed, task.latent_state.value, got.value))
    assert not mismatches, f"oracle failed on: {mismatches[:15]}"


def test_oracle_reaches_correct_disposition_ci_sweep() -> None:
    _sweep(_CI_SWEEP)


@pytest.mark.slow
def test_oracle_go_no_go_large_sweep() -> None:
    _sweep(1000)


def _correct_rate(make_policy: Callable[[SftTask], Policy], sweep: int = _CI_SWEEP) -> float:
    correct = 0
    for seed in range(sweep):
        task = generate_sft_task(seed, ENV_VERSION)
        result = run_episode(make_policy(task), task_seed=seed)
        if is_correct_outcome(_outcome(result)):
            correct += 1
    return correct / sweep


def test_baselines_are_measurably_worse_than_oracle() -> None:
    oracle_rate = _correct_rate(OraclePolicy)
    repair_rate = _correct_rate(lambda _task: AlwaysRepairPolicy())
    abstain_rate = _correct_rate(lambda _task: AlwaysAbstainPolicy())
    assert oracle_rate == 1.0
    # edit-first and blanket-refusal both lose badly
    assert repair_rate < 0.35, repair_rate
    assert abstain_rate < 0.35, abstain_rate


def test_always_repair_false_commits_on_repairable_and_noop() -> None:
    # a concrete demonstration that aggressive editing harms already-correct
    # data and never fixes the real defect
    outcomes: Counter[str] = Counter()
    for seed in range(_CI_SWEEP):
        task = generate_sft_task(seed, ENV_VERSION)
        if task.latent_state not in (
            LatentTaskState.REPAIRABLE,
            LatentTaskState.ALREADY_CORRECT,
        ):
            continue
        result = run_episode(AlwaysRepairPolicy(), task_seed=seed)
        outcomes[_outcome(result).value] += 1
    assert outcomes["false_commit"] > 0
