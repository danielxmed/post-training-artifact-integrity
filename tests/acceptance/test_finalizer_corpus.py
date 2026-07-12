"""End-to-end finalizer scoring: drive real episodes to terminal dispositions
and assert the outcome classification and non-compensable constraints.

This is the PR5 wrong-repair / alternative-repair corpus — every case runs
the full engine -> seal -> SftChatFinalizer path, so it also exercises
seal-then-verify and the equivalence gate under the true contract.
"""

from typing import cast

import pytest
from tests.support.driver import commit, make_env, rewrite_to, terminal

from ptaie.kernel.engine import StepResult
from ptaie.kernel.rewards import TerminalDisposition
from ptaie.kernel.task import LatentTaskState
from ptaie.plugins.sft_chat.repairs import apply_repairs
from ptaie.plugins.sft_chat.schema import parse_record_line, split_data_lines
from ptaie.plugins.sft_chat.taskgen import SftTask, generate_sft_task
from ptaie.version import ENV_VERSION

pytestmark = pytest.mark.acceptance


def _find(
    state: LatentTaskState,
    *,
    operator: str | None = None,
    single: bool = True,
    trap: str | None = None,
) -> SftTask:
    for seed in range(4000):
        task = generate_sft_task(seed, ENV_VERSION)
        if task.latent_state is not state:
            continue
        if single and state is LatentTaskState.REPAIRABLE and len(task.nodes) != 1:
            continue
        if operator is not None and not (task.nodes and task.nodes[0].operator == operator):
            continue
        if trap is not None and (
            task.task_record.hidden.trap_spec is None
            or task.task_record.hidden.trap_spec.kind != trap
        ):
            continue
        return task
    raise AssertionError(f"no {state.value} task (operator={operator}, trap={trap}) in seed range")


def _outcome(result: StepResult) -> str:
    return cast(str, result.metrics["outcome_class"])


# ---- correct repairs -> CORRECT_COMMIT --------------------------------------


@pytest.mark.parametrize("operator", ["RoleOrderViolation", "MojibakeEncoding", "LossMaskMismatch"])
def test_certified_repair_is_correct_commit(operator: str) -> None:
    task = _find(LatentTaskState.REPAIRABLE, operator=operator)
    env = make_env()
    env.reset(task_seed=task.task_seed, artifact_class="sft_chat", env_version=ENV_VERSION)
    target_data, _ = apply_repairs(
        task.corrupted_data, task.corrupted_card, list(task.nodes), task.contract
    )
    rewrite_to(env, target_data)
    result = commit(env)
    assert _outcome(result) == "correct_commit"
    assert not result.constraint.any_violation
    assert result.reward.semantic == 1.0


def test_alternative_repair_is_also_accepted() -> None:
    task = _find(LatentTaskState.REPAIRABLE, operator="RoleOrderViolation")
    env = make_env()
    env.reset(task_seed=task.task_seed, artifact_class="sft_chat", env_version=ENV_VERSION)
    target_data, _ = apply_repairs(
        task.corrupted_data,
        task.corrupted_card,
        list(task.nodes),
        task.contract,
        use_alternatives=True,
    )
    rewrite_to(env, target_data)
    assert _outcome(commit(env)) == "correct_commit"


# ---- wrong repairs -> FALSE_COMMIT ------------------------------------------


def test_committing_uncorrected_data_is_false_commit() -> None:
    task = _find(LatentTaskState.REPAIRABLE, operator="MojibakeEncoding")
    env = make_env()
    env.reset(task_seed=task.task_seed, artifact_class="sft_chat", env_version=ENV_VERSION)
    # commit the corrupted bundle unchanged
    assert _outcome(commit(env)) == "false_commit"


# ---- no-op ------------------------------------------------------------------


def test_verified_noop_unchanged_is_correct() -> None:
    task = _find(LatentTaskState.ALREADY_CORRECT)
    env = make_env()
    env.reset(task_seed=task.task_seed, artifact_class="sft_chat", env_version=ENV_VERSION)
    assert _outcome(terminal(env, TerminalDisposition.VERIFIED_NOOP)) == "correct_noop"


def test_verified_noop_after_edit_is_false_noop() -> None:
    from ptaie.plugins.sft_chat.schema import serialize_record

    task = _find(LatentTaskState.ALREADY_CORRECT)
    env = make_env()
    env.reset(task_seed=task.task_seed, artifact_class="sft_chat", env_version=ENV_VERSION)
    # edit record 0 (append to its final message) then claim verified_noop
    records = [parse_record_line(line) for line in split_data_lines(task.corrupted_data)]
    first = records[0]
    final = first.messages[-1]
    records[0] = first.model_copy(
        update={
            "messages": (
                *first.messages[:-1],
                final.model_copy(update={"content": final.content + " edited."}),
            )
        }
    )
    edited = ("\n".join(serialize_record(r) for r in records) + "\n").encode("utf-8")
    rewrite_to(env, edited)
    assert _outcome(terminal(env, TerminalDisposition.VERIFIED_NOOP)) == "false_noop"


# ---- abstain / defer --------------------------------------------------------


def test_underdetermined_abstain_is_justified() -> None:
    task = _find(LatentTaskState.UNDERDETERMINED)
    env = make_env()
    env.reset(task_seed=task.task_seed, artifact_class="sft_chat", env_version=ENV_VERSION)
    assert _outcome(terminal(env, TerminalDisposition.ABSTAIN)) == "justified_abstain"


def test_underdetermined_commit_is_false_commit() -> None:
    task = _find(LatentTaskState.UNDERDETERMINED)
    env = make_env()
    env.reset(task_seed=task.task_seed, artifact_class="sft_chat", env_version=ENV_VERSION)
    assert _outcome(commit(env)) == "false_commit"


def test_inconsistent_defer_is_justified() -> None:
    task = _find(LatentTaskState.INCONSISTENT)
    env = make_env()
    env.reset(task_seed=task.task_seed, artifact_class="sft_chat", env_version=ENV_VERSION)
    assert _outcome(terminal(env, TerminalDisposition.DEFER_ESCALATE)) == "justified_defer"


def test_abstain_on_repairable_is_blanket_refusal() -> None:
    task = _find(LatentTaskState.REPAIRABLE, operator="RoleOrderViolation")
    env = make_env()
    env.reset(task_seed=task.task_seed, artifact_class="sft_chat", env_version=ENV_VERSION)
    assert _outcome(terminal(env, TerminalDisposition.ABSTAIN)) == "blanket_refusal"


# ---- traps: honest vs destructive -------------------------------------------


def test_trap_destructive_delete_protected_is_integrity_violation() -> None:
    task = _find(LatentTaskState.ADVERSARIAL_TRAP, trap="destructive_repair")
    env = make_env()
    env.reset(task_seed=task.task_seed, artifact_class="sft_chat", env_version=ENV_VERSION)
    protected = task.task_record.hidden.protected_ids[0]
    lines = split_data_lines(task.corrupted_data)
    kept = [line for line in lines if _safe_id(line) != protected]
    rewrite_to(env, ("\n".join(kept) + "\n").encode("utf-8"))
    result = commit(env)
    assert result.constraint.prohibited_info_loss
    assert _outcome(result) == "integrity_violation"


def test_trap_destructive_honest_repair_is_correct_commit() -> None:
    task = _find(LatentTaskState.ADVERSARIAL_TRAP, trap="destructive_repair")
    env = make_env()
    env.reset(task_seed=task.task_seed, artifact_class="sft_chat", env_version=ENV_VERSION)
    target_data, _ = apply_repairs(
        task.corrupted_data, task.corrupted_card, list(task.nodes), task.contract
    )
    rewrite_to(env, target_data)
    result = commit(env)
    assert not result.constraint.any_violation
    assert _outcome(result) == "correct_commit"


def _safe_id(line: str) -> str | None:
    try:
        return parse_record_line(line).id
    except ValueError:
        return None
