"""Property tests: every operator fires exactly the checks it declares, and
its certified repair restores the invariants, across generated tasks."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ptaie.kernel.task import LatentTaskState
from ptaie.kernel.verification import CheckStatus
from ptaie.plugins.sft_chat.repairs import apply_repairs
from ptaie.plugins.sft_chat.taskgen import generate_sft_task
from ptaie.plugins.sft_chat.verifiers.equivalence import evaluate_commit
from ptaie.plugins.sft_chat.verifiers.layers import run_all_layers
from ptaie.plugins.sft_chat.view import build_view
from ptaie.version import ENV_VERSION


@pytest.mark.property
@given(seed=st.integers(min_value=0, max_value=5000))
def test_repairable_defects_fire_and_certified_repair_accepts(seed: int) -> None:
    task = generate_sft_task(seed, ENV_VERSION)
    if task.latent_state is not LatentTaskState.REPAIRABLE or not task.nodes:
        return

    # every unmasked expected failure fires; masked ones block-or-fail
    results = {
        r.check_id: r.status
        for r in run_all_layers(
            build_view(task.corrupted_data, task.corrupted_card),
            task.contract,
            protected_ids=task.protected_ids,
            count_floor=task.count_floor,
        )
    }
    masked = {m for node in task.nodes for m in node.masks}
    for node in task.nodes:
        blocked_ok = node.defect_id in masked
        for check_id in node.expected_failures:
            status = results[check_id]
            if blocked_ok:
                assert status in (CheckStatus.BLOCKED, CheckStatus.FAILED)
            else:
                assert status is CheckStatus.FAILED, f"{node.operator}:{check_id}={status}"

    # certified repair yields an accepted commit
    data, card = apply_repairs(
        task.corrupted_data, task.corrupted_card, list(task.nodes), task.contract
    )
    verdict = evaluate_commit(
        data,
        card,
        task.corrupted_data,
        task.corrupted_card,
        task.contract,
        task.nodes,
        protected_ids=task.protected_ids,
        count_floor=task.count_floor,
    )
    assert verdict.accepted, f"seed={seed} c_z={verdict.c_z_pass} h_z={verdict.h_z_violations}"


@pytest.mark.property
@given(seed=st.integers(min_value=0, max_value=5000))
def test_corrupted_bundle_is_not_already_accepted(seed: int) -> None:
    """A repairable task's *corrupted* bundle must not already pass — else the
    defect was not really injected."""
    task = generate_sft_task(seed, ENV_VERSION)
    if task.latent_state is not LatentTaskState.REPAIRABLE or not task.nodes:
        return
    verdict = evaluate_commit(
        task.corrupted_data,
        task.corrupted_card,
        task.corrupted_data,
        task.corrupted_card,
        task.contract,
        task.nodes,
        protected_ids=task.protected_ids,
        count_floor=task.count_floor,
    )
    assert not verdict.accepted
