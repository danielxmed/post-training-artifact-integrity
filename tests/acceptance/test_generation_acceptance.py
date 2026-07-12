"""Generation acceptance over a seed sweep (NORTH_STAR 5.2).

Marked ``acceptance`` (excluded from the fast CI test job, run in the
replay-acceptance job). Every generated task in the sweep must pass the
acceptance checks, and the rejected-seed set must be byte-stable.
"""

from collections import Counter

import pytest

from ptaie.kernel.task import LatentTaskState
from ptaie.plugins.sft_chat.acceptance import run_acceptance
from ptaie.plugins.sft_chat.taskgen import generate_sft_task
from ptaie.version import ENV_VERSION

pytestmark = pytest.mark.acceptance

_SWEEP = 500


def _rejects(sweep: int) -> list[tuple[int, list[str]]]:
    rejects: list[tuple[int, list[str]]] = []
    for seed in range(sweep):
        report = run_acceptance(generate_sft_task(seed, ENV_VERSION))
        if not report.ok:
            rejects.append((seed, report.failures))
    return rejects


def test_all_generated_tasks_pass_acceptance() -> None:
    rejects = _rejects(_SWEEP)
    assert not rejects, f"rejected seeds: {rejects[:10]}"


def test_every_latent_state_is_exercised_in_the_sweep() -> None:
    counts: Counter[LatentTaskState] = Counter(
        generate_sft_task(seed, ENV_VERSION).latent_state for seed in range(_SWEEP)
    )
    assert set(counts) == set(LatentTaskState)


def test_rejected_seed_set_is_byte_stable() -> None:
    assert _rejects(_SWEEP) == _rejects(_SWEEP)


@pytest.mark.slow
def test_rejection_rate_under_two_percent_large_sweep() -> None:
    sweep = 2000
    rejects = _rejects(sweep)
    assert len(rejects) / sweep < 0.02, f"{len(rejects)} rejects"
