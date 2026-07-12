"""Generator determinism and latent-state mixture."""

from collections import Counter

from ptaie.kernel.task import LatentTaskState
from ptaie.plugins.sft_chat.taskgen import generate_sft_task, generate_task
from ptaie.version import ENV_VERSION


def test_generation_is_deterministic() -> None:
    a = generate_sft_task(1234, ENV_VERSION)
    b = generate_sft_task(1234, ENV_VERSION)
    assert a.corrupted_data == b.corrupted_data
    assert a.clean_data == b.clean_data
    assert a.corrupted_card == b.corrupted_card
    assert a.latent_state == b.latent_state
    assert a.task_record.model_dump(mode="json") == b.task_record.model_dump(mode="json")


def test_task_id_is_opaque_and_env_version_sensitive() -> None:
    task = generate_sft_task(1, ENV_VERSION)
    assert task.task_record.public.task_id.startswith("task-")
    other = generate_sft_task(1, "9.9.9")
    assert other.task_record.public.task_id != task.task_record.public.task_id


def test_env_version_changes_generation() -> None:
    a = generate_sft_task(7, ENV_VERSION)
    b = generate_sft_task(7, "9.9.9")
    assert a.corrupted_data != b.corrupted_data or a.latent_state != b.latent_state


def test_kernel_task_record_is_valid_and_serializable() -> None:
    record = generate_task(42, ENV_VERSION)
    dumped = record.model_dump(mode="json")
    assert dumped["public"]["artifact_class"] == "sft_chat"
    assert dumped["public"]["initial_manifest"] == record.public.initial_manifest


def test_latent_state_mixture_within_tolerance() -> None:
    counts: Counter[LatentTaskState] = Counter(
        generate_sft_task(seed, ENV_VERSION).latent_state for seed in range(1000)
    )
    fractions = {state: counts[state] / 1000 for state in LatentTaskState}
    targets = {
        LatentTaskState.REPAIRABLE: 0.55,
        LatentTaskState.ALREADY_CORRECT: 0.10,
        LatentTaskState.AMBIGUOUS_RESOLVABLE: 0.15,
        LatentTaskState.UNDERDETERMINED: 0.10,
        LatentTaskState.INCONSISTENT: 0.05,
        LatentTaskState.ADVERSARIAL_TRAP: 0.05,
    }
    for state, target in targets.items():
        assert abs(fractions[state] - target) < 0.04, f"{state}={fractions[state]:.3f}"


def test_all_states_represented() -> None:
    seen = {generate_sft_task(seed, ENV_VERSION).latent_state for seed in range(200)}
    assert seen == set(LatentTaskState)
