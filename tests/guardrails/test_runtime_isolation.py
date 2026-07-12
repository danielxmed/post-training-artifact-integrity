"""Runtime isolation guardrails: the tool capability object cannot reach
hidden state, and hidden latent state never appears in observations."""

import json

from tests.support.finalizer import StubFinalizer

from ptaie.kernel.actions import AskClarification, ToolAction
from ptaie.kernel.engine import PtaieEnv
from ptaie.kernel.plugin import PluginRegistry
from ptaie.kernel.tools.base import ToolContext
from ptaie.plugins.sft_chat.plugin import SftChatPlugin
from ptaie.plugins.sft_chat.taskgen import generate_sft_task
from ptaie.version import ENV_VERSION


def test_toolcontext_has_no_reference_to_task_vault_or_engine() -> None:
    slots = set(ToolContext.__slots__)
    for forbidden in ("task", "vault", "engine", "hidden", "audit", "contract", "finalizer"):
        assert not any(forbidden in slot for slot in slots), f"ToolContext exposes {forbidden!r}"
    # exactly the intended capabilities
    assert slots == {"_workspace", "_can_mutate", "rng", "budget", "constraints"}


def _env() -> PtaieEnv:
    registry = PluginRegistry()
    registry.register(SftChatPlugin())
    return PtaieEnv(registry, StubFinalizer())


def test_latent_state_never_appears_in_observations() -> None:
    for seed in range(120):
        task = generate_sft_task(seed, ENV_VERSION)
        hidden = task.task_record.hidden
        # tokens that must never surface in what the agent observes
        forbidden = {hidden.latent_state.value}
        for node in hidden.defect_dag:
            forbidden.add(node.defect_id)
        env = _env()
        observations = [
            env.reset(task_seed=seed, artifact_class="sft_chat", env_version=ENV_VERSION)
        ]
        # heavy inspection then a couple of clarifications
        for action in (
            ToolAction(tool_name="kernel.list_files"),
            ToolAction(tool_name="kernel.read_file", arguments={"path": "data/train.jsonl"}),
            ToolAction(tool_name="kernel.read_file", arguments={"path": "dataset_card.json"}),
            ToolAction(tool_name="kernel.read_file", arguments={"path": "checks/format_check.py"}),
            ToolAction(tool_name="kernel.run_visible_validators"),
            AskClarification(question_key="dedup_policy", question_text="?"),
        ):
            observations.append(env.step(action).observation)
        blob = "\n".join(json.dumps(obs.model_dump(mode="json")) for obs in observations)
        for token in forbidden:
            assert token not in blob, f"seed {seed} leaked {token!r}"
