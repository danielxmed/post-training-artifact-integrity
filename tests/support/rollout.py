"""Run a scripted policy through a fresh episode and return the final step."""

from ptaie.kernel.engine import PtaieEnv, StepResult
from ptaie.kernel.plugin import PluginRegistry
from ptaie.kernel.replay import Policy
from ptaie.plugins.sft_chat.finalize import SftChatFinalizer
from ptaie.plugins.sft_chat.plugin import SftChatPlugin
from ptaie.version import ENV_VERSION


def make_env() -> PtaieEnv:
    registry = PluginRegistry()
    registry.register(SftChatPlugin())
    return PtaieEnv(registry, SftChatFinalizer())


def run_episode(policy: Policy, *, task_seed: int, max_steps: int = 100) -> StepResult:
    env = make_env()
    observation = env.reset(task_seed=task_seed, artifact_class="sft_chat", env_version=ENV_VERSION)
    result: StepResult | None = None
    for _ in range(max_steps):
        result = env.step(policy.act(observation))
        observation = result.observation
        if result.terminated or result.truncated:
            break
    assert result is not None
    return result
