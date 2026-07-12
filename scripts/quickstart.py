"""Generate one task, run the oracle policy, and print the outcome.

    python scripts/quickstart.py [SEED]

A minimal end-to-end smoke of the environment, used by the README.
"""

import sys

from ptaie.kernel.engine import PtaieEnv
from ptaie.kernel.plugin import PluginRegistry
from ptaie.plugins.sft_chat.finalize import SftChatFinalizer
from ptaie.plugins.sft_chat.plugin import SftChatPlugin
from ptaie.plugins.sft_chat.taskgen import generate_sft_task
from ptaie.policies.oracle import OraclePolicy
from ptaie.version import ENV_VERSION


def main(argv: list[str]) -> int:
    seed = int(argv[0]) if argv else 0
    task = generate_sft_task(seed, ENV_VERSION)

    registry = PluginRegistry()
    registry.register(SftChatPlugin())
    env = PtaieEnv(registry, SftChatFinalizer())

    observation = env.reset(task_seed=seed, artifact_class="sft_chat", env_version=ENV_VERSION)
    policy = OraclePolicy(task)
    steps = 0
    result = None
    while steps < 100:
        result = env.step(policy.act(observation))
        observation = result.observation
        steps += 1
        if result.terminated or result.truncated:
            break
    if result is None:
        raise RuntimeError("episode produced no step result")

    print(f"seed              {seed}")  # noqa: T201
    print(f"latent state      {task.latent_state.value}  (hidden — shown here for the demo)")  # noqa: T201
    print(f"terminal code     {result.terminal_code.value if result.terminal_code else None}")  # noqa: T201
    print(f"outcome class     {result.metrics.get('outcome_class')}")  # noqa: T201
    print(f"reward vector     {result.reward.model_dump()}")  # noqa: T201
    print(f"constraint        {result.constraint.model_dump()}")  # noqa: T201
    print(f"steps             {steps}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
