"""Deterministic replay: record a policy's trace, replay it, compare.

The determinism invariant is literally ``replay(trace) == recorded_result``,
including the audit head hash (which transitively pins every event payload).
An ``env_version`` mismatch is a hard error, never a silent best-effort.
"""

from typing import Protocol

from pydantic import BaseModel, ConfigDict

from ptaie.kernel.actions import Action, ActionTrace
from ptaie.kernel.engine import PtaieEnv, StepResult
from ptaie.kernel.errors import ReplayMismatchError
from ptaie.kernel.observation import Observation
from ptaie.kernel.plugin import PluginRegistry
from ptaie.kernel.rewards import ConstraintVector, RewardVector, TerminalCode
from ptaie.kernel.verification import Finalizer
from ptaie.version import ENV_VERSION


class Policy(Protocol):
    def act(self, observation: Observation) -> Action: ...


class ReplayResult(BaseModel):
    """The deterministic signature of an episode outcome."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    env_version: str
    task_seed: int
    artifact_class: str
    terminal_code: TerminalCode | None
    reward_totals: RewardVector
    constraint: ConstraintVector
    audit_head_hash: str
    staged_root: str
    steps: int


def _accumulate(rewards: list[RewardVector]) -> RewardVector:
    total = RewardVector.zero()
    fields = RewardVector.model_fields
    values = {name: sum(getattr(r, name) for r in rewards) for name in fields}
    return total.model_copy(update=values)


class ReplayHarness:
    """Record and replay episodes for determinism verification."""

    __slots__ = ("_finalizer", "_plugins")

    def __init__(self, plugins: PluginRegistry, finalizer: Finalizer) -> None:
        self._plugins = plugins
        self._finalizer = finalizer

    def record(
        self,
        policy: Policy,
        *,
        task_seed: int,
        artifact_class: str,
        env_version: str,
        max_steps: int = 1000,
    ) -> tuple[ActionTrace, ReplayResult]:
        env = PtaieEnv(self._plugins, self._finalizer)
        observation = env.reset(
            task_seed=task_seed, artifact_class=artifact_class, env_version=env_version
        )
        actions: list[Action] = []
        rewards: list[RewardVector] = []
        result: StepResult | None = None
        for _ in range(max_steps):
            action = policy.act(observation)
            actions.append(action)
            result = env.step(action)
            rewards.append(result.reward)
            observation = result.observation
            if result.terminated or result.truncated:
                break
        if result is None:
            raise ReplayMismatchError("steps", ">0", 0)
        trace = ActionTrace(
            env_version=env_version,
            task_seed=task_seed,
            artifact_class=artifact_class,
            actions=tuple(actions),
        )
        outcome = self._result(
            env, result, rewards, task_seed, artifact_class, env_version, len(actions)
        )
        return trace, outcome

    def replay(self, trace: ActionTrace) -> ReplayResult:
        if trace.env_version != ENV_VERSION:
            raise ReplayMismatchError("env_version", ENV_VERSION, trace.env_version)
        env = PtaieEnv(self._plugins, self._finalizer)
        env.reset(
            task_seed=trace.task_seed,
            artifact_class=trace.artifact_class,
            env_version=trace.env_version,
        )
        rewards: list[RewardVector] = []
        result: StepResult | None = None
        for action in trace.actions:
            result = env.step(action)
            rewards.append(result.reward)
            if result.terminated or result.truncated:
                break
        if result is None:
            raise ReplayMismatchError("steps", ">0", 0)
        return self._result(
            env,
            result,
            rewards,
            trace.task_seed,
            trace.artifact_class,
            trace.env_version,
            len(rewards),
        )

    def assert_identical(self, trace: ActionTrace, expected: ReplayResult) -> None:
        actual = self.replay(trace)
        for field in ReplayResult.model_fields:
            if getattr(actual, field) != getattr(expected, field):
                raise ReplayMismatchError(field, getattr(expected, field), getattr(actual, field))

    def _result(
        self,
        env: PtaieEnv,
        final: StepResult,
        rewards: list[RewardVector],
        task_seed: int,
        artifact_class: str,
        env_version: str,
        steps: int,
    ) -> ReplayResult:
        return ReplayResult(
            env_version=env_version,
            task_seed=task_seed,
            artifact_class=artifact_class,
            terminal_code=final.terminal_code,
            reward_totals=_accumulate(rewards),
            constraint=final.constraint,
            audit_head_hash=env.engine.audit_head_hash(),
            staged_root=final.observation.staged_root,
            steps=steps,
        )
