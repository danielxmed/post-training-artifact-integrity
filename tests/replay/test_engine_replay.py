"""Deterministic replay of episodes through the engine."""

import pytest

from ptaie.kernel.actions import Action, TerminalAction, ToolAction
from ptaie.kernel.claims import ClaimBundle
from ptaie.kernel.observation import Observation
from ptaie.kernel.plugin import PluginRegistry
from ptaie.kernel.replay import ReplayHarness
from ptaie.kernel.rewards import TerminalDisposition
from ptaie.plugins.sft_chat.plugin import SftChatPlugin
from ptaie.version import ENV_VERSION

pytestmark = pytest.mark.replay


class InspectThenCommit:
    """List files, read the data, then commit the unchanged staged root."""

    def __init__(self) -> None:
        self._step = 0

    def act(self, observation: Observation) -> Action:
        self._step += 1
        if self._step == 1:
            return ToolAction(tool_name="kernel.list_files")
        if self._step == 2:
            return ToolAction(tool_name="kernel.read_file", arguments={"path": "data/train.jsonl"})
        if self._step == 3:
            return ToolAction(tool_name="kernel.run_visible_validators")
        return TerminalAction(
            claim_bundle=ClaimBundle(
                disposition=TerminalDisposition.COMMIT,
                artifact_root_hash=observation.staged_root,
                confidence=0.5,
            )
        )


def _harness() -> ReplayHarness:
    from tests.support.finalizer import StubFinalizer

    registry = PluginRegistry()
    registry.register(SftChatPlugin())
    return ReplayHarness(registry, StubFinalizer())


def test_record_then_replay_is_identical() -> None:
    harness = _harness()
    trace, result = harness.record(
        InspectThenCommit(), task_seed=11, artifact_class="sft_chat", env_version=ENV_VERSION
    )
    harness.assert_identical(trace, result)


def test_two_recordings_match() -> None:
    harness = _harness()
    _, a = harness.record(
        InspectThenCommit(), task_seed=12, artifact_class="sft_chat", env_version=ENV_VERSION
    )
    _, b = harness.record(
        InspectThenCommit(), task_seed=12, artifact_class="sft_chat", env_version=ENV_VERSION
    )
    assert a == b


def test_replay_across_many_seeds() -> None:
    harness = _harness()
    for seed in range(40):
        trace, result = harness.record(
            InspectThenCommit(), task_seed=seed, artifact_class="sft_chat", env_version=ENV_VERSION
        )
        harness.assert_identical(trace, result)


def test_env_version_mismatch_changes_result() -> None:
    harness = _harness()
    _, a = harness.record(
        InspectThenCommit(), task_seed=13, artifact_class="sft_chat", env_version=ENV_VERSION
    )
    _, b = harness.record(
        InspectThenCommit(), task_seed=13, artifact_class="sft_chat", env_version="9.9.9"
    )
    # different env_version => different episode identity (audit hash differs)
    assert a.audit_head_hash != b.audit_head_hash


def test_replay_rejects_a_cross_version_trace() -> None:
    import pytest as _pytest

    from ptaie.kernel.actions import ActionTrace, ToolAction
    from ptaie.kernel.errors import ReplayMismatchError

    harness = _harness()
    stale = ActionTrace(
        env_version="0.0.0-old",
        task_seed=1,
        artifact_class="sft_chat",
        actions=(ToolAction(tool_name="kernel.list_files"),),
    )
    with _pytest.raises(ReplayMismatchError):
        harness.replay(stale)
