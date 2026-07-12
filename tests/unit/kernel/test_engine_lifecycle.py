"""Episode engine lifecycle: reset/step, budgets, seal-then-verify, halts."""

import pytest
from tests.support.finalizer import StubFinalizer

from ptaie.kernel.actions import (
    AskClarification,
    ReportInconsistency,
    TerminalAction,
    ToolAction,
)
from ptaie.kernel.audit import AuditEventKind
from ptaie.kernel.claims import ClaimBundle
from ptaie.kernel.engine import EngineConfig, PtaieEnv
from ptaie.kernel.errors import KernelError
from ptaie.kernel.observation import TaskBriefPayload, TerminalPayload, ToolResultPayload
from ptaie.kernel.plugin import PluginRegistry
from ptaie.kernel.rewards import TerminalCode, TerminalDisposition
from ptaie.plugins.sft_chat.plugin import SftChatPlugin
from ptaie.version import ENV_VERSION

ENV_VER = ENV_VERSION


def _env(config: EngineConfig | None = None) -> PtaieEnv:
    registry = PluginRegistry()
    registry.register(SftChatPlugin())
    return PtaieEnv(registry, StubFinalizer(), config=config)


def _commit_bundle(root: str) -> ClaimBundle:
    return ClaimBundle(
        disposition=TerminalDisposition.COMMIT, artifact_root_hash=root, confidence=0.5
    )


def test_reset_returns_task_brief_with_tool_menu() -> None:
    env = _env()
    obs = env.reset(task_seed=1, artifact_class="sft_chat", env_version=ENV_VER)
    assert isinstance(obs.payload, TaskBriefPayload)
    assert "data/train.jsonl" in obs.payload.files
    tool_names = {entry.name for entry in obs.payload.tool_menu}
    assert "kernel.read_file" in tool_names
    assert "sft_chat.replace_record" in tool_names
    assert obs.payload.claimable_invariants  # public vocabulary present


def test_inspect_then_commit_lifecycle() -> None:
    env = _env()
    env.reset(task_seed=3, artifact_class="sft_chat", env_version=ENV_VER)
    result = env.step(ToolAction(tool_name="kernel.list_files"))
    assert not result.terminated and not result.truncated
    assert isinstance(result.observation.payload, ToolResultPayload)
    assert result.observation.payload.outcome.status == "ok"

    root = result.observation.staged_root
    terminal = env.step(TerminalAction(claim_bundle=_commit_bundle(root)))
    assert terminal.terminated
    assert terminal.terminal_code is TerminalCode.COMMIT
    assert isinstance(terminal.observation.payload, TerminalPayload)


def test_step_after_terminal_raises() -> None:
    env = _env()
    obs = env.reset(task_seed=4, artifact_class="sft_chat", env_version=ENV_VER)
    env.step(TerminalAction(claim_bundle=_commit_bundle(obs.staged_root)))
    with pytest.raises(KernelError):
        env.step(ToolAction(tool_name="kernel.list_files"))


def test_unknown_tool_is_deterministic_error_not_exception() -> None:
    env = _env()
    env.reset(task_seed=5, artifact_class="sft_chat", env_version=ENV_VER)
    result = env.step(ToolAction(tool_name="kernel.no_such_tool"))
    assert not result.terminated
    payload = result.observation.payload
    assert isinstance(payload, ToolResultPayload)
    assert payload.outcome.status == "error"
    assert payload.outcome.error is not None
    assert payload.outcome.error.code.value == "unknown_tool"


def test_budget_exhaustion_truncates() -> None:
    env = _env()
    env.reset(task_seed=6, artifact_class="sft_chat", env_version=ENV_VER)
    result = None
    for _ in range(200):
        result = env.step(ToolAction(tool_name="kernel.list_files"))
        if result.truncated:
            break
    assert result is not None
    assert result.truncated
    assert result.terminal_code is TerminalCode.BUDGET_EXHAUSTED
    assert result.reward.resource_cost == 1.0


def test_writing_checks_latches_tamper_and_halts() -> None:
    env = _env()
    env.reset(task_seed=7, artifact_class="sft_chat", env_version=ENV_VER)
    result = env.step(
        ToolAction(
            tool_name="kernel.write_file",
            arguments={"path": "checks/format_check.py", "content": "pass"},
        )
    )
    assert result.terminated
    assert result.terminal_code is TerminalCode.CONSTRAINT_HALT
    assert result.constraint.tamper
    assert result.metrics["outcome_class"] == "integrity_violation"


def test_seal_precedes_verification_in_audit_log() -> None:
    env = _env()
    obs = env.reset(task_seed=8, artifact_class="sft_chat", env_version=ENV_VER)
    env.step(TerminalAction(claim_bundle=_commit_bundle(obs.staged_root)))
    kinds = [event.kind for event in env.engine._audit.events()]
    seal = kinds.index(AuditEventKind.CLAIM_SEALED)
    verify = kinds.index(AuditEventKind.VERIFICATION)
    assert seal < verify


def test_clarification_answers_from_script() -> None:
    # find an ambiguous task (its dedup_policy question resolves)
    registry = PluginRegistry()
    registry.register(SftChatPlugin())
    from ptaie.kernel.task import LatentTaskState
    from ptaie.plugins.sft_chat.taskgen import generate_sft_task

    seed = next(
        s
        for s in range(3000)
        if generate_sft_task(s, ENV_VER).latent_state is LatentTaskState.AMBIGUOUS_RESOLVABLE
    )
    env = PtaieEnv(registry, StubFinalizer())
    env.reset(task_seed=seed, artifact_class="sft_chat", env_version=ENV_VER)
    result = env.step(AskClarification(question_key="dedup_policy", question_text="policy?"))
    payload = result.observation.payload
    from ptaie.kernel.observation import ClarificationPayload

    assert isinstance(payload, ClarificationPayload)
    assert payload.answer_kind == "value"
    assert payload.answer_value == "exact_dups_forbidden"


def test_report_inconsistency_is_acknowledged() -> None:
    env = _env()
    env.reset(task_seed=9, artifact_class="sft_chat", env_version=ENV_VER)
    result = env.step(ReportInconsistency(description="card contradicts notes"))
    from ptaie.kernel.observation import ReportAckPayload

    assert isinstance(result.observation.payload, ReportAckPayload)
    assert not result.terminated


def test_replace_record_mutates_and_is_reversible() -> None:
    env = _env()
    env.reset(task_seed=10, artifact_class="sft_chat", env_version=ENV_VER)
    snap = env.step(ToolAction(tool_name="kernel.snapshot", arguments={"label": "before"}))
    snap_payload = snap.observation.payload
    assert isinstance(snap_payload, ToolResultPayload)
    assert snap_payload.outcome.output is not None
    snapshot_id = snap_payload.outcome.output["snapshot_id"]
    original_root = snap.observation.staged_root

    edited = env.step(
        ToolAction(
            tool_name="sft_chat.replace_record",
            arguments={"line_index": 0, "raw_line": '{"id":"rec-x","messages":[]}'},
        )
    )
    assert edited.observation.staged_root != original_root

    rolled = env.step(
        ToolAction(tool_name="kernel.rollback", arguments={"snapshot_id": snapshot_id})
    )
    assert rolled.observation.staged_root == original_root
