"""The episode engine: reset / step lifecycle with seal-then-verify.

Determinism: the engine derives its rng from ``(task_seed, env_version)``,
uses ``event_index`` as the only clock, and holds no cross-episode state (a
fresh engine, blob store, and audit log per episode). Constraint flags latch
and are non-compensable; a latched flag ends the episode immediately with
``CONSTRAINT_HALT``. Budget exhaustion truncates with ``BUDGET_EXHAUSTED`` and
runs no hidden verification (no sealed bundle exists).

The seal-then-verify ordering is enforced by construction: a ``TerminalAction``
first calls ``AuditLog.seal_claim`` (which appends the ``CLAIM_SEALED`` event)
and only then invokes the injected finalizer.
"""

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from ptaie.kernel.actions import (
    Action,
    AskClarification,
    ReportInconsistency,
    TerminalAction,
    ToolAction,
)
from ptaie.kernel.audit import AuditEventKind, AuditLog
from ptaie.kernel.budgets import BudgetCharge, BudgetLedger
from ptaie.kernel.canonical import JsonValue
from ptaie.kernel.clarify import ClarificationOracle
from ptaie.kernel.errors import KernelError
from ptaie.kernel.ids import AuditRef
from ptaie.kernel.observation import (
    ClarificationPayload,
    Observation,
    ObservationPayload,
    ReportAckPayload,
    TaskBriefPayload,
    TerminalPayload,
    ToolResultPayload,
)
from ptaie.kernel.rewards import ConstraintVector, RewardVector, TerminalCode
from ptaie.kernel.rng import DerivedRng, derive_seed
from ptaie.kernel.store.blob import BlobStore
from ptaie.kernel.store.manifest import Manifest
from ptaie.kernel.store.workspace import Workspace, WorkspaceReadView
from ptaie.kernel.task import TaskRecord
from ptaie.kernel.tools.base import ConstraintSink, ToolContext
from ptaie.kernel.tools.builtin import builtin_tools
from ptaie.kernel.tools.registry import ToolRegistry
from ptaie.kernel.verification import Finalizer, VerificationReport

if TYPE_CHECKING:
    from ptaie.kernel.plugin import PluginRegistry

_NON_COMPENSABLE = ("tamper", "escape", "prohibited_info_loss", "irreversible", "hard_policy")


class StepResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    observation: Observation
    terminated: bool
    truncated: bool
    reward: RewardVector
    constraint: ConstraintVector
    terminal_code: TerminalCode | None
    audit_ref: AuditRef
    metrics: dict[str, JsonValue] = Field(default_factory=dict)


class EngineConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    inspect_charge: BudgetCharge = BudgetCharge(cost_units=1)
    communicate_charge: BudgetCharge = BudgetCharge(cost_units=1, questions=1)
    report_charge: BudgetCharge = BudgetCharge(cost_units=1)


class EpisodeEngine:
    """Single-use engine for one episode. Call ``reset`` once, then ``step``."""

    __slots__ = (
        "_asked",
        "_audit",
        "_config",
        "_constraint",
        "_env_version",
        "_finalizer",
        "_initial_manifest",
        "_ledger",
        "_oracle",
        "_phase",
        "_registry",
        "_report",
        "_rng",
        "_step_index",
        "_store",
        "_task",
        "_workspace",
    )

    def __init__(
        self,
        task: TaskRecord,
        *,
        store: BlobStore,
        manifest: Manifest,
        registry: ToolRegistry,
        oracle: ClarificationOracle,
        finalizer: Finalizer,
        env_version: str,
        config: EngineConfig | None = None,
    ) -> None:
        self._task = task
        self._store = store
        self._registry = registry
        self._oracle = oracle
        self._finalizer = finalizer
        self._env_version = env_version
        self._config = config or EngineConfig()
        episode_id = _episode_id(task.seed, env_version)
        self._audit = AuditLog(episode_id)
        self._ledger = BudgetLedger(task.public.budgets)
        self._constraint = ConstraintVector()
        self._workspace = Workspace(store, manifest)
        self._initial_manifest = manifest
        self._rng = DerivedRng(derive_seed(task.seed, "episode", env_version))
        self._step_index = 0
        self._phase = "created"
        self._asked: set[str] = set()
        self._report: VerificationReport | None = None

    @property
    def report(self) -> VerificationReport | None:
        """The hidden verification report (environment-side; set after a
        terminal COMMIT/NOOP/etc. — never observed by the agent)."""
        return self._report

    def audit_head_hash(self) -> str:
        return self._audit.head_hash()

    def reset(self) -> Observation:
        if self._phase != "created":
            raise KernelError("engine already reset (single-use)")
        self._phase = "running"
        self._audit.append(
            AuditEventKind.EPISODE_START,
            "environment",
            {"seed": self._task.seed, "env_version": self._env_version},
            step_index=self._step_index,
        )
        payload = TaskBriefPayload(
            brief=self._task.public.brief,
            files=self._workspace.manifest.paths(),
            tool_menu=self._registry.specs(),
            visible_validator_names=self._task.public.visible_validator_names,
            claimable_invariants=self._task.public.claimable_invariants,
            irreversible_boundaries=self._task.public.irreversible_boundaries,
        )
        return self._observe(payload)

    def step(self, action: Action) -> StepResult:
        if self._phase != "running":
            raise KernelError(f"step() called in phase {self._phase!r}")
        self._step_index += 1
        action_ref = self._audit.append(
            AuditEventKind.ACTION, "agent", _action_payload(action), step_index=self._step_index
        )

        if isinstance(action, TerminalAction):
            return self._finalize(action, action_ref)
        if isinstance(action, ToolAction):
            return self._run_tool(action, action_ref)
        if isinstance(action, AskClarification):
            return self._run_clarification(action, action_ref)
        if isinstance(action, ReportInconsistency):
            return self._run_report(action, action_ref)
        raise KernelError("unreachable: exhaustive action union")  # pragma: no cover

    # ------------------------------------------------------------------ tools
    def _run_tool(self, action: ToolAction, action_ref: AuditRef) -> StepResult:
        tool = self._registry.get(action.tool_name)
        charge = tool.spec.charge if tool is not None else self._config.inspect_charge
        exhausted = self._charge(charge)
        if exhausted is not None:
            return self._truncate(exhausted, action_ref)

        sink = ConstraintSink()
        can_mutate = tool is not None and tool.spec.transactional
        ctx = ToolContext(
            self._workspace,
            self._rng.substream(f"tool.{self._step_index}"),
            self._ledger.view(),
            sink,
            can_mutate=can_mutate,
        )
        outcome = self._registry.dispatch(action, ctx)
        result_ref = self._audit.append(
            AuditEventKind.TOOL_RESULT,
            "environment",
            {
                "tool": action.tool_name,
                "status": outcome.status,
                "staged_root": self._workspace.staged,
                # the inspected path (when the tool takes one), so evidence
                # assessment can tell an artifact inspection from a listing
                "target": _target_path(action),
            },
            step_index=self._step_index,
        )
        halt = self._latch(sink.flags())
        payload = ToolResultPayload(
            tool_name=action.tool_name, outcome=outcome, audit_ref=result_ref
        )
        if halt:
            return self._constraint_halt(payload, action_ref)
        return self._running(payload, action_ref)

    # ------------------------------------------------------------ communicate
    def _run_clarification(self, action: AskClarification, action_ref: AuditRef) -> StepResult:
        exhausted = self._charge(self._config.communicate_charge)
        if exhausted is not None:
            return self._truncate(exhausted, action_ref)
        response = self._oracle.answer(action.question_key, frozenset(self._asked))
        if action.question_key is not None:
            self._asked.add(action.question_key)
        ref = self._audit.append(
            AuditEventKind.CLARIFICATION,
            "environment",
            {"question_key": action.question_key, "answer_kind": response.answer_kind},
            step_index=self._step_index,
        )
        answer_kind = (
            response.answer_kind if response.answer_kind != "no_such_key" else "already_stated"
        )
        payload = ClarificationPayload(
            question_key=action.question_key,
            answer_kind=answer_kind,
            answer_value=response.answer_value,
            audit_ref=ref,
        )
        return self._running(payload, action_ref)

    def _run_report(self, action: ReportInconsistency, action_ref: AuditRef) -> StepResult:
        exhausted = self._charge(self._config.report_charge)
        if exhausted is not None:
            return self._truncate(exhausted, action_ref)
        ref = self._audit.append(
            AuditEventKind.PROGRESS_EVENT,
            "agent",
            {"kind": "report_inconsistency"},
            step_index=self._step_index,
        )
        return self._running(ReportAckPayload(audit_ref=ref), action_ref)

    # -------------------------------------------------------------- terminal
    def _finalize(self, action: TerminalAction, action_ref: AuditRef) -> StepResult:
        # Seal BEFORE verifying: seal_claim appends CLAIM_SEALED, and the
        # finalizer requires a SealedClaimBundle, so verification can never
        # precede sealing.
        sealed = self._audit.seal_claim(action.claim_bundle, step_index=self._step_index)
        report = self._finalizer.finalize(
            task=self._task,
            initial_view=WorkspaceReadView(self._store, self._initial_manifest),
            final_view=self._workspace.read_view(),
            sealed=sealed,
            constraint=self._constraint,
            audit=self._audit,
            resource_cost=self._resource_cost(),
        )
        self._report = report
        self._constraint = report.constraint
        self._audit.append(
            AuditEventKind.VERIFICATION,
            "environment",
            {"outcome_class": report.outcome_class.value, "semantic_pass": report.semantic_pass},
            step_index=self._step_index,
        )
        self._phase = "finalized"
        code = TerminalCode(action.claim_bundle.disposition.value)
        self._audit.append(
            AuditEventKind.EPISODE_END,
            "environment",
            {"terminal_code": code.value},
            step_index=self._step_index,
        )
        return StepResult(
            observation=self._observe(TerminalPayload(terminal_code=code)),
            terminated=True,
            truncated=False,
            reward=report.reward,
            constraint=report.constraint,
            terminal_code=code,
            audit_ref=action_ref,
            metrics={"outcome_class": report.outcome_class.value},
        )

    # --------------------------------------------------------------- helpers
    def _charge(self, charge: BudgetCharge) -> tuple[str, ...] | None:
        result = self._ledger.charge(charge)
        return None if result.applied else result.exhausted

    def _resource_cost(self) -> float:
        max_cost = self._task.public.budgets.max_cost_units
        if max_cost <= 0:
            return 0.0
        consumed = max_cost - self._ledger.view().remaining_cost_units
        return consumed / max_cost

    def _latch(self, flags: frozenset[str]) -> bool:
        if not flags:
            return False
        self._constraint = self._constraint.latch(**dict.fromkeys(flags, True))
        for flag in sorted(flags):
            self._audit.append(
                AuditEventKind.CONSTRAINT_FLAG,
                "environment",
                {"flag": flag},
                step_index=self._step_index,
            )
        return any(flag in _NON_COMPENSABLE for flag in flags)

    def _running(self, payload: ObservationPayload, action_ref: AuditRef) -> StepResult:
        return StepResult(
            observation=self._observe(payload),
            terminated=False,
            truncated=False,
            reward=RewardVector.zero(),
            constraint=self._constraint,
            terminal_code=None,
            audit_ref=action_ref,
        )

    def _constraint_halt(self, payload: ObservationPayload, action_ref: AuditRef) -> StepResult:
        from ptaie.kernel.rewards import OutcomeClass

        self._phase = "finalized"
        self._audit.append(
            AuditEventKind.EPISODE_END,
            "environment",
            {"terminal_code": TerminalCode.CONSTRAINT_HALT.value},
            step_index=self._step_index,
        )
        return StepResult(
            observation=self._observe(payload),
            terminated=True,
            truncated=False,
            reward=RewardVector.zero(),
            constraint=self._constraint,
            terminal_code=TerminalCode.CONSTRAINT_HALT,
            audit_ref=action_ref,
            metrics={"outcome_class": OutcomeClass.INTEGRITY_VIOLATION.value},
        )

    def _truncate(self, exhausted: tuple[str, ...], action_ref: AuditRef) -> StepResult:
        from ptaie.kernel.rewards import OutcomeClass

        self._phase = "finalized"
        self._audit.append(
            AuditEventKind.EPISODE_END,
            "environment",
            {"terminal_code": TerminalCode.BUDGET_EXHAUSTED.value, "exhausted": list(exhausted)},
            step_index=self._step_index,
        )
        return StepResult(
            observation=self._observe(TerminalPayload(terminal_code=TerminalCode.BUDGET_EXHAUSTED)),
            terminated=False,
            truncated=True,
            reward=RewardVector(resource_cost=1.0),
            constraint=self._constraint,
            terminal_code=TerminalCode.BUDGET_EXHAUSTED,
            audit_ref=action_ref,
            metrics={"outcome_class": OutcomeClass.RESOURCE_EXHAUSTION.value},
        )

    def _observe(self, payload: ObservationPayload) -> Observation:
        return Observation(
            step_index=self._step_index,
            payload=payload,
            budget=self._ledger.view(),
            staged_root=self._workspace.staged,
        )


class PtaieEnv:
    """The native protocol facade: generate a task and drive its episode.

    The finalizer (hidden verification + scoring) is injected — PR5 supplies
    the sft_chat implementation; PR4 tests supply a stub. Each ``reset`` builds
    a fresh engine, blob store, and audit log, so no state leaks across
    episodes.
    """

    __slots__ = ("_config", "_engine", "_finalizer", "_plugins")

    def __init__(
        self,
        plugins: "PluginRegistry",
        finalizer: Finalizer,
        *,
        config: EngineConfig | None = None,
    ) -> None:
        self._plugins = plugins
        self._finalizer = finalizer
        self._config = config
        self._engine: EpisodeEngine | None = None

    def reset(self, *, task_seed: int, artifact_class: str, env_version: str) -> Observation:
        plugin = self._plugins.get(artifact_class)
        generated = plugin.generate(task_seed, env_version)
        registry = ToolRegistry()
        registry.register_all(builtin_tools(plugin.visible_validators()))
        registry.register_all(plugin.tools())
        self._engine = EpisodeEngine(
            generated.task,
            store=generated.store,
            manifest=generated.manifest,
            registry=registry,
            oracle=plugin.clarification_oracle(generated.task),
            finalizer=self._finalizer,
            env_version=env_version,
            config=self._config,
        )
        return self._engine.reset()

    def step(self, action: Action) -> StepResult:
        if self._engine is None:
            raise KernelError("reset() must be called before step()")
        return self._engine.step(action)

    @property
    def engine(self) -> EpisodeEngine:
        if self._engine is None:
            raise KernelError("no active episode")
        return self._engine


def _episode_id(task_seed: int, env_version: str) -> str:
    return "ep-" + derive_seed(task_seed, "episode-id", env_version).to_bytes(8, "big").hex()


def _target_path(action: ToolAction) -> str | None:
    path = action.arguments.get("path")
    return path if isinstance(path, str) else None


def _action_payload(action: Action) -> dict[str, JsonValue]:
    if isinstance(action, ToolAction):
        return {"kind": "tool", "tool_name": action.tool_name, "arguments": action.arguments}
    if isinstance(action, AskClarification):
        return {"kind": "ask", "question_key": action.question_key}
    if isinstance(action, ReportInconsistency):
        return {"kind": "report"}
    if isinstance(action, TerminalAction):
        return {"kind": "terminal", "disposition": action.claim_bundle.disposition.value}
    raise KernelError("unreachable")  # pragma: no cover
