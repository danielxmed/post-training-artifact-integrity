"""Task records: the public/hidden split at the heart of partial observability.

``TaskPublic`` is the only half that may ever be serialized into an
observation. ``TaskHidden`` lives in the engine's vault; its models inherit
the redacted repr from :class:`~ptaie.kernel.contract.HiddenBaseModel`.

``expected_disposition`` exists for tests and the oracle policy only — the
scoring path classifies from verifier outcomes plus latent state and must
never read it (guardrail-tested).
"""

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ptaie.kernel.budgets import Budgets
from ptaie.kernel.canonical import JsonValue
from ptaie.kernel.contract import HiddenBaseModel, InconsistencyWitness, LatentContract
from ptaie.kernel.ids import require_hex256
from ptaie.kernel.rewards import TerminalDisposition


class LatentTaskState(StrEnum):
    """The six latent task states. Generation prior (M1):
    55 / 10 / 15 / 10 / 5 / 5 in declaration order."""

    REPAIRABLE = "repairable"
    ALREADY_CORRECT = "already_correct"
    AMBIGUOUS_RESOLVABLE = "ambiguous"
    UNDERDETERMINED = "underdetermined"
    INCONSISTENT = "inconsistent"
    ADVERSARIAL_TRAP = "adversarial_trap"


class DefectNode(HiddenBaseModel):
    """One corruption applied during generation — a node in the causal defect DAG.

    ``masks`` lists defects whose symptoms this node hides (e.g. a record
    that fails schema parsing blocks the role-order checks on that record);
    masked expectations report ``blocked``, never ``passed``.
    """

    defect_id: str
    operator: str
    operator_version: int
    params: dict[str, JsonValue] = Field(default_factory=dict)
    expected_failures: tuple[str, ...]
    violated_invariants: tuple[str, ...]
    affected_ids: tuple[str, ...] = ()
    observability: Literal["public_validator_visible", "inspection_visible", "hidden_only"]
    severity: Literal["cosmetic", "semantic", "training_breaking"]
    accepted_postconditions: tuple[str, ...] = ()
    prohibited_side_effects: tuple[str, ...] = ()
    certified_repair_ref: str
    alternative_repair_refs: tuple[str, ...] = ()
    parents: tuple[str, ...] = ()
    masks: tuple[str, ...] = ()


class TrapSpec(HiddenBaseModel):
    """What shortcut an adversarial-trap task baits, and which canaries detect it.

    Detection is latent-state-independent (the same constraint checks run in
    every episode); the trap only differs in making the bait profitable.
    """

    kind: str
    bait: str
    canary_ids: tuple[str, ...] = ()


class ClarificationEntry(HiddenBaseModel):
    """One scripted clarification answer, keyed by contract dimension."""

    question_key: str
    answer_kind: Literal["value", "unknown", "already_stated"]
    answer_value: str | None = None
    resolves_variant: str | None = None
    decision_relevant: bool = False

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.answer_kind == "value" and self.answer_value is None:
            raise ValueError("answer_kind 'value' requires answer_value")
        if self.answer_kind != "value" and self.answer_value is not None:
            raise ValueError(f"answer_kind {self.answer_kind!r} must not carry answer_value")
        return self


class TaskPublic(BaseModel):
    """The agent-visible half of a task record.

    ``claimable_invariants`` is the stable, state-independent vocabulary the
    agent's claims must reference — the identical list in every episode of an
    artifact class, so it can never leak latent state.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    env_version: str
    artifact_class: str
    brief: str
    initial_manifest: str
    budgets: Budgets
    visible_validator_names: tuple[str, ...] = ()
    claimable_invariants: tuple[str, ...] = ()
    irreversible_boundaries: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> Self:
        require_hex256(self.initial_manifest, what="initial_manifest")
        return self


class TaskHidden(HiddenBaseModel):
    """The hidden half: latent state, contract, defect DAG, scripts, and canaries."""

    latent_state: LatentTaskState
    contract: LatentContract
    defect_dag: tuple[DefectNode, ...] = ()
    clean_manifest: str | None = None
    protected_ids: tuple[str, ...] = ()
    inconsistency_witness: InconsistencyWitness | None = None
    underdetermination_note: str | None = None
    trap_spec: TrapSpec | None = None
    clarification_script: tuple[ClarificationEntry, ...] = ()
    expected_disposition: TerminalDisposition
    hidden_fixture_manifest: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        state = self.latent_state
        self._check_dag()

        if (self.inconsistency_witness is not None) != (state is LatentTaskState.INCONSISTENT):
            raise ValueError("inconsistency_witness is required iff latent_state is INCONSISTENT")
        if (self.underdetermination_note is not None) != (state is LatentTaskState.UNDERDETERMINED):
            raise ValueError(
                "underdetermination_note is required iff latent_state is UNDERDETERMINED"
            )
        if (self.trap_spec is not None) != (state is LatentTaskState.ADVERSARIAL_TRAP):
            raise ValueError("trap_spec is required iff latent_state is ADVERSARIAL_TRAP")

        if state is LatentTaskState.REPAIRABLE and not self.defect_dag:
            raise ValueError("REPAIRABLE tasks require a non-empty defect DAG")
        if state is LatentTaskState.ALREADY_CORRECT and self.defect_dag:
            raise ValueError("ALREADY_CORRECT tasks must have an empty defect DAG")

        if state is LatentTaskState.AMBIGUOUS_RESOLVABLE:
            if len(self.contract.variants) < 2 or self.contract.true_variant_id is None:
                raise ValueError(
                    "AMBIGUOUS_RESOLVABLE requires >=2 contract variants and a true variant"
                )
            if not any(
                entry.answer_kind == "value" and entry.decision_relevant
                for entry in self.clarification_script
            ):
                raise ValueError(
                    "AMBIGUOUS_RESOLVABLE requires a decision-relevant scripted answer"
                )
        if state is LatentTaskState.UNDERDETERMINED and (
            len(self.contract.variants) < 2 or self.contract.true_variant_id is not None
        ):
            raise ValueError("UNDERDETERMINED requires >=2 contract variants and no true variant")
        if (
            state not in (LatentTaskState.AMBIGUOUS_RESOLVABLE, LatentTaskState.UNDERDETERMINED)
            and len(self.contract.variants) > 1
            and self.contract.true_variant_id is None
        ):
            raise ValueError(
                f"{state} requires a determinate contract (single variant or true_variant_id set)"
            )
        return self

    def _check_dag(self) -> None:
        defect_ids = [node.defect_id for node in self.defect_dag]
        known = set(defect_ids)
        if len(known) != len(defect_ids):
            raise ValueError("defect ids must be unique")
        for node in self.defect_dag:
            if node.defect_id in node.parents or node.defect_id in node.masks:
                raise ValueError(f"defect {node.defect_id} references itself")
            for parent in node.parents:
                if parent not in known:
                    raise ValueError(f"defect {node.defect_id} has unknown parent {parent}")
            for masked in node.masks:
                if masked not in known:
                    raise ValueError(f"defect {node.defect_id} masks unknown defect {masked}")
        # Kahn's algorithm over parent edges: reject cycles (a DAG, not a graph).
        indegree = {node.defect_id: len(node.parents) for node in self.defect_dag}
        children: dict[str, list[str]] = {node.defect_id: [] for node in self.defect_dag}
        for node in self.defect_dag:
            for parent in node.parents:
                children[parent].append(node.defect_id)
        queue = [defect_id for defect_id, degree in indegree.items() if degree == 0]
        visited = 0
        while queue:
            current = queue.pop()
            visited += 1
            for child in children[current]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if visited != len(self.defect_dag):
            raise ValueError("defect_dag contains a cycle")


class TaskRecord(BaseModel):
    """A complete generated task: root seed + public and hidden halves."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int
    public: TaskPublic
    hidden: TaskHidden
