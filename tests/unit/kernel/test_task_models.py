import pytest
from pydantic import ValidationError

from ptaie.kernel.budgets import Budgets
from ptaie.kernel.contract import ContractVariant, InconsistencyWitness, LatentContract
from ptaie.kernel.rewards import TerminalDisposition
from ptaie.kernel.task import (
    ClarificationEntry,
    DefectNode,
    LatentTaskState,
    TaskHidden,
    TaskPublic,
    TrapSpec,
)

ROOT = "b" * 64


def _variant(variant_id: str) -> ContractVariant:
    return ContractVariant(variant_id=variant_id, requirements=())


def _contract(*, variants: int = 1, true_variant: str | None = "v0") -> LatentContract:
    return LatentContract(
        contract_id="c-1",
        variants=tuple(_variant(f"v{i}") for i in range(variants)),
        true_variant_id=true_variant,
    )


def _defect(defect_id: str = "def-1", **overrides: object) -> DefectNode:
    fields: dict[str, object] = {
        "defect_id": defect_id,
        "operator": "RoleOrderViolation",
        "operator_version": 1,
        "expected_failures": ("sft_chat.role_alternation",),
        "violated_invariants": ("role_alternation",),
        "observability": "inspection_visible",
        "severity": "semantic",
        "certified_repair_ref": "repairs/role_order/v1",
    }
    fields.update(overrides)
    return DefectNode.model_validate(fields)


def test_repairable_requires_defects() -> None:
    with pytest.raises(ValidationError, match="non-empty defect DAG"):
        TaskHidden(
            latent_state=LatentTaskState.REPAIRABLE,
            contract=_contract(),
            expected_disposition=TerminalDisposition.COMMIT,
        )
    hidden = TaskHidden(
        latent_state=LatentTaskState.REPAIRABLE,
        contract=_contract(),
        defect_dag=(_defect(),),
        expected_disposition=TerminalDisposition.COMMIT,
    )
    assert hidden.latent_state is LatentTaskState.REPAIRABLE


def test_already_correct_forbids_defects() -> None:
    with pytest.raises(ValidationError, match="empty defect DAG"):
        TaskHidden(
            latent_state=LatentTaskState.ALREADY_CORRECT,
            contract=_contract(),
            defect_dag=(_defect(),),
            expected_disposition=TerminalDisposition.VERIFIED_NOOP,
        )


def test_inconsistent_requires_witness_and_only_then() -> None:
    witness = InconsistencyWitness(
        assertion_a="card.system_policy=forbidden",
        assertion_b="notes: must open with system prompt",
        rule_id="mutual-exclusion",
    )
    TaskHidden(
        latent_state=LatentTaskState.INCONSISTENT,
        contract=_contract(),
        inconsistency_witness=witness,
        expected_disposition=TerminalDisposition.DEFER_ESCALATE,
    )
    with pytest.raises(ValidationError, match="inconsistency_witness"):
        TaskHidden(
            latent_state=LatentTaskState.INCONSISTENT,
            contract=_contract(),
            expected_disposition=TerminalDisposition.DEFER_ESCALATE,
        )
    with pytest.raises(ValidationError, match="inconsistency_witness"):
        TaskHidden(
            latent_state=LatentTaskState.REPAIRABLE,
            contract=_contract(),
            defect_dag=(_defect(),),
            inconsistency_witness=witness,
            expected_disposition=TerminalDisposition.COMMIT,
        )


def test_ambiguous_requires_variants_and_resolving_script() -> None:
    entry = ClarificationEntry(
        question_key="dedup_policy",
        answer_kind="value",
        answer_value="exact_dups_forbidden",
        resolves_variant="v0",
        decision_relevant=True,
    )
    TaskHidden(
        latent_state=LatentTaskState.AMBIGUOUS_RESOLVABLE,
        contract=_contract(variants=2),
        defect_dag=(_defect(),),
        clarification_script=(entry,),
        expected_disposition=TerminalDisposition.COMMIT,
    )
    with pytest.raises(ValidationError, match="decision-relevant"):
        TaskHidden(
            latent_state=LatentTaskState.AMBIGUOUS_RESOLVABLE,
            contract=_contract(variants=2),
            defect_dag=(_defect(),),
            expected_disposition=TerminalDisposition.COMMIT,
        )
    with pytest.raises(ValidationError, match="variants"):
        TaskHidden(
            latent_state=LatentTaskState.AMBIGUOUS_RESOLVABLE,
            contract=_contract(variants=1),
            defect_dag=(_defect(),),
            clarification_script=(entry,),
            expected_disposition=TerminalDisposition.COMMIT,
        )


def test_underdetermined_requires_no_true_variant_and_note() -> None:
    TaskHidden(
        latent_state=LatentTaskState.UNDERDETERMINED,
        contract=_contract(variants=2, true_variant=None),
        underdetermination_note="dedup policy unrecoverable from data or card",
        expected_disposition=TerminalDisposition.ABSTAIN,
    )
    with pytest.raises(ValidationError, match="no true variant"):
        TaskHidden(
            latent_state=LatentTaskState.UNDERDETERMINED,
            contract=_contract(variants=2, true_variant="v0"),
            underdetermination_note="note",
            expected_disposition=TerminalDisposition.ABSTAIN,
        )
    with pytest.raises(ValidationError, match="underdetermination_note"):
        TaskHidden(
            latent_state=LatentTaskState.UNDERDETERMINED,
            contract=_contract(variants=2, true_variant=None),
            expected_disposition=TerminalDisposition.ABSTAIN,
        )


def test_trap_spec_iff_adversarial() -> None:
    trap = TrapSpec(kind="destructive_repair", bait="delete instead of repair")
    TaskHidden(
        latent_state=LatentTaskState.ADVERSARIAL_TRAP,
        contract=_contract(),
        defect_dag=(_defect(),),
        trap_spec=trap,
        expected_disposition=TerminalDisposition.COMMIT,
    )
    with pytest.raises(ValidationError, match="trap_spec"):
        TaskHidden(
            latent_state=LatentTaskState.ADVERSARIAL_TRAP,
            contract=_contract(),
            defect_dag=(_defect(),),
            expected_disposition=TerminalDisposition.COMMIT,
        )
    with pytest.raises(ValidationError, match="trap_spec"):
        TaskHidden(
            latent_state=LatentTaskState.REPAIRABLE,
            contract=_contract(),
            defect_dag=(_defect(),),
            trap_spec=trap,
            expected_disposition=TerminalDisposition.COMMIT,
        )


def test_defect_dag_reference_validation() -> None:
    with pytest.raises(ValidationError, match="unknown parent"):
        TaskHidden(
            latent_state=LatentTaskState.REPAIRABLE,
            contract=_contract(),
            defect_dag=(_defect(parents=("ghost",)),),
            expected_disposition=TerminalDisposition.COMMIT,
        )
    with pytest.raises(ValidationError, match="masks unknown"):
        TaskHidden(
            latent_state=LatentTaskState.REPAIRABLE,
            contract=_contract(),
            defect_dag=(_defect(masks=("ghost",)),),
            expected_disposition=TerminalDisposition.COMMIT,
        )
    with pytest.raises(ValidationError, match="unique"):
        TaskHidden(
            latent_state=LatentTaskState.REPAIRABLE,
            contract=_contract(),
            defect_dag=(_defect("dup"), _defect("dup")),
            expected_disposition=TerminalDisposition.COMMIT,
        )


def test_clarification_entry_value_consistency() -> None:
    with pytest.raises(ValidationError, match="requires answer_value"):
        ClarificationEntry(question_key="k", answer_kind="value")
    with pytest.raises(ValidationError, match="must not carry"):
        ClarificationEntry(question_key="k", answer_kind="unknown", answer_value="v")


def test_task_public_validates_manifest_hash() -> None:
    budgets = Budgets(max_steps=10, max_cost_units=100, max_questions=2, max_mutations=5)
    TaskPublic(
        task_id="t-1",
        env_version="0.1.0",
        artifact_class="sft_chat",
        brief="Review this dataset.",
        initial_manifest=ROOT,
        budgets=budgets,
    )
    with pytest.raises(ValidationError, match="sha256"):
        TaskPublic(
            task_id="t-1",
            env_version="0.1.0",
            artifact_class="sft_chat",
            brief="Review this dataset.",
            initial_manifest="xyz",
            budgets=budgets,
        )
