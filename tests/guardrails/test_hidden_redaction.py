"""Hidden-side models must never leak field values through repr/str.

This is the string-formatting backstop of the leakage defense: even if a
hidden object ends up interpolated into an error message or log line, its
values must not appear.
"""

from typing import Any, cast

from pydantic import ValidationError

from ptaie.kernel.budgets import Budgets
from ptaie.kernel.contract import (
    ContractVariant,
    InconsistencyWitness,
    LatentContract,
    Requirement,
)
from ptaie.kernel.rewards import (
    ConstraintVector,
    OutcomeClass,
    RewardVector,
    TerminalDisposition,
)
from ptaie.kernel.store.vault import HiddenVault
from ptaie.kernel.task import (
    ClarificationEntry,
    DefectNode,
    LatentTaskState,
    TaskHidden,
    TaskPublic,
    TaskRecord,
)
from ptaie.kernel.verification import (
    CheckStatus,
    ClaimAssessment,
    VerificationReport,
    VerifierLayer,
    VerifierResult,
)

CANARY = "CANARY-93af1-SECRET"


def _hidden_with_canary() -> TaskHidden:
    contract = LatentContract(
        contract_id=CANARY,
        variants=(
            ContractVariant(
                variant_id="v0",
                requirements=(
                    Requirement(requirement_id=CANARY, predicate=CANARY, params={"x": CANARY}),
                ),
            ),
        ),
        true_variant_id="v0",
    )
    defect = DefectNode(
        defect_id=CANARY,
        operator=CANARY,
        operator_version=1,
        expected_failures=(CANARY,),
        violated_invariants=(CANARY,),
        observability="hidden_only",
        severity="semantic",
        certified_repair_ref=CANARY,
    )
    return TaskHidden(
        latent_state=LatentTaskState.REPAIRABLE,
        contract=contract,
        defect_dag=(defect,),
        protected_ids=(CANARY,),
        clarification_script=(
            ClarificationEntry(question_key=CANARY, answer_kind="value", answer_value=CANARY),
        ),
        expected_disposition=TerminalDisposition.COMMIT,
    )


def test_hidden_models_redact_repr_and_str() -> None:
    hidden = _hidden_with_canary()
    subjects: list[object] = [
        hidden,
        hidden.contract,
        hidden.contract.variants[0],
        hidden.contract.variants[0].requirements[0],
        hidden.defect_dag[0],
        hidden.clarification_script[0],
        HiddenVault(hidden),
        InconsistencyWitness(assertion_a=CANARY, assertion_b=CANARY, rule_id=CANARY),
    ]
    for subject in subjects:
        for rendering in (repr(subject), str(subject), f"{subject}"):
            assert CANARY not in rendering, f"{type(subject).__name__} leaks values via repr/str"


def test_validation_errors_do_not_echo_hidden_input() -> None:
    """pydantic ValidationError must not leak hidden values (hide_input_in_errors)."""
    try:
        ClarificationEntry(question_key=CANARY, answer_kind="unknown", answer_value=CANARY)
    except ValidationError as error:
        assert CANARY not in str(error)
        assert CANARY not in repr(error)
    else:
        raise AssertionError("expected ValidationError")


def test_rich_and_devtools_renderers_get_no_values() -> None:
    """__repr_args__ is the source rich/devtools formatters read — it must be empty."""
    hidden = _hidden_with_canary()
    assert list(hidden.__repr_args__()) == []
    # pydantic types __rich_repr__ as a plain callable attribute; go through
    # Any to call it the way rich would.
    rendered = "".join(str(part) for part in cast(Any, hidden).__rich_repr__())
    assert CANARY not in rendered


def test_verification_models_are_redacted() -> None:
    report = VerificationReport(
        sealed_bundle_hash="f" * 64,
        results=(
            VerifierResult(
                check_id="sft_chat.schema",
                layer=VerifierLayer.SCHEMA,
                status=CheckStatus.FAILED,
                failure_codes=(CANARY,),
                details={"secret": CANARY},
            ),
        ),
        claim_assessments=(ClaimAssessment(claim_id="c1", supported=False, reason=CANARY),),
        semantic_pass=False,
        outcome_class=OutcomeClass.FALSE_COMMIT,
        constraint=ConstraintVector(),
        reward=RewardVector.zero(),
    )
    for rendering in (repr(report), str(report), f"{report}"):
        assert CANARY not in rendering
    assessment = report.claim_assessments[0]
    for rendering in (repr(assessment), str(assessment), f"{assessment}"):
        assert CANARY not in rendering


def test_task_record_repr_redacts_the_hidden_half() -> None:
    hidden = _hidden_with_canary()
    public = TaskPublic(
        task_id="t-1",
        env_version="0.1.0",
        artifact_class="sft_chat",
        brief="Review this dataset.",
        initial_manifest="c" * 64,
        budgets=Budgets(max_steps=10, max_cost_units=100, max_questions=2, max_mutations=5),
    )
    record = TaskRecord(seed=1, public=public, hidden=hidden)
    for rendering in (repr(record), str(record), f"{record}"):
        assert CANARY not in rendering
