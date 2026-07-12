"""Serialization round-trips for the frozen protocol types.

These types are the native environment protocol (actions in, observations
out). PR2 freezes them; later PRs build on them. Round-tripping through JSON
here pins the wire behavior — a change that breaks these tests is a protocol
change and must be deliberate.
"""

import pytest
from pydantic import TypeAdapter, ValidationError

from ptaie.kernel.actions import (
    Action,
    ActionCategory,
    ActionTrace,
    AskClarification,
    ReportInconsistency,
    TerminalAction,
    ToolAction,
)
from ptaie.kernel.budgets import BudgetView
from ptaie.kernel.claims import ClaimBundle
from ptaie.kernel.ids import AuditRef
from ptaie.kernel.observation import (
    ClarificationPayload,
    Observation,
    TaskBriefPayload,
    TerminalPayload,
    ToolError,
    ToolErrorCode,
    ToolMenuEntry,
    ToolOutcome,
    ToolResultPayload,
)
from ptaie.kernel.rewards import (
    ConstraintVector,
    OutcomeClass,
    RewardVector,
    TerminalCode,
    TerminalDisposition,
)
from ptaie.kernel.verification import (
    CheckStatus,
    ClaimAssessment,
    VerificationReport,
    VerifierLayer,
    VerifierResult,
)

_ACTION_ADAPTER: TypeAdapter[object] = TypeAdapter(Action)

_REF = AuditRef(episode_id="ep-1", event_index=0, event_hash="d" * 64)
_BUNDLE = ClaimBundle(
    disposition=TerminalDisposition.COMMIT, artifact_root_hash="a" * 64, confidence=0.75
)


@pytest.mark.parametrize(
    "action",
    [
        ToolAction(tool_name="kernel.read_file", arguments={"path": "data/train.jsonl"}),
        AskClarification(question_key="dedup_policy", question_text="Are duplicates intended?"),
        ReportInconsistency(description="card contradicts notes", evidence_refs=(_REF,)),
        TerminalAction(claim_bundle=_BUNDLE),
    ],
)
def test_action_union_roundtrip(action: object) -> None:
    dumped = _ACTION_ADAPTER.dump_json(action)
    restored = _ACTION_ADAPTER.validate_json(dumped)
    assert restored == action


def test_unknown_action_kind_rejected() -> None:
    with pytest.raises(ValidationError):
        _ACTION_ADAPTER.validate_python({"kind": "shell", "command": "rm -rf /"})


def test_wire_rejects_non_finite_floats() -> None:
    """A wire-valid value must always be canonicalizable: NaN/Infinity are
    rejected at validation time, matching the canonical hashing contract."""
    with pytest.raises(ValidationError):
        ToolAction.model_validate(
            {"kind": "tool", "tool_name": "x", "arguments": {"v": float("nan")}}
        )
    with pytest.raises(ValidationError):
        ToolAction.model_validate(
            {"kind": "tool", "tool_name": "x", "arguments": {"v": float("inf")}}
        )
    with pytest.raises(ValidationError):
        ToolAction.model_validate_json('{"kind":"tool","tool_name":"x","arguments":{"v":Infinity}}')


def test_action_trace_roundtrip() -> None:
    trace = ActionTrace(
        env_version="0.1.0",
        task_seed=42,
        artifact_class="sft_chat",
        actions=(
            ToolAction(tool_name="kernel.list_files"),
            TerminalAction(claim_bundle=_BUNDLE),
        ),
    )
    assert ActionTrace.model_validate_json(trace.model_dump_json()) == trace


def _budget_view() -> BudgetView:
    return BudgetView(
        remaining_steps=5,
        remaining_cost_units=50,
        remaining_questions=1,
        remaining_mutations=2,
    )


@pytest.mark.parametrize(
    "payload",
    [
        TaskBriefPayload(
            brief="Review this dataset.",
            files=("data/train.jsonl", "dataset_card.json"),
            tool_menu=(
                ToolMenuEntry(
                    name="kernel.read_file",
                    category=ActionCategory.INSPECT,
                    description="Read a file from the workspace.",
                ),
            ),
            claimable_invariants=("sft_chat.role_alternation",),
        ),
        ToolResultPayload(
            tool_name="kernel.read_file",
            outcome=ToolOutcome(status="ok", output={"content": "{}"}),
            audit_ref=_REF,
        ),
        ToolResultPayload(
            tool_name="kernel.read_file",
            outcome=ToolOutcome(
                status="error",
                error=ToolError(code=ToolErrorCode.FILE_NOT_FOUND, message="no such path"),
            ),
            audit_ref=_REF,
        ),
        ClarificationPayload(
            question_key="dedup_policy",
            answer_kind="value",
            answer_value="exact_dups_forbidden",
            audit_ref=_REF,
        ),
        TerminalPayload(terminal_code=TerminalCode.COMMIT),
    ],
)
def test_observation_roundtrip(payload: object) -> None:
    observation = Observation(
        step_index=3,
        payload=payload,  # type: ignore[arg-type]
        budget=_budget_view(),
        staged_root="e" * 64,
        notices=("mutation budget low",),
    )
    assert Observation.model_validate_json(observation.model_dump_json()) == observation


def test_tool_outcome_consistency_enforced() -> None:
    with pytest.raises(ValidationError, match="ok outcomes"):
        ToolOutcome(status="ok")
    with pytest.raises(ValidationError, match="error outcomes"):
        ToolOutcome(status="error", output={"x": 1})
    with pytest.raises(ValidationError, match="error outcomes"):
        ToolOutcome(
            status="error",
            output={"x": 1},
            error=ToolError(code=ToolErrorCode.INTERNAL_ERROR, message="m"),
        )


def test_verifier_result_passed_cannot_carry_failure_codes() -> None:
    with pytest.raises(ValidationError, match="failure codes"):
        VerifierResult(
            check_id="sft_chat.schema",
            layer=VerifierLayer.SCHEMA,
            status=CheckStatus.PASSED,
            failure_codes=("oops",),
        )


def test_verification_report_roundtrip() -> None:
    report = VerificationReport(
        sealed_bundle_hash="f" * 64,
        results=(
            VerifierResult(
                check_id="sft_chat.role_alternation",
                layer=VerifierLayer.LOCAL_SEMANTICS,
                status=CheckStatus.FAILED,
                failure_codes=("consecutive_user_turns",),
            ),
            VerifierResult(
                check_id="sft_chat.schema",
                layer=VerifierLayer.SCHEMA,
                status=CheckStatus.BLOCKED,
            ),
        ),
        claim_assessments=(
            ClaimAssessment(claim_id="c1", supported=False, reason="cited event reads no file"),
        ),
        semantic_pass=False,
        outcome_class=OutcomeClass.FALSE_COMMIT,
        constraint=ConstraintVector(),
        reward=RewardVector(semantic=0.0, resource_cost=0.4),
    )
    assert VerificationReport.model_validate_json(report.model_dump_json()) == report


def test_verifier_layers_m1_ships_zero_through_four() -> None:
    assert [layer.value for layer in VerifierLayer] == list(range(9))
    assert VerifierLayer.INTEGRITY.value == 0
    assert VerifierLayer.RELATIONAL.value == 4
