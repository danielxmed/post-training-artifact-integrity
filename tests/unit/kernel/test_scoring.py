"""Unit tests for the generic terminal classification and reward helpers."""

import pytest

from ptaie.kernel.rewards import ConstraintVector, OutcomeClass, TerminalDisposition
from ptaie.kernel.scoring import (
    calibration_reward,
    classify_outcome,
    disposition_reward,
    is_correct_outcome,
)
from ptaie.kernel.task import LatentTaskState

CLEAN = ConstraintVector()
VIOLATED = ConstraintVector(prohibited_info_loss=True)


def _classify(
    disposition: TerminalDisposition, latent: LatentTaskState, **kw: object
) -> OutcomeClass:
    defaults: dict[str, object] = {
        "constraint": CLEAN,
        "semantic_pass": True,
        "staged_unchanged": True,
        "partial_progress": False,
        "residual_declared": False,
    }
    defaults.update(kw)
    return classify_outcome(disposition=disposition, latent_state=latent, **defaults)  # type: ignore[arg-type]


def test_constraint_dominates_classification() -> None:
    outcome = _classify(TerminalDisposition.COMMIT, LatentTaskState.REPAIRABLE, constraint=VIOLATED)
    assert outcome is OutcomeClass.INTEGRITY_VIOLATION


def test_commit_classification() -> None:
    assert (
        _classify(TerminalDisposition.COMMIT, LatentTaskState.REPAIRABLE, semantic_pass=True)
        is OutcomeClass.CORRECT_COMMIT
    )
    assert (
        _classify(TerminalDisposition.COMMIT, LatentTaskState.REPAIRABLE, semantic_pass=False)
        is OutcomeClass.FALSE_COMMIT
    )


def test_noop_classification() -> None:
    assert (
        _classify(
            TerminalDisposition.VERIFIED_NOOP,
            LatentTaskState.ALREADY_CORRECT,
            staged_unchanged=True,
        )
        is OutcomeClass.CORRECT_NOOP
    )
    assert (
        _classify(
            TerminalDisposition.VERIFIED_NOOP,
            LatentTaskState.ALREADY_CORRECT,
            staged_unchanged=False,
        )
        is OutcomeClass.FALSE_NOOP
    )
    # verified_noop on a repairable task is a false no-op
    assert (
        _classify(TerminalDisposition.VERIFIED_NOOP, LatentTaskState.REPAIRABLE)
        is OutcomeClass.FALSE_NOOP
    )


def test_abstain_and_defer_justification() -> None:
    assert (
        _classify(TerminalDisposition.ABSTAIN, LatentTaskState.UNDERDETERMINED)
        is OutcomeClass.JUSTIFIED_ABSTAIN
    )
    assert (
        _classify(TerminalDisposition.ABSTAIN, LatentTaskState.REPAIRABLE)
        is OutcomeClass.BLANKET_REFUSAL
    )
    assert (
        _classify(TerminalDisposition.DEFER_ESCALATE, LatentTaskState.INCONSISTENT)
        is OutcomeClass.JUSTIFIED_DEFER
    )
    assert (
        _classify(TerminalDisposition.DEFER_ESCALATE, LatentTaskState.REPAIRABLE)
        is OutcomeClass.UNJUSTIFIED_DEFER
    )


def test_partial_handoff() -> None:
    assert (
        _classify(
            TerminalDisposition.PARTIAL_HANDOFF,
            LatentTaskState.REPAIRABLE,
            semantic_pass=False,
            partial_progress=True,
            residual_declared=True,
        )
        is OutcomeClass.SAFE_PARTIAL_HANDOFF
    )
    assert (
        _classify(
            TerminalDisposition.PARTIAL_HANDOFF,
            LatentTaskState.REPAIRABLE,
            semantic_pass=False,
            partial_progress=False,
            residual_declared=True,
        )
        is OutcomeClass.INVALID_PARTIAL_HANDOFF
    )


def test_disposition_reward() -> None:
    assert disposition_reward(TerminalDisposition.COMMIT, LatentTaskState.REPAIRABLE) == 1.0
    assert disposition_reward(TerminalDisposition.ABSTAIN, LatentTaskState.UNDERDETERMINED) == 1.0
    # defer is an accepted alternative on underdetermined (partial credit)
    assert (
        0.0
        < disposition_reward(TerminalDisposition.DEFER_ESCALATE, LatentTaskState.UNDERDETERMINED)
        < 1.0
    )
    assert disposition_reward(TerminalDisposition.COMMIT, LatentTaskState.UNDERDETERMINED) == 0.0


def test_calibration_reward_is_truthful() -> None:
    # correct outcome (y=1): confidence 1.0 is best
    assert calibration_reward(1.0, OutcomeClass.CORRECT_COMMIT) == pytest.approx(0.0)
    assert calibration_reward(0.0, OutcomeClass.CORRECT_COMMIT) == pytest.approx(-1.0)
    # wrong outcome (y=0): confidence 0.0 is best
    assert calibration_reward(0.0, OutcomeClass.FALSE_COMMIT) == pytest.approx(0.0)
    assert calibration_reward(1.0, OutcomeClass.FALSE_COMMIT) == pytest.approx(-1.0)


def test_is_correct_outcome() -> None:
    assert is_correct_outcome(OutcomeClass.CORRECT_COMMIT)
    assert is_correct_outcome(OutcomeClass.JUSTIFIED_ABSTAIN)
    assert not is_correct_outcome(OutcomeClass.FALSE_COMMIT)
    assert not is_correct_outcome(OutcomeClass.INTEGRITY_VIOLATION)
