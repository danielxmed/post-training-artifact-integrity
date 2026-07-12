import pytest
from pydantic import ValidationError

from ptaie.kernel.claims import Claim, ClaimBundle
from ptaie.kernel.rewards import TerminalDisposition

ROOT = "a" * 64


def test_partial_handoff_requires_unresolved() -> None:
    with pytest.raises(ValidationError, match="unresolved"):
        ClaimBundle(
            disposition=TerminalDisposition.PARTIAL_HANDOFF,
            artifact_root_hash=ROOT,
            confidence=0.5,
        )
    bundle = ClaimBundle(
        disposition=TerminalDisposition.PARTIAL_HANDOFF,
        artifact_root_hash=ROOT,
        confidence=0.5,
        unresolved=("mojibake in rec-7 not repaired",),
    )
    assert bundle.unresolved


@pytest.mark.parametrize("confidence", [-0.1, 1.5])
def test_confidence_bounds(confidence: float) -> None:
    with pytest.raises(ValidationError):
        ClaimBundle(
            disposition=TerminalDisposition.COMMIT,
            artifact_root_hash=ROOT,
            confidence=confidence,
        )


def test_duplicate_claim_ids_rejected() -> None:
    claim = Claim(claim_id="c1", invariant_id="sft_chat.role_alternation", statement="restored")
    with pytest.raises(ValidationError, match="unique"):
        ClaimBundle(
            disposition=TerminalDisposition.COMMIT,
            artifact_root_hash=ROOT,
            confidence=0.9,
            claims=(claim, claim),
        )


def test_artifact_root_hash_validated() -> None:
    with pytest.raises(ValidationError, match="sha256"):
        ClaimBundle(
            disposition=TerminalDisposition.COMMIT,
            artifact_root_hash="not-a-hash",
            confidence=0.9,
        )
