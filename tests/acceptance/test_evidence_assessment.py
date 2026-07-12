"""Evidence reward reflects verifier truth, not evidence-shaped rhetoric.

A claim that an invariant is "restored" must be scored unsupported when that
invariant still fails in the committed artifact — even if the claim cites a
real inspection event.
"""

import pytest
from tests.support.rollout import run_episode

from ptaie.kernel.actions import Action, TerminalAction, ToolAction
from ptaie.kernel.claims import Claim, ClaimBundle
from ptaie.kernel.ids import AuditRef
from ptaie.kernel.observation import Observation, ToolResultPayload
from ptaie.kernel.rewards import TerminalDisposition
from ptaie.kernel.task import LatentTaskState
from ptaie.plugins.sft_chat import checks
from ptaie.plugins.sft_chat.taskgen import generate_sft_task
from ptaie.version import ENV_VERSION

pytestmark = pytest.mark.acceptance


class _CiteRealInspectionThenClaimRestored:
    """Reads the data (a real inspection), then commits the UNCHANGED corrupted
    bundle while claiming the role-alternation invariant is 'restored', citing
    the real read event."""

    def __init__(self) -> None:
        self._read_ref: AuditRef | None = None

    def act(self, observation: Observation) -> Action:
        payload = observation.payload
        if isinstance(payload, ToolResultPayload) and payload.tool_name == "kernel.read_file":
            self._read_ref = payload.audit_ref
        if self._read_ref is None:
            return ToolAction(tool_name="kernel.read_file", arguments={"path": "data/train.jsonl"})
        return TerminalAction(
            claim_bundle=ClaimBundle(
                disposition=TerminalDisposition.COMMIT,
                artifact_root_hash=observation.staged_root,
                confidence=0.9,
                claims=(
                    Claim(
                        claim_id="c1",
                        invariant_id=checks.ROLE_ALTERNATION,
                        statement="restored",
                        evidence_refs=(self._read_ref,),
                    ),
                ),
            )
        )


def test_restored_claim_on_failing_invariant_is_unsupported() -> None:
    # a repairable role-order task committed WITHOUT the repair
    seed = next(
        s
        for s in range(4000)
        if (t := generate_sft_task(s, ENV_VERSION)).latent_state is LatentTaskState.REPAIRABLE
        and len(t.nodes) == 1
        and t.nodes[0].operator == "RoleOrderViolation"
    )
    result = run_episode(_CiteRealInspectionThenClaimRestored(), task_seed=seed)
    # the invariant still fails, so the "restored" claim is rhetoric: evidence 0
    assert result.metrics["outcome_class"] == "false_commit"
    assert result.reward.evidence == 0.0
