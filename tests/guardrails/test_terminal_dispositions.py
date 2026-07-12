"""Exactly five agent-selectable terminal dispositions (binding invariant)."""

from ptaie.kernel.rewards import TerminalCode, TerminalDisposition


def test_exactly_five_dispositions() -> None:
    assert {member.value for member in TerminalDisposition} == {
        "commit",
        "verified_noop",
        "abstain",
        "defer_escalate",
        "partial_handoff",
    }
    assert len(TerminalDisposition) == 5


def test_terminal_code_is_dispositions_plus_engine_endings() -> None:
    disposition_values = {member.value for member in TerminalDisposition}
    code_values = {member.value for member in TerminalCode}
    assert code_values == disposition_values | {"budget_exhausted", "constraint_halt"}
