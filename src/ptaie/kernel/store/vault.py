"""The engine-held container for hidden task state.

V1 trust boundary (documented honestly): the agent is out-of-process by
definition — it submits JSON actions and receives JSON observations, and
there is no code-execution tool in M1, so agent-controlled code never runs
inside the environment process. Within the process, discipline is API-shaped:
``ToolContext`` and every observation-building code path hold no reference to
the vault (guardrail-tested). Real process sandboxing arrives with any future
code-execution tool; the verifier protocol already takes only read views so
it can move behind a process boundary without interface change.
"""

from ptaie.kernel.task import TaskHidden


class HiddenVault:
    """Holds the hidden half of the task record. Engine-only."""

    __slots__ = ("_hidden",)

    def __init__(self, hidden: TaskHidden) -> None:
        self._hidden = hidden

    @property
    def hidden(self) -> TaskHidden:
        return self._hidden

    def __repr__(self) -> str:
        return "HiddenVault(<redacted>)"

    def __str__(self) -> str:
        return "HiddenVault(<redacted>)"
