"""The typed action space.

Five categories — inspect, mutate, test, communicate, terminal. The first
three flow through the tool registry as :class:`ToolAction`; communication
and termination are dedicated action kinds, not tools. There is no free-form
"done" action, no shell, and no code execution in M1.
"""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from ptaie.kernel.canonical import JsonValue
from ptaie.kernel.claims import ClaimBundle
from ptaie.kernel.ids import AuditRef


class ActionCategory(StrEnum):
    INSPECT = "inspect"
    MUTATE = "mutate"
    TEST = "test"
    COMMUNICATE = "communicate"
    TERMINAL = "terminal"


class ToolAction(BaseModel):
    """Invoke a registered typed tool (inspect / mutate / test category)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["tool"] = "tool"
    tool_name: str
    arguments: dict[str, JsonValue] = {}


class AskClarification(BaseModel):
    """Ask one structured clarification.

    ``question_key`` is matched against the task's scripted answers (M1:
    contract dimension names). ``question_text`` is free prose — logged for
    interpretability, never parsed or matched.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["ask"] = "ask"
    question_key: str | None = None
    question_text: str = ""


class ReportInconsistency(BaseModel):
    """Flag an inconsistency in the task's requirements (non-terminal).

    Strengthens a later ABSTAIN/DEFER disposition on inconsistent tasks.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["report"] = "report"
    description: str
    evidence_refs: tuple[AuditRef, ...] = ()


class TerminalAction(BaseModel):
    """End the episode with a sealed claim bundle (Evidence-Locked Commit)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["terminal"] = "terminal"
    claim_bundle: ClaimBundle


type Action = Annotated[
    ToolAction | AskClarification | ReportInconsistency | TerminalAction,
    Field(discriminator="kind"),
]


class ActionTrace(BaseModel):
    """A serialized episode input: everything needed to replay deterministically."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    env_version: str
    task_seed: int
    artifact_class: str
    actions: tuple[Action, ...]
