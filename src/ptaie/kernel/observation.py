"""The agent-facing observation protocol.

Observations are built by explicit constructors from whitelisted data — never
by filtering hidden objects. Terminal observations carry the terminal code
only, never hidden verifier verdict contents. Tool failures are deterministic
values (:class:`ToolOutcome`), never exceptions, and their messages come from
fixed templates that must not mention hidden fixtures, verifier internals, or
latent state (leakage-tested).
"""

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ptaie.kernel.actions import ActionCategory
from ptaie.kernel.budgets import BudgetView
from ptaie.kernel.canonical import JsonValue
from ptaie.kernel.ids import AuditRef
from ptaie.kernel.rewards import TerminalCode


class ToolErrorCode(StrEnum):
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    PATH_OUT_OF_SCOPE = "path_out_of_scope"
    FILE_NOT_FOUND = "file_not_found"
    PRECONDITION_FAILED = "precondition_failed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    INTERNAL_ERROR = "internal_error"


class ToolError(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    code: ToolErrorCode
    message: str
    details: dict[str, JsonValue] = {}


class ToolOutcome(BaseModel):
    """Deterministic result of one tool call: ok with output, or a typed error."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["ok", "error"]
    output: dict[str, JsonValue] | None = None
    error: ToolError | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.status == "ok" and (self.output is None or self.error is not None):
            raise ValueError("ok outcomes carry output and no error")
        if self.status == "error" and (self.error is None or self.output is not None):
            raise ValueError("error outcomes carry an error and no output")
        return self


class ToolMenuEntry(BaseModel):
    """One tool the agent may call, as presented in the task brief."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    category: ActionCategory
    description: str
    input_schema: dict[str, JsonValue] = {}


class TaskBriefPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["task_brief"] = "task_brief"
    brief: str
    files: tuple[str, ...]
    tool_menu: tuple[ToolMenuEntry, ...] = ()
    visible_validator_names: tuple[str, ...] = ()
    claimable_invariants: tuple[str, ...] = ()
    irreversible_boundaries: tuple[str, ...] = ()


class ToolResultPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["tool_result"] = "tool_result"
    tool_name: str
    outcome: ToolOutcome
    audit_ref: AuditRef


class ClarificationPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["clarification"] = "clarification"
    question_key: str | None
    answer_kind: Literal["value", "unknown", "already_stated", "budget_exhausted"]
    answer_value: str | None = None
    audit_ref: AuditRef


class ReportAckPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["report_ack"] = "report_ack"
    audit_ref: AuditRef


class TerminalPayload(BaseModel):
    """Episode end. Carries the terminal code only — hidden verification
    verdicts are never observed by the agent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["terminal"] = "terminal"
    terminal_code: TerminalCode


type ObservationPayload = Annotated[
    TaskBriefPayload
    | ToolResultPayload
    | ClarificationPayload
    | ReportAckPayload
    | TerminalPayload,
    Field(discriminator="kind"),
]


class Observation(BaseModel):
    """One agent-visible observation.

    ``staged_root`` is the current staged workspace hash — the agent-visible
    provenance anchor. ``notices`` are deterministic engine notices (budget
    warnings and the like), never hidden information.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    step_index: Annotated[int, Field(ge=0)]
    payload: ObservationPayload
    budget: BudgetView
    staged_root: str
    notices: tuple[str, ...] = ()
