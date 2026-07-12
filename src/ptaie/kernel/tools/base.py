"""Tool specification, capability context, and the ``Tool`` protocol.

``ToolContext`` is the *only* capability object handed to a tool. It holds a
workspace handle, a per-call rng substream, a read-only budget view, and a
constraint sink — and deliberately **no** reference to the hidden vault, the
task record, or the engine, so a tool cannot reach hidden state (guardrail-
tested). Mutation is gated on ``spec.transactional``.
"""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from ptaie.kernel.actions import ActionCategory
from ptaie.kernel.budgets import BudgetCharge, BudgetView
from ptaie.kernel.canonical import JsonValue
from ptaie.kernel.rng import DerivedRng
from ptaie.kernel.store.manifest import validate_workspace_path
from ptaie.kernel.store.workspace import Workspace, WorkspaceReadView


class ToolPreconditionError(Exception):
    """A tool's precondition failed (deterministic PRECONDITION_FAILED)."""

    def __init__(self, message: str, details: dict[str, JsonValue] | None = None) -> None:
        self.message = message
        self.details: dict[str, JsonValue] = details or {}
        super().__init__(message)


class ToolNotFoundError(Exception):
    """A tool referenced a workspace path that does not exist (FILE_NOT_FOUND)."""

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(path)


class ToolSpec(BaseModel):
    """Static declaration of one tool."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    name: str
    category: ActionCategory
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    charge: BudgetCharge
    transactional: bool
    description: str


class ConstraintSink:
    """Collects constraint flags raised during one tool call. The engine
    reads and latches them after execution — a tool can flag, never clear."""

    __slots__ = ("_flags",)

    _VALID = frozenset({"tamper", "escape", "prohibited_info_loss", "irreversible", "hard_policy"})

    def __init__(self) -> None:
        self._flags: set[str] = set()

    def flag(self, name: str) -> None:
        if name not in self._VALID:
            raise ValueError(f"unknown constraint flag {name!r}")
        self._flags.add(name)

    def flags(self) -> frozenset[str]:
        return frozenset(self._flags)


class ToolContext:
    """Capability object for one tool call."""

    __slots__ = ("_can_mutate", "_workspace", "budget", "constraints", "rng")

    def __init__(
        self,
        workspace: Workspace,
        rng: DerivedRng,
        budget: BudgetView,
        constraints: ConstraintSink,
        *,
        can_mutate: bool,
    ) -> None:
        self._workspace = workspace
        self._can_mutate = can_mutate
        self.rng = rng
        self.budget = budget
        self.constraints = constraints

    # read side (all tools)
    def paths(self) -> tuple[str, ...]:
        return self._workspace.manifest.paths()

    def has(self, path: str) -> bool:
        return self._workspace.read_view().has(path)

    def read(self, path: str) -> bytes:
        return self._workspace.read_view().read(path)

    def read_view(self) -> WorkspaceReadView:
        """A read-only view of the current staged workspace (for validators)."""
        return self._workspace.read_view()

    @property
    def staged_root(self) -> str:
        return self._workspace.staged

    def snapshot(self, label: str) -> str:
        return self._workspace.snapshot(label)

    def rollback(self, snapshot_id: str) -> str:
        if not self._can_mutate:
            raise PermissionError("rollback is only available to transactional tools")
        return self._workspace.rollback(snapshot_id)

    def diff_roots(self, root_a: str, root_b: str) -> object:
        return self._workspace.diff_roots(root_a, root_b)

    # write side (transactional tools only)
    def write(self, path: str, data: bytes, media_type: str | None = None) -> str:
        if not self._can_mutate:
            raise PermissionError(f"tool cannot write (non-transactional): {path!r}")
        validate_workspace_path(path)
        if media_type is None:
            return self._workspace.write(path, data)
        return self._workspace.write(path, data, media_type)

    def delete(self, path: str) -> str:
        if not self._can_mutate:
            raise PermissionError(f"tool cannot delete (non-transactional): {path!r}")
        return self._workspace.delete(path)


@runtime_checkable
class Tool(Protocol):
    spec: ToolSpec

    def execute(self, ctx: ToolContext, args: BaseModel) -> BaseModel: ...
