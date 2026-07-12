"""The per-episode tool registry and deterministic dispatch.

Dispatch never lets an exception cross to the agent: argument-validation
failures, path violations, and unexpected errors all become typed
``ToolOutcome`` error values with stable codes. Error messages come from fixed
templates and must never mention hidden fixtures, verifier internals, or
latent state.
"""

import json

from pydantic import BaseModel, ValidationError

from ptaie.kernel.actions import ToolAction
from ptaie.kernel.canonical import JsonValue
from ptaie.kernel.errors import PathViolationError, RegistryError
from ptaie.kernel.observation import ToolError, ToolErrorCode, ToolMenuEntry, ToolOutcome
from ptaie.kernel.tools.base import (
    Tool,
    ToolContext,
    ToolNotFoundError,
    ToolPreconditionError,
)


class ToolRegistry:
    __slots__ = ("_tools",)

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        spec = tool.spec
        if spec.name in self._tools:
            raise RegistryError(f"duplicate tool name {spec.name!r}")
        if spec.category.value == "mutate" and not spec.transactional:
            raise RegistryError(f"mutate tool {spec.name!r} must be transactional")
        self._tools[spec.name] = tool

    def register_all(self, tools: object) -> None:
        for tool in tools:  # type: ignore[attr-defined]
            self.register(tool)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> tuple[ToolMenuEntry, ...]:
        return tuple(
            ToolMenuEntry(
                name=tool.spec.name,
                category=tool.spec.category,
                description=tool.spec.description,
                input_schema=_schema(tool.spec.input_model),
            )
            for _, tool in sorted(self._tools.items())
        )

    def dispatch(self, action: ToolAction, ctx: ToolContext) -> ToolOutcome:
        tool = self._tools.get(action.tool_name)
        if tool is None:
            return _err(ToolErrorCode.UNKNOWN_TOOL, "no such tool", {"tool": action.tool_name})
        try:
            args = tool.spec.input_model.model_validate(action.arguments)
        except ValidationError as exc:
            return _err(
                ToolErrorCode.INVALID_ARGUMENTS,
                "arguments did not validate",
                {"errors": [e["type"] + ":" + ".".join(map(str, e["loc"])) for e in exc.errors()]},
            )
        try:
            output = tool.execute(ctx, args)
        except PathViolationError:
            ctx.constraints.flag("escape")
            return _err(ToolErrorCode.PATH_OUT_OF_SCOPE, "path is out of scope", {})
        except ToolPreconditionError as exc:
            return _err(ToolErrorCode.PRECONDITION_FAILED, exc.message, exc.details)
        except ToolNotFoundError as exc:
            return _err(ToolErrorCode.FILE_NOT_FOUND, "path not found", {"path": exc.path})
        except Exception:
            return _err(ToolErrorCode.INTERNAL_ERROR, "internal tool error", {"engine_fault": True})
        return ToolOutcome(status="ok", output=_dump(output))


def _err(code: ToolErrorCode, message: str, details: dict[str, JsonValue]) -> ToolOutcome:
    return ToolOutcome(status="error", error=ToolError(code=code, message=message, details=details))


def _dump(output: BaseModel) -> dict[str, JsonValue]:
    dumped: dict[str, JsonValue] = json.loads(output.model_dump_json())
    return dumped


def _schema(model: type[BaseModel]) -> dict[str, JsonValue]:
    schema: dict[str, JsonValue] = json.loads(json.dumps(model.model_json_schema()))
    return schema
