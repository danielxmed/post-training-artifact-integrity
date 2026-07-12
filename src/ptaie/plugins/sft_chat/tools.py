"""sft_chat line-level record tools plus card regeneration.

Editing is at the raw-line level so a schema-broken record stays editable.
These coexist with the kernel's generic ``write_file``; they are the ergonomic
path an agent (or the oracle policy) uses to repair records.
"""

from typing import Annotated, cast

from pydantic import BaseModel, ConfigDict, Field

from ptaie.kernel.actions import ActionCategory
from ptaie.kernel.budgets import BudgetCharge
from ptaie.kernel.tools.base import Tool, ToolContext, ToolPreconditionError, ToolSpec
from ptaie.plugins.sft_chat import CARD_PATH, DATA_PATH
from ptaie.plugins.sft_chat.repairs import regenerate_card
from ptaie.plugins.sft_chat.schema import parse_record_line, split_data_lines


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _load_lines(ctx: ToolContext) -> list[str]:
    if not ctx.has(DATA_PATH):
        raise ToolPreconditionError("data file is missing", {"path": DATA_PATH})
    return split_data_lines(ctx.read(DATA_PATH))


def _write_lines(ctx: ToolContext, lines: list[str]) -> str:
    body = ("\n".join(lines) + "\n").encode("utf-8") if lines else b""
    return ctx.write(DATA_PATH, body, media_type="application/jsonl")


class ReplaceRecordArgs(_Model):
    line_index: Annotated[int, Field(ge=0)]
    raw_line: str


class ReplaceRecordOut(_Model):
    root: str
    line_count: int


class ReplaceRecord:
    spec = ToolSpec(
        name="sft_chat.replace_record",
        category=ActionCategory.MUTATE,
        input_model=ReplaceRecordArgs,
        output_model=ReplaceRecordOut,
        charge=BudgetCharge(cost_units=2, mutations=1),
        transactional=True,
        description="Replace the JSONL line at line_index with raw_line (a JSON object).",
    )

    def execute(self, ctx: ToolContext, args: BaseModel) -> BaseModel:
        args = cast(ReplaceRecordArgs, args)
        lines = _load_lines(ctx)
        if args.line_index >= len(lines):
            raise ToolPreconditionError("line_index out of range", {"line_count": len(lines)})
        if "\n" in args.raw_line:
            raise ToolPreconditionError("raw_line must be a single line", {})
        lines[args.line_index] = args.raw_line
        root = _write_lines(ctx, lines)
        return ReplaceRecordOut(root=root, line_count=len(lines))


class DeleteRecordArgs(_Model):
    line_index: Annotated[int, Field(ge=0)]


class DeleteRecordOut(_Model):
    root: str
    line_count: int


class DeleteRecord:
    spec = ToolSpec(
        name="sft_chat.delete_record",
        category=ActionCategory.MUTATE,
        input_model=DeleteRecordArgs,
        output_model=DeleteRecordOut,
        charge=BudgetCharge(cost_units=2, mutations=1),
        transactional=True,
        description="Delete the JSONL line at line_index.",
    )

    def execute(self, ctx: ToolContext, args: BaseModel) -> BaseModel:
        args = cast(DeleteRecordArgs, args)
        lines = _load_lines(ctx)
        if args.line_index >= len(lines):
            raise ToolPreconditionError("line_index out of range", {"line_count": len(lines)})
        del lines[args.line_index]
        root = _write_lines(ctx, lines)
        return DeleteRecordOut(root=root, line_count=len(lines))


class InsertRecordArgs(_Model):
    line_index: Annotated[int, Field(ge=0)]
    raw_line: str


class InsertRecordOut(_Model):
    root: str
    line_count: int


class InsertRecord:
    spec = ToolSpec(
        name="sft_chat.insert_record",
        category=ActionCategory.MUTATE,
        input_model=InsertRecordArgs,
        output_model=InsertRecordOut,
        charge=BudgetCharge(cost_units=2, mutations=1),
        transactional=True,
        description="Insert raw_line as a new JSONL line before line_index (append if at the end).",
    )

    def execute(self, ctx: ToolContext, args: BaseModel) -> BaseModel:
        args = cast(InsertRecordArgs, args)
        lines = _load_lines(ctx)
        if args.line_index > len(lines):
            raise ToolPreconditionError("line_index out of range", {"line_count": len(lines)})
        if "\n" in args.raw_line:
            raise ToolPreconditionError("raw_line must be a single line", {})
        lines.insert(args.line_index, args.raw_line)
        root = _write_lines(ctx, lines)
        return InsertRecordOut(root=root, line_count=len(lines))


class RegenerateCardArgs(_Model):
    pass


class RegenerateCardOut(_Model):
    root: str


class RegenerateCard:
    spec = ToolSpec(
        name="sft_chat.regenerate_card",
        category=ActionCategory.MUTATE,
        input_model=RegenerateCardArgs,
        output_model=RegenerateCardOut,
        charge=BudgetCharge(cost_units=1, mutations=1),
        transactional=True,
        description="Recompute the dataset card's record_count and data_sha256 from the data.",
    )

    def execute(self, ctx: ToolContext, args: BaseModel) -> BaseModel:
        args = cast(RegenerateCardArgs, args)
        if not ctx.has(DATA_PATH) or not ctx.has(CARD_PATH):
            raise ToolPreconditionError("data or card file is missing", {})
        # The card can only be regenerated from parseable data; an unparseable
        # record is an agent-caused precondition failure (PRECONDITION_FAILED),
        # not an environment fault.
        for index, line in enumerate(split_data_lines(ctx.read(DATA_PATH))):
            try:
                parse_record_line(line)
            except ValueError:
                raise ToolPreconditionError(
                    "data has an unparseable record; fix it before regenerating the card",
                    {"line_index": index},
                ) from None
        # The card regeneration reuses the declared block from the current card
        # but recomputes count + hash from the current data. The contract is
        # not needed here (declared/notes are carried through), so a throwaway
        # placeholder is passed; regenerate_card only reads it on card-parse
        # failure, which does not occur for a well-formed card.
        from ptaie.plugins.sft_chat.contract import SftChatContract

        placeholder = SftChatContract(
            role_alternation="strict_user_assistant",
            system_policy="forbidden",
            mask_convention="implicit_assistant",
            dedup_policy="dups_allowed",
            dimension_status=dict.fromkeys(
                ("role_alternation", "system_policy", "mask_convention", "dedup_policy"), "surfaced"
            ),
        )
        lines = split_data_lines(ctx.read(DATA_PATH))
        new_card = regenerate_card(lines, ctx.read(CARD_PATH), placeholder)
        root = ctx.write(CARD_PATH, new_card, media_type="application/json")
        return RegenerateCardOut(root=root)


def sft_chat_tools() -> tuple[Tool, ...]:
    return (ReplaceRecord(), DeleteRecord(), InsertRecord(), RegenerateCard())
