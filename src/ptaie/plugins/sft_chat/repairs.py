"""Certified and alternative repairs, keyed by the refs operators declare.

Each repair is a pure line-level transform ``(lines, node, contract) ->
lines`` (line level so schema-broken records stay editable). Repairs never
touch the card; a repair sequence regenerates the card afterward (see
``apply_repairs``), mirroring how an agent edits records then calls
``regenerate_card``.

These are used two ways: generation acceptance applies them to prove
repairability, and the oracle policy (a later PR) drives the equivalent tool
actions to the same target state. Neither is the acceptance criterion — the
sole gate is the equivalence predicate in ``equivalence.py``.
"""

import json
from collections.abc import Callable

from ptaie.kernel.task import DefectNode
from ptaie.plugins.sft_chat.bundle import build_card
from ptaie.plugins.sft_chat.contract import SftChatContract
from ptaie.plugins.sft_chat.schema import (
    ChatMessage,
    ChatRecord,
    normalized_content_key,
    parse_card,
    parse_record_line,
    serialize_card,
    serialize_dataset,
    serialize_record,
    split_data_lines,
)

RepairFn = Callable[[list[str], DefectNode, SftChatContract], list[str]]


def _find_index(lines: list[str], record_id: str) -> int | None:
    for index, line in enumerate(lines):
        try:
            record = parse_record_line(line)
        except ValueError:
            continue
        if record.id == record_id:
            return index
    return None


def _implied_loss(role: str, is_final_assistant: bool, contract: SftChatContract) -> bool:
    if contract.mask_convention == "explicit_final_assistant_only":
        return role == "assistant" and is_final_assistant
    return role == "assistant"


def role_order_delete_dup(
    lines: list[str], node: DefectNode, contract: SftChatContract
) -> list[str]:
    index = _find_index(lines, str(node.params["record_id"]))
    if index is None:
        return lines
    record = parse_record_line(lines[index])
    messages = list(record.messages)
    deduped: list[ChatMessage] = []
    for message in messages:
        if (
            deduped
            and deduped[-1].role == message.role == "user"
            and deduped[-1].content == message.content
        ):
            continue
        deduped.append(message)
    out = list(lines)
    out[index] = serialize_record(record.model_copy(update={"messages": tuple(deduped)}))
    return out


def role_order_merge(lines: list[str], node: DefectNode, contract: SftChatContract) -> list[str]:
    index = _find_index(lines, str(node.params["record_id"]))
    if index is None:
        return lines
    record = parse_record_line(lines[index])
    messages = list(record.messages)
    merged: list[ChatMessage] = []
    for message in messages:
        if merged and merged[-1].role == message.role == "user":
            if merged[-1].content == message.content:
                continue  # identical duplicate: collapse
            merged[-1] = merged[-1].model_copy(
                update={"content": merged[-1].content + "\n" + message.content}
            )
            continue
        merged.append(message)
    out = list(lines)
    out[index] = serialize_record(record.model_copy(update={"messages": tuple(merged)}))
    return out


def _normalize_mask(messages: list[ChatMessage], contract: SftChatContract) -> list[ChatMessage]:
    """Set loss flags to the convention-correct values for these messages.

    Trimming changes which assistant turn is final, so the mask must be
    recomputed or an ``explicit_final_assistant_only`` record ends up
    inconsistent.
    """
    final_assistant = max((i for i, m in enumerate(messages) if m.role == "assistant"), default=-1)
    if contract.mask_convention == "implicit_assistant":
        return [m.model_copy(update={"loss": None}) for m in messages]
    return [
        m.model_copy(update={"loss": _implied_loss(m.role, i == final_assistant, contract)})
        for i, m in enumerate(messages)
    ]


def truncation_trim(lines: list[str], node: DefectNode, contract: SftChatContract) -> list[str]:
    index = _find_index(lines, str(node.params["record_id"]))
    if index is None:
        return lines
    record = parse_record_line(lines[index])
    messages = list(record.messages)
    # Drop trailing turns until the record ends on a complete assistant turn.
    while messages:
        final = messages[-1]
        if final.role == "assistant":
            stripped = final.content.strip()
            if stripped and stripped[-1] in (".", "!", "?"):
                break
        messages.pop()
    # Re-normalize the mask: the final assistant turn has changed.
    messages = _normalize_mask(messages, contract)
    out = list(lines)
    out[index] = serialize_record(record.model_copy(update={"messages": tuple(messages)}))
    return out


def mask_set_correct(lines: list[str], node: DefectNode, contract: SftChatContract) -> list[str]:
    index = _find_index(lines, str(node.params["record_id"]))
    if index is None:
        return lines
    record = parse_record_line(lines[index])
    final_assistant = max(
        (i for i, m in enumerate(record.messages) if m.role == "assistant"), default=-1
    )
    if contract.mask_convention == "implicit_assistant":
        messages = tuple(m.model_copy(update={"loss": None}) for m in record.messages)
    else:
        messages = tuple(
            m.model_copy(update={"loss": _implied_loss(m.role, i == final_assistant, contract)})
            for i, m in enumerate(record.messages)
        )
    out = list(lines)
    out[index] = serialize_record(record.model_copy(update={"messages": messages}))
    return out


def mask_strip(lines: list[str], node: DefectNode, contract: SftChatContract) -> list[str]:
    index = _find_index(lines, str(node.params["record_id"]))
    if index is None:
        return lines
    record = parse_record_line(lines[index])
    messages = tuple(m.model_copy(update={"loss": None}) for m in record.messages)
    out = list(lines)
    out[index] = serialize_record(record.model_copy(update={"messages": messages}))
    return out


def _drop_duplicates(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        try:
            record = parse_record_line(line)
        except ValueError:
            out.append(line)
            continue
        key = normalized_content_key(record)
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


def dedup_drop_clone(lines: list[str], node: DefectNode, contract: SftChatContract) -> list[str]:
    return _drop_duplicates(lines)


def dedup_keep_first(lines: list[str], node: DefectNode, contract: SftChatContract) -> list[str]:
    return _drop_duplicates(lines)


def _invert_mojibake(text: str) -> str:
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def mojibake_invert(lines: list[str], node: DefectNode, contract: SftChatContract) -> list[str]:
    index = _find_index(lines, str(node.params["record_id"]))
    if index is None:
        return lines
    record = parse_record_line(lines[index])
    messages = tuple(
        m.model_copy(update={"content": _invert_mojibake(m.content)}) for m in record.messages
    )
    out = list(lines)
    out[index] = serialize_record(record.model_copy(update={"messages": messages}))
    return out


def schema_fix(lines: list[str], node: DefectNode, contract: SftChatContract) -> list[str]:
    # The schema-broken record can't be parsed; operate on raw JSON, locating
    # it by the affected id recorded on the node.
    target_id = node.params["record_id"]
    out = list(lines)
    for index, line in enumerate(lines):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, dict):
            continue
        if "turns" in raw and "messages" not in raw:
            raw["messages"] = raw.pop("turns")
        if raw.get("id") != target_id:
            continue
        for message in raw.get("messages", []):
            content = message.get("content")
            if isinstance(content, list):
                message["content"] = content[0] if content else ""
        record = ChatRecord.model_validate(raw)
        out[index] = serialize_record(record)
        return out
    return out


REPAIRS: dict[str, RepairFn] = {
    "role_order_delete_dup": role_order_delete_dup,
    "role_order_merge": role_order_merge,
    "truncation_trim": truncation_trim,
    "mask_set_correct": mask_set_correct,
    "mask_strip": mask_strip,
    "dedup_drop_clone": dedup_drop_clone,
    "dedup_keep_first": dedup_keep_first,
    "mojibake_invert": mojibake_invert,
    "schema_fix": schema_fix,
}


def regenerate_card(lines: list[str], old_card_bytes: bytes, contract: SftChatContract) -> bytes:
    """Recompute record_count and data_sha256 from the current data, keeping
    the declared block and notes. This is honest (the data really changed);
    laundering would instead recompute against unrepaired data."""
    records = [parse_record_line(line) for line in lines]
    data_bytes = serialize_dataset(records)
    try:
        old = parse_card(old_card_bytes)
        notes = old.notes
        declared = old.declared
    except (ValueError, UnicodeDecodeError):
        notes = ()
        declared = contract.surfaced_declarations()
    card = build_card(records, data_bytes, contract, notes)
    # Preserve the original declared block rather than re-deriving it.
    return serialize_card(card.model_copy(update={"declared": declared}))


def apply_repairs(
    data_bytes: bytes,
    card_bytes: bytes,
    nodes: list[DefectNode],
    contract: SftChatContract,
    *,
    use_alternatives: bool = False,
) -> tuple[bytes, bytes]:
    """Apply each node's repair, schema fixes first (to unmask), then
    regenerate the card. Returns (data_bytes, card_bytes)."""
    lines = split_data_lines(data_bytes)
    ordered = sorted(nodes, key=lambda n: 0 if n.operator == "SchemaFieldCorruption" else 1)
    for node in ordered:
        refs = (
            node.alternative_repair_refs
            if use_alternatives and node.alternative_repair_refs
            else (node.certified_repair_ref,)
        )
        repair = REPAIRS[refs[0]]
        lines = repair(lines, node, contract)
    new_card = regenerate_card(lines, card_bytes, contract)
    records = [parse_record_line(line) for line in lines]
    return serialize_dataset(records), new_card
