"""The six M1 corruption operators for sft_chat.

Each declares the hidden checks it is expected to fire and the accepted /
prohibited repair predicates, so generation acceptance can prove the defect
is detectable, repairable, and not over-repairable.
"""

import json

from ptaie.kernel.rng import DerivedRng
from ptaie.kernel.task import DefectNode
from ptaie.plugins.sft_chat import checks
from ptaie.plugins.sft_chat.contract import SftChatContract
from ptaie.plugins.sft_chat.corruption.base import CorruptionOperator, OperatorResult, try_parse
from ptaie.plugins.sft_chat.schema import ChatRecord, parse_record_line, serialize_record


def _record_id(line: str) -> str | None:
    record = try_parse(line)
    return record.id if record is not None else None


def _replace(lines: list[str], index: int, record: ChatRecord) -> list[str]:
    out = list(lines)
    out[index] = serialize_record(record)
    return out


class RoleOrderViolation(CorruptionOperator):
    name = "RoleOrderViolation"

    def eligible_line_indexes(self, lines: list[str], contract: SftChatContract) -> list[int]:
        eligible = []
        for index, line in enumerate(lines):
            record = try_parse(line)
            if record is None:
                continue
            if any(m.role == "user" for m in record.messages):
                eligible.append(index)
        return eligible

    def apply(
        self,
        lines: list[str],
        contract: SftChatContract,
        rng: DerivedRng,
        defect_id: str,
        target_index: int | None = None,
    ) -> OperatorResult:
        index = self._pick(lines, contract, rng, target_index)
        record = parse_record_line(lines[index])
        # Duplicate the first user message so two identical user turns are
        # consecutive: a rich equivalence class (delete-dup or merge both fix).
        user_pos = next(i for i, m in enumerate(record.messages) if m.role == "user")
        messages = list(record.messages)
        messages.insert(user_pos + 1, messages[user_pos])
        corrupted = record.model_copy(update={"messages": tuple(messages)})
        node = DefectNode(
            defect_id=defect_id,
            operator=self.name,
            operator_version=self.version,
            params={"record_id": record.id},
            expected_failures=(checks.ROLE_ALTERNATION,),
            violated_invariants=("role_alternation",),
            affected_ids=(record.id,),
            observability="inspection_visible",
            severity="semantic",
            accepted_postconditions=(checks.ROLE_ALTERNATION,),
            prohibited_side_effects=(checks.SE_FABRICATED_CONTENT,),
            certified_repair_ref="role_order_delete_dup",
            alternative_repair_refs=("role_order_merge",),
        )
        return OperatorResult(lines=_replace(lines, index, corrupted), node=node)


class TruncatedFinalAssistant(CorruptionOperator):
    name = "TruncatedFinalAssistant"

    def eligible_line_indexes(self, lines: list[str], contract: SftChatContract) -> list[int]:
        eligible = []
        for index, line in enumerate(lines):
            record = try_parse(line)
            if record is None:
                continue
            assistant_count = sum(1 for m in record.messages if m.role == "assistant")
            if assistant_count >= 2 and record.messages[-1].role == "assistant":
                eligible.append(index)
        return eligible

    def apply(
        self,
        lines: list[str],
        contract: SftChatContract,
        rng: DerivedRng,
        defect_id: str,
        target_index: int | None = None,
    ) -> OperatorResult:
        index = self._pick(lines, contract, rng, target_index)
        record = parse_record_line(lines[index])
        messages = list(record.messages)
        if rng.random() < 0.5:
            # variant a: cut final assistant content mid-sentence
            final = messages[-1]
            cut = final.content.rstrip(".!?")[: max(4, len(final.content) // 2)].rstrip()
            messages[-1] = final.model_copy(update={"content": cut + " and"})
        else:
            # variant b: delete the final assistant message (record ends on user)
            messages.pop()
        corrupted = record.model_copy(update={"messages": tuple(messages)})
        node = DefectNode(
            defect_id=defect_id,
            operator=self.name,
            operator_version=self.version,
            params={"record_id": record.id},
            expected_failures=(checks.FINAL_TURN,),
            violated_invariants=("final_turn",),
            affected_ids=(record.id,),
            observability="hidden_only",
            severity="training_breaking",
            accepted_postconditions=(checks.FINAL_TURN,),
            prohibited_side_effects=(checks.SE_FABRICATED_CONTENT,),
            # The lost content is unrecoverable from the agent's view; the
            # accepted repair is a structural trim, not restoration.
            certified_repair_ref="truncation_trim",
            alternative_repair_refs=(),
        )
        return OperatorResult(lines=_replace(lines, index, corrupted), node=node)


class LossMaskMismatch(CorruptionOperator):
    name = "LossMaskMismatch"

    def eligible_line_indexes(self, lines: list[str], contract: SftChatContract) -> list[int]:
        return [i for i, line in enumerate(lines) if try_parse(line) is not None]

    def apply(
        self,
        lines: list[str],
        contract: SftChatContract,
        rng: DerivedRng,
        defect_id: str,
        target_index: int | None = None,
    ) -> OperatorResult:
        index = self._pick(lines, contract, rng, target_index)
        record = parse_record_line(lines[index])
        messages = list(record.messages)
        if contract.mask_convention == "implicit_assistant":
            # inject a spurious inconsistent flag on a user turn
            target = next((i for i, m in enumerate(messages) if m.role == "user"), 0)
            messages[target] = messages[target].model_copy(update={"loss": True})
        else:
            # flip one assistant flag to the wrong value
            target = next((i for i, m in enumerate(messages) if m.role == "assistant"), 0)
            current = messages[target].loss
            messages[target] = messages[target].model_copy(update={"loss": not bool(current)})
        corrupted = record.model_copy(update={"messages": tuple(messages)})
        node = DefectNode(
            defect_id=defect_id,
            operator=self.name,
            operator_version=self.version,
            params={"record_id": record.id},
            expected_failures=(checks.MASK_CONSISTENCY,),
            violated_invariants=("mask_convention",),
            affected_ids=(record.id,),
            observability="hidden_only",
            severity="training_breaking",
            accepted_postconditions=(checks.MASK_CONSISTENCY,),
            prohibited_side_effects=(checks.SE_FABRICATED_CONTENT,),
            certified_repair_ref="mask_set_correct",
            alternative_repair_refs=(
                ("mask_strip",) if contract.mask_convention == "implicit_assistant" else ()
            ),
        )
        return OperatorResult(lines=_replace(lines, index, corrupted), node=node)


class DuplicateRecords(CorruptionOperator):
    name = "DuplicateRecords"

    def eligible_line_indexes(self, lines: list[str], contract: SftChatContract) -> list[int]:
        if contract.dedup_policy != "exact_dups_forbidden":
            return []
        return [i for i, line in enumerate(lines) if try_parse(line) is not None]

    def apply(
        self,
        lines: list[str],
        contract: SftChatContract,
        rng: DerivedRng,
        defect_id: str,
        target_index: int | None = None,
    ) -> OperatorResult:
        index = self._pick(lines, contract, rng, target_index)
        record = parse_record_line(lines[index])
        # fresh-id content duplicate (append at end)
        clone_id = record.id + "-dup"
        clone = record.model_copy(update={"id": clone_id})
        out = [*lines, serialize_record(clone)]
        node = DefectNode(
            defect_id=defect_id,
            operator=self.name,
            operator_version=self.version,
            params={"source_id": record.id, "clone_id": clone_id},
            expected_failures=(checks.DEDUP,),
            violated_invariants=("dedup_policy",),
            affected_ids=(record.id, clone_id),
            observability="hidden_only",
            severity="semantic",
            accepted_postconditions=(checks.DEDUP,),
            prohibited_side_effects=(checks.SE_FABRICATED_CONTENT,),
            certified_repair_ref="dedup_drop_clone",
            alternative_repair_refs=("dedup_keep_first",),
        )
        return OperatorResult(lines=out, node=node)


class MojibakeEncoding(CorruptionOperator):
    name = "MojibakeEncoding"

    def eligible_line_indexes(self, lines: list[str], contract: SftChatContract) -> list[int]:
        eligible = []
        for index, line in enumerate(lines):
            record = try_parse(line)
            if record is None:
                continue
            if any(_has_double_encodable(m.content) for m in record.messages):
                eligible.append(index)
        return eligible

    def apply(
        self,
        lines: list[str],
        contract: SftChatContract,
        rng: DerivedRng,
        defect_id: str,
        target_index: int | None = None,
    ) -> OperatorResult:
        index = self._pick(lines, contract, rng, target_index)
        record = parse_record_line(lines[index])
        messages = tuple(
            m.model_copy(update={"content": _mojibake(m.content)})
            if _has_double_encodable(m.content)
            else m
            for m in record.messages
        )
        corrupted = record.model_copy(update={"messages": messages})
        node = DefectNode(
            defect_id=defect_id,
            operator=self.name,
            operator_version=self.version,
            params={"record_id": record.id},
            expected_failures=(checks.ENCODING,),
            violated_invariants=("encoding_policy",),
            affected_ids=(record.id,),
            observability="hidden_only",
            severity="semantic",
            accepted_postconditions=(checks.ENCODING,),
            prohibited_side_effects=(checks.SE_DEL_PROTECTED, checks.SE_MOJIBAKE_EXACT),
            certified_repair_ref="mojibake_invert",
            alternative_repair_refs=(),
        )
        return OperatorResult(lines=_replace(lines, index, corrupted), node=node)


class SchemaFieldCorruption(CorruptionOperator):
    name = "SchemaFieldCorruption"
    # Breaking the schema blocks every per-record local check on that record.
    masks_check_ids = checks.PER_RECORD_LOCAL_CHECKS

    def eligible_line_indexes(self, lines: list[str], contract: SftChatContract) -> list[int]:
        return [i for i, line in enumerate(lines) if try_parse(line) is not None]

    def apply(
        self,
        lines: list[str],
        contract: SftChatContract,
        rng: DerivedRng,
        defect_id: str,
        target_index: int | None = None,
    ) -> OperatorResult:
        index = self._pick(lines, contract, rng, target_index)
        record = parse_record_line(lines[index])
        # Both variants make valid JSON that is not a valid ChatRecord (an
        # unknown role would still be a valid string, so it is a layer-3
        # concern, not a schema break — hence it is not used here).
        variant = rng.choice(("rename_messages", "wrap_content"))
        raw = json.loads(lines[index])
        if variant == "rename_messages":
            raw["turns"] = raw.pop("messages")
        else:  # wrap_content
            raw["messages"][0]["content"] = [raw["messages"][0]["content"]]
        corrupted_line = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        out = list(lines)
        out[index] = corrupted_line
        node = DefectNode(
            defect_id=defect_id,
            operator=self.name,
            operator_version=self.version,
            params={"record_id": record.id, "variant": variant},
            expected_failures=(checks.SCHEMA,),
            violated_invariants=("unknown_fields", "schema"),
            affected_ids=(record.id,),
            observability="public_validator_visible",
            severity="training_breaking",
            accepted_postconditions=(checks.SCHEMA,),
            prohibited_side_effects=(checks.SE_FABRICATED_CONTENT,),
            certified_repair_ref="schema_fix",
            alternative_repair_refs=(),
        )
        return OperatorResult(lines=out, node=node)


def _has_double_encodable(text: str) -> bool:
    return any(ord(ch) > 0x7F for ch in text)


def _mojibake(text: str) -> str:
    """Deterministic, exactly-invertible UTF-8 -> latin-1 double encode."""
    return text.encode("utf-8").decode("latin-1")


ALL_OPERATORS: tuple[type[CorruptionOperator], ...] = (
    RoleOrderViolation,
    TruncatedFinalAssistant,
    LossMaskMismatch,
    DuplicateRecords,
    MojibakeEncoding,
    SchemaFieldCorruption,
)

OPERATORS_BY_NAME: dict[str, type[CorruptionOperator]] = {op.name: op for op in ALL_OPERATORS}
