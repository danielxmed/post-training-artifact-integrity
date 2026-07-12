"""Corruption operator interface — the (P,Q,I,O,S,R) tuple made concrete.

Operators mutate the JSONL at line level and declare, in the emitted
:class:`~ptaie.kernel.task.DefectNode`: which hidden checks are expected to
fire (Q), which contract invariants they violate (I), an observability
profile (O), severity and side effects (S), and certified/alternative repair
refs (R — registry keys into ``repairs.py``). Preconditions (P) are the
``eligible_line_indexes`` filter.

Composition applies operators sequentially in a canonical order (schema
corruption last, because a schema-broken record *masks* layer-3/4 symptoms
on the same record — those checks report ``blocked``). Masking edges are
derived from a static rule, not discovered dynamically.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ptaie.kernel.rng import DerivedRng
from ptaie.kernel.task import DefectNode
from ptaie.plugins.sft_chat.contract import SftChatContract
from ptaie.plugins.sft_chat.schema import ChatRecord, parse_record_line

_HEX = "0123456789abcdef"


def new_defect_id(rng: DerivedRng) -> str:
    return "def-" + "".join(rng.choice(_HEX) for _ in range(8))


def try_parse(line: str) -> ChatRecord | None:
    try:
        return parse_record_line(line)
    except ValueError:
        return None


@dataclass(frozen=True)
class OperatorResult:
    lines: list[str]
    node: DefectNode


class CorruptionOperator(ABC):
    """One typed corruption. Subclasses are stateless; all randomness comes
    from the rng handed to :meth:`apply`."""

    name: str
    version: int = 1
    # Checks on the same record that a defect from this operator blocks
    # (non-empty only for operators that break parsing/roles):
    masks_check_ids: tuple[str, ...] = ()

    @abstractmethod
    def eligible_line_indexes(self, lines: list[str], contract: SftChatContract) -> list[int]:
        """Preconditions (P): which lines this operator may corrupt."""

    @abstractmethod
    def apply(
        self,
        lines: list[str],
        contract: SftChatContract,
        rng: DerivedRng,
        defect_id: str,
        target_index: int | None = None,
    ) -> OperatorResult:
        """Corrupt and describe. Must only touch eligible lines. When
        ``target_index`` is given it is used instead of a random eligible
        line (used to compose two defects onto the same record)."""

    def _pick(
        self, lines: list[str], contract: SftChatContract, rng: DerivedRng, target_index: int | None
    ) -> int:
        eligible = self.eligible_line_indexes(lines, contract)
        if target_index is not None:
            if target_index not in eligible:
                raise ValueError(f"{self.name}: forced target {target_index} is not eligible")
            return target_index
        return rng.choice(eligible)


def compose(
    operators: list[CorruptionOperator],
    lines: list[str],
    contract: SftChatContract,
    rng: DerivedRng,
) -> tuple[list[str], list[DefectNode]]:
    """Apply operators to *distinct* records so they cannot interfere.

    Independent operators sharing a record would destroy each other's defect
    (e.g. corrupting one copy of a duplicate pair makes them unequal) or make
    a repair's exact-inverse predicate unsatisfiable. Deliberate masking
    pairs (schema breakage over another defect on the same record) are
    constructed explicitly elsewhere, not here, so this path keeps targets
    disjoint. An operator with no non-conflicting target is skipped.
    """
    nodes: list[DefectNode] = []
    current = list(lines)
    used_ids: set[str] = set()
    for index, operator in enumerate(operators):
        eligible = [
            idx
            for idx in operator.eligible_line_indexes(current, contract)
            if _line_id(current[idx]) not in used_ids
        ]
        if not eligible:
            continue
        target = rng.substream(f"pick-{index}").choice(eligible)
        result = operator.apply(
            current,
            contract,
            rng.substream(f"op-{index}"),
            new_defect_id(rng.substream(f"id-{index}")),
            target,
        )
        current = result.lines
        nodes.append(result.node)
        used_ids.update(result.node.affected_ids)
    return current, nodes


def _line_id(line: str) -> str | None:
    record = try_parse(line)
    return record.id if record is not None else None
