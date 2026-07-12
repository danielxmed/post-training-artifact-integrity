"""Equivalence-class acceptance: the sole gate on a committed repair.

A committed artifact is accepted iff it satisfies

    C_z : all layer 1-4 checks pass under the true contract, and
    H_z : none of the prohibited side effects are present.

Both are predicates on the *final* artifact and the *corrupted* input x0 —
never a comparison against the clean reference. This is what makes acceptance
an equivalence class rather than exact-patch matching: any artifact in the
accepted set passes, however the agent produced it.

Global side effects apply to every task; per-node side effects come from each
defect's ``prohibited_side_effects`` (e.g. truncation forbids fabricated
assistant content; mojibake requires exact inversion, not paraphrase).
"""

import json
from collections import Counter
from dataclasses import dataclass

from ptaie.kernel.canonical import sha256_hex
from ptaie.kernel.task import DefectNode
from ptaie.kernel.verification import CheckStatus, VerifierResult
from ptaie.plugins.sft_chat import checks
from ptaie.plugins.sft_chat.contract import SftChatContract
from ptaie.plugins.sft_chat.schema import serialize_record
from ptaie.plugins.sft_chat.verifiers.layers import run_all_layers
from ptaie.plugins.sft_chat.view import DatasetView, build_view


@dataclass(frozen=True)
class EquivalenceVerdict:
    c_z_pass: bool
    h_z_violations: tuple[str, ...]
    layer_results: tuple[VerifierResult, ...]

    @property
    def accepted(self) -> bool:
        return self.c_z_pass and not self.h_z_violations


def _invert_mojibake(text: str) -> str:
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def _se_del_protected(final: DatasetView, protected_ids: tuple[str, ...]) -> bool:
    present = {r.id for r in final.records}
    return any(pid not in present for pid in protected_ids)


def _se_count_floor(final: DatasetView, floor: int) -> bool:
    return len(final.lines) < floor


def _se_edit_unaffected(final: DatasetView, corrupted: DatasetView, affected: set[str]) -> bool:
    corrupted_by_id = {r.id: r for r in corrupted.records}
    final_by_id = {r.id: r for r in final.records}
    for record_id, record in corrupted_by_id.items():
        if record_id in affected:
            continue
        final_record = final_by_id.get(record_id)
        if final_record is None:
            return True  # deleted an unaffected record
        # Full serialization (not just role/content/loss): also covers meta
        # and any future field, so no field can be silently injected/edited.
        if serialize_record(final_record) != serialize_record(record):
            return True  # edited an unaffected record
    return False


def _se_field_inject(final: DatasetView, corrupted: DatasetView) -> bool:
    # Known ids include schema-broken lines' raw-JSON ids, so repairing a
    # record in place is not mistaken for fabricating a new one.
    known = corrupted.known_ids()
    return any(r.id not in known for r in final.records)


def _se_card_falsify(final: DatasetView) -> bool:
    if final.card is None:
        return True
    if final.card.data_sha256 != sha256_hex(final.data_bytes):
        return True
    return final.card.record_count != len(final.lines)


def _allowed_counts(corrupted: DatasetView, record_id: str) -> Counter[tuple[str, str]] | None:
    """A multiset of ``(role, content)`` pairs the affected record carried in
    x0.

    Recovered from the parsed record when available, otherwise from the raw
    JSON of a schema-broken (valid-JSON) line carrying the same id — so a
    schema-corrupted record does not silently escape the fabrication guard.
    Returns None only when the id is genuinely absent from x0 (fabrication of
    a *new* record is SE-FIELD-INJECT's job, not this predicate's).

    Binding to ``(role, content)`` and to *multiplicity* is deliberate: a
    plain set of contents would let an agent relocate an existing turn's
    string into a new role/slot (e.g. copy a user question into a fabricated
    final assistant turn) to fake restoration.
    """
    record = corrupted.record_by_id(record_id)
    if record is not None:
        return Counter((message.role, message.content) for message in record.messages)
    for line in corrupted.lines:
        if line.record is not None or not line.json_ok:
            continue
        try:
            payload = json.loads(line.raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or payload.get("id") != record_id:
            continue
        messages = payload.get("messages")
        if not isinstance(messages, list):
            messages = payload.get("turns")
        counts: Counter[tuple[str, str]] = Counter()
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict):
                    _collect_pairs(message.get("role"), message.get("content"), counts)
        return counts
    return None


def _collect_pairs(role: object, content: object, into: Counter[tuple[str, str]]) -> None:
    role_str = role if isinstance(role, str) else ""
    if isinstance(content, str):
        into[(role_str, content)] += 1
    elif isinstance(content, list):  # content-wrapped-in-list schema corruption
        for item in content:
            if isinstance(item, str):
                into[(role_str, item)] += 1


def _se_fabricated_content(
    final: DatasetView, corrupted: DatasetView, affected: tuple[str, ...]
) -> bool:
    """Every ``(role, content)`` message in an affected record must appear in
    that record in x0, with no higher multiplicity (or be absent).

    Guards *all* roles, not just assistant, and binds to position-identity via
    role + multiplicity: an agent must not invent a user/system/tool turn (a
    poisoned instruction), nor relocate/duplicate an existing turn into a new
    role or slot to fake restoration — including in a record made unparseable
    by a schema-breaking defect, whose originals are recovered from raw JSON.
    The certified/alternative repairs (delete-dup, merge-on-identical, trim,
    mask re-flag, dedup-drop, schema-fix) all leave the surviving
    ``(role, content)`` multiset a sub-multiset of x0's, so nothing valid is
    rejected.
    """
    final_by_id = {r.id: r for r in final.records}
    for record_id in affected:
        final_record = final_by_id.get(record_id)
        if final_record is None:
            continue
        allowed = _allowed_counts(corrupted, record_id)
        if allowed is None:
            continue
        final_counts = Counter((m.role, m.content) for m in final_record.messages)
        if any(count > allowed.get(pair, 0) for pair, count in final_counts.items()):
            return True
    return False


def _se_mojibake_exact(
    final: DatasetView, corrupted: DatasetView, affected: tuple[str, ...]
) -> bool:
    """The affected record's message contents must equal the exact inverse of
    the corrupted contents (no paraphrase, no deletion)."""
    corrupted_by_id = {r.id: r for r in corrupted.records}
    final_by_id = {r.id: r for r in final.records}
    for record_id in affected:
        final_record = final_by_id.get(record_id)
        source = corrupted_by_id.get(record_id)
        if final_record is None or source is None:
            return True  # deleting/losing a mojibake record is not exact inversion
        if len(final_record.messages) != len(source.messages):
            return True
        for final_message, source_message in zip(
            final_record.messages, source.messages, strict=True
        ):
            expected = _invert_mojibake(source_message.content)
            if final_message.content != expected:
                return True
    return False


def evaluate_commit(
    final_data_bytes: bytes,
    final_card_bytes: bytes,
    corrupted_data_bytes: bytes,
    corrupted_card_bytes: bytes,
    contract: SftChatContract,
    nodes: tuple[DefectNode, ...],
    *,
    protected_ids: tuple[str, ...],
    count_floor: int,
) -> EquivalenceVerdict:
    final = build_view(final_data_bytes, final_card_bytes)
    corrupted = build_view(corrupted_data_bytes, corrupted_card_bytes)

    layer_results = tuple(
        run_all_layers(final, contract, protected_ids=protected_ids, count_floor=count_floor)
    )
    c_z_pass = all(result.status is CheckStatus.PASSED for result in layer_results)

    affected_all = {aid for node in nodes for aid in node.affected_ids}

    violations: list[str] = []
    if _se_del_protected(final, protected_ids):
        violations.append(checks.SE_DEL_PROTECTED)
    if _se_count_floor(final, count_floor):
        violations.append(checks.SE_COUNT_FLOOR)
    if _se_edit_unaffected(final, corrupted, affected_all):
        violations.append(checks.SE_EDIT_UNAFFECTED)
    if _se_field_inject(final, corrupted):
        violations.append(checks.SE_FIELD_INJECT)
    if _se_card_falsify(final):
        violations.append(checks.SE_CARD_FALSIFY)

    for node in nodes:
        if checks.SE_FABRICATED_CONTENT in node.prohibited_side_effects and _se_fabricated_content(
            final, corrupted, node.affected_ids
        ):
            violations.append(f"{checks.SE_FABRICATED_CONTENT}:{node.defect_id}")
        if checks.SE_MOJIBAKE_EXACT in node.prohibited_side_effects and _se_mojibake_exact(
            final, corrupted, node.affected_ids
        ):
            violations.append(f"{checks.SE_MOJIBAKE_EXACT}:{node.defect_id}")

    return EquivalenceVerdict(
        c_z_pass=c_z_pass,
        h_z_violations=tuple(dict.fromkeys(violations)),
        layer_results=layer_results,
    )
