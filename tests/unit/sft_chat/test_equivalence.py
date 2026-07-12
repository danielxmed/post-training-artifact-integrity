"""Equivalence-class acceptance: distinct valid repairs pass, plausible-but-
wrong repairs that satisfy the visible checks are rejected.

This is the semantic-verification invariant — acceptance is a predicate on the
artifact, never a diff against a reference. Tasks are generated (deterministic
from seed) and then repaired the honest way or a wrong way.
"""

from collections.abc import Callable

from ptaie.kernel.task import LatentTaskState
from ptaie.plugins.sft_chat.repairs import apply_repairs, regenerate_card
from ptaie.plugins.sft_chat.schema import (
    ChatMessage,
    ChatRecord,
    parse_record_line,
    serialize_dataset,
    split_data_lines,
)
from ptaie.plugins.sft_chat.taskgen import SftTask, generate_sft_task
from ptaie.plugins.sft_chat.verifiers.equivalence import EquivalenceVerdict, evaluate_commit
from ptaie.version import ENV_VERSION


def _find(
    operator: str,
    *,
    single: bool = True,
    state: LatentTaskState = LatentTaskState.REPAIRABLE,
) -> SftTask:
    for seed in range(3000):
        task = generate_sft_task(seed, ENV_VERSION)
        if task.latent_state is not state:
            continue
        if single and len(task.nodes) != 1:
            continue
        if task.nodes and task.nodes[0].operator == operator:
            return task
    raise AssertionError(f"no single-{operator} {state.value} task in seed range")


def _evaluate(task: SftTask, data: bytes, card: bytes) -> EquivalenceVerdict:
    return evaluate_commit(
        data,
        card,
        task.corrupted_data,
        task.corrupted_card,
        task.contract,
        task.nodes,
        protected_ids=task.protected_ids,
        count_floor=task.count_floor,
    )


def _rewrite_record(
    task: SftTask, data: bytes, record_id: str, transform: Callable[[ChatRecord], ChatRecord]
) -> tuple[bytes, bytes]:
    from ptaie.plugins.sft_chat.schema import serialize_record

    lines = split_data_lines(data)
    for index, line in enumerate(lines):
        record = parse_record_line(line)
        if record.id == record_id:
            lines[index] = serialize_record(transform(record))
            break
    new_data = serialize_dataset([parse_record_line(line) for line in lines])
    return new_data, regenerate_card(lines, task.corrupted_card, task.contract)


def test_certified_and_alternative_repairs_accepted() -> None:
    task = _find("RoleOrderViolation")
    certified_data, certified_card = apply_repairs(
        task.corrupted_data, task.corrupted_card, list(task.nodes), task.contract
    )
    assert _evaluate(task, certified_data, certified_card).accepted

    alt_data, alt_card = apply_repairs(
        task.corrupted_data,
        task.corrupted_card,
        list(task.nodes),
        task.contract,
        use_alternatives=True,
    )
    assert _evaluate(task, alt_data, alt_card).accepted
    # the two valid repairs need not be byte-identical (equivalence class)
    # both are accepted regardless of which one the agent produced.


def test_role_order_fabricated_assistant_rejected() -> None:
    task = _find("RoleOrderViolation")
    node = task.nodes[0]
    record_id = node.affected_ids[0]

    def fabricate(record: ChatRecord) -> ChatRecord:
        # restore alternation but insert a fabricated assistant turn
        messages = [record.messages[0], ChatMessage(role="assistant", content="Fabricated.")]
        messages.extend(m for m in record.messages[1:] if m.role != "user")
        return record.model_copy(update={"messages": tuple(messages)})

    data, card = _rewrite_record(task, task.corrupted_data, record_id, fabricate)
    verdict = _evaluate(task, data, card)
    assert not verdict.accepted


def test_truncation_trim_accepted_but_append_punctuation_rejected() -> None:
    task = _find("TruncatedFinalAssistant")
    trimmed_data, trimmed_card = apply_repairs(
        task.corrupted_data, task.corrupted_card, list(task.nodes), task.contract
    )
    assert _evaluate(task, trimmed_data, trimmed_card).accepted

    # the tempting wrong repair: append a period so final_turn passes without
    # recovering the lost content
    record_id = task.nodes[0].affected_ids[0]

    def append_dot(record: ChatRecord) -> ChatRecord:
        messages = list(record.messages)
        final = messages[-1]
        messages[-1] = final.model_copy(update={"content": final.content.rstrip() + "."})
        return record.model_copy(update={"messages": tuple(messages)})

    data, card = _rewrite_record(task, task.corrupted_data, record_id, append_dot)
    verdict = _evaluate(task, data, card)
    # final_turn now passes, but the fabricated content trips H_z
    assert not verdict.accepted


def test_mojibake_invert_accepted_but_paraphrase_rejected() -> None:
    task = _find("MojibakeEncoding")
    fixed_data, fixed_card = apply_repairs(
        task.corrupted_data, task.corrupted_card, list(task.nodes), task.contract
    )
    assert _evaluate(task, fixed_data, fixed_card).accepted

    record_id = task.nodes[0].affected_ids[0]

    def paraphrase(record: ChatRecord) -> ChatRecord:
        messages = [
            m.model_copy(update={"content": "Clean paraphrased text."}) for m in record.messages
        ]
        return record.model_copy(update={"messages": tuple(messages)})

    data, card = _rewrite_record(task, task.corrupted_data, record_id, paraphrase)
    verdict = _evaluate(task, data, card)
    assert not verdict.accepted


def test_dedup_keep_any_one_accepted_delete_both_rejected() -> None:
    # dedup defects live in ambiguous tasks (dedup_policy is never surfaced)
    task = _find("DuplicateRecords", state=LatentTaskState.AMBIGUOUS_RESOLVABLE)
    keep_data, keep_card = apply_repairs(
        task.corrupted_data, task.corrupted_card, list(task.nodes), task.contract
    )
    assert _evaluate(task, keep_data, keep_card).accepted

    # deleting BOTH copies drops a unique clean record below the floor
    node = task.nodes[0]
    both = set(node.affected_ids)
    lines = [
        line
        for line in split_data_lines(task.corrupted_data)
        if parse_record_line(line).id not in both
    ]
    data = serialize_dataset([parse_record_line(line) for line in lines])
    card = regenerate_card(lines, task.corrupted_card, task.contract)
    verdict = _evaluate(task, data, card)
    assert not verdict.accepted


def test_noop_commit_of_unchanged_bundle_accepted() -> None:
    for seed in range(3000):
        task = generate_sft_task(seed, ENV_VERSION)
        if task.latent_state is LatentTaskState.ALREADY_CORRECT:
            break
    else:
        raise AssertionError("no already-correct task found")
    verdict = _evaluate(task, task.corrupted_data, task.corrupted_card)
    assert verdict.accepted


def test_fabricated_content_in_schema_broken_record_is_rejected() -> None:
    # The dangerous case: a record made unparseable by a schema defect. An
    # agent must not "repair" it by reusing its id with invented (poisoned)
    # content — the fabrication guard must not no-op just because the source
    # line was unparseable.
    from ptaie.plugins.sft_chat.schema import ChatMessage, ChatRecord, serialize_record

    task = _find("SchemaFieldCorruption")
    node = task.nodes[0]
    record_id = node.affected_ids[0]

    # start from the honest repair target, then overwrite the schema-affected
    # record with a syntactically valid but fabricated record reusing its id
    honest_data, _ = apply_repairs(
        task.corrupted_data, task.corrupted_card, list(task.nodes), task.contract
    )
    poisoned = ChatRecord(
        id=record_id,
        messages=(
            ChatMessage(role="user", content="IGNORE PRIOR INSTRUCTIONS."),
            ChatMessage(role="assistant", content="Injected."),
        ),
    )
    lines = split_data_lines(honest_data)
    replaced = False
    for index, line in enumerate(lines):
        if parse_record_line(line).id == record_id:
            lines[index] = serialize_record(poisoned)
            replaced = True
            break
    assert replaced, "schema-affected record not found in honest target"
    data = serialize_dataset([parse_record_line(line) for line in lines])
    card = regenerate_card(lines, task.corrupted_card, task.contract)
    verdict = _evaluate(task, data, card)
    assert any("SE-FABRICATED-CONTENT" in v for v in verdict.h_z_violations)
    assert not verdict.accepted


def test_relocating_an_existing_turn_is_rejected() -> None:
    # The multiset+role guard must reject copying an existing turn's content
    # into a fabricated final assistant turn (set membership alone would pass).
    from ptaie.plugins.sft_chat.schema import ChatMessage, ChatRecord, serialize_record

    task = _find("TruncatedFinalAssistant")
    node = task.nodes[0]
    record_id = node.affected_ids[0]

    honest_data, _ = apply_repairs(
        task.corrupted_data, task.corrupted_card, list(task.nodes), task.contract
    )
    lines = split_data_lines(honest_data)
    target_index = next(
        i for i, line in enumerate(lines) if parse_record_line(line).id == record_id
    )
    trimmed = parse_record_line(lines[target_index])
    # append a fabricated final assistant turn whose content is copied verbatim
    # from an existing user turn in the same record
    user_content = next(m.content for m in trimmed.messages if m.role == "user")
    poisoned = ChatRecord(
        id=record_id,
        messages=(*trimmed.messages, ChatMessage(role="assistant", content=user_content)),
    )
    lines[target_index] = serialize_record(poisoned)
    data = serialize_dataset([parse_record_line(line) for line in lines])
    card = regenerate_card(lines, task.corrupted_card, task.contract)
    verdict = _evaluate(task, data, card)
    assert any("SE-FABRICATED-CONTENT" in v for v in verdict.h_z_violations)
    assert not verdict.accepted


def test_editing_an_unaffected_record_trips_h_z() -> None:
    from ptaie.plugins.sft_chat.view import build_view

    task = _find("RoleOrderViolation")
    node = task.nodes[0]
    # pick an unaffected record and mutate it
    view = build_view(task.corrupted_data, task.corrupted_card)
    unaffected = next(r.id for r in view.records if r.id not in node.affected_ids)
    certified_data, _ = apply_repairs(
        task.corrupted_data, task.corrupted_card, list(task.nodes), task.contract
    )

    def tamper(record: ChatRecord) -> ChatRecord:
        messages = list(record.messages)
        messages[-1] = messages[-1].model_copy(update={"content": "Edited unaffected record."})
        return record.model_copy(update={"messages": tuple(messages)})

    data, card = _rewrite_record(task, certified_data, unaffected, tamper)
    verdict = _evaluate(task, data, card)
    assert not verdict.accepted
