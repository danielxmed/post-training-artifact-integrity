"""Unit tests for the sft_chat verifier layers on hand-built bundles."""

from ptaie.kernel.canonical import sha256_hex
from ptaie.kernel.verification import CheckStatus, VerifierResult
from ptaie.plugins.sft_chat import checks
from ptaie.plugins.sft_chat.contract import SftChatContract
from ptaie.plugins.sft_chat.schema import (
    ChatMessage,
    ChatRecord,
    DatasetCard,
    serialize_card,
    serialize_dataset,
)
from ptaie.plugins.sft_chat.verifiers.layers import run_all_layers
from ptaie.plugins.sft_chat.view import build_view


def _contract(**overrides: str) -> SftChatContract:
    base = {
        "role_alternation": "strict_user_assistant",
        "system_policy": "forbidden",
        "mask_convention": "implicit_assistant",
        "dedup_policy": "exact_dups_forbidden",
        "dimension_status": {
            "role_alternation": "surfaced",
            "system_policy": "surfaced",
            "mask_convention": "surfaced",
            "dedup_policy": "surfaced",
        },
    }
    base.update(overrides)
    return SftChatContract.model_validate(base)


def _bundle(records: list[ChatRecord]) -> tuple[bytes, bytes]:
    data = serialize_dataset(records)
    card = serialize_card(DatasetCard(record_count=len(records), data_sha256=sha256_hex(data)))
    return data, card


def _status(results: list[VerifierResult], check_id: str) -> CheckStatus:
    return next(r.status for r in results if r.check_id == check_id)


def _qa(record_id: str, answer: str | None = None) -> ChatRecord:
    # Distinct content per id so records are not accidental exact duplicates.
    return ChatRecord(
        id=record_id,
        messages=(
            ChatMessage(role="user", content=f"A question about {record_id}?"),
            ChatMessage(role="assistant", content=answer or f"A complete answer for {record_id}."),
        ),
    )


def test_clean_bundle_passes_all() -> None:
    data, card = _bundle([_qa("rec-1"), _qa("rec-2")])
    results = run_all_layers(build_view(data, card), _contract(), count_floor=2)
    assert all(r.status is CheckStatus.PASSED for r in results)


def test_role_alternation_failure() -> None:
    bad = ChatRecord(
        id="rec-1",
        messages=(
            ChatMessage(role="user", content="Q?"),
            ChatMessage(role="user", content="Q?"),
            ChatMessage(role="assistant", content="Answer."),
        ),
    )
    data, card = _bundle([bad])
    results = run_all_layers(build_view(data, card), _contract(), count_floor=1)
    assert _status(results, checks.ROLE_ALTERNATION) is CheckStatus.FAILED


def test_final_turn_truncation_detected() -> None:
    truncated = ChatRecord(
        id="rec-1",
        messages=(
            ChatMessage(role="user", content="Q?"),
            ChatMessage(role="assistant", content="An incomplete answer that just trails"),
        ),
    )
    data, card = _bundle([truncated])
    results = run_all_layers(build_view(data, card), _contract(), count_floor=1)
    assert _status(results, checks.FINAL_TURN) is CheckStatus.FAILED


def test_schema_break_blocks_local_checks_without_hiding_other_failures() -> None:
    # rec-1 is schema-broken (valid JSON, invalid record); rec-2 has a real
    # role-alternation violation. The violation must still FAIL, and the
    # per-record checks with no parseable offender must BLOCK, not pass.
    broken_line = '{"id":"rec-1","turns":[{"role":"user","content":"Q?"}]}'
    bad_alt = ChatRecord(
        id="rec-2",
        messages=(
            ChatMessage(role="user", content="Q?"),
            ChatMessage(role="user", content="Q again?"),
            ChatMessage(role="assistant", content="Answer."),
        ),
    )
    data = (broken_line + "\n" + serialize_dataset([bad_alt]).decode()).encode()
    card = serialize_card(DatasetCard(record_count=2, data_sha256=sha256_hex(data)))
    results = run_all_layers(build_view(data, card), _contract(), count_floor=2)
    assert _status(results, checks.SCHEMA) is CheckStatus.FAILED
    assert _status(results, checks.JSONL_PARSE) is CheckStatus.PASSED  # valid JSON
    assert _status(results, checks.ROLE_ALTERNATION) is CheckStatus.FAILED  # rec-2 real failure
    assert _status(results, checks.FINAL_TURN) is CheckStatus.BLOCKED  # no offender, rec-1 unparsed


def test_dedup_and_card_consistency() -> None:
    dup = _qa("rec-1")
    dup2 = dup.model_copy(update={"id": "rec-2"})  # same content, different id
    data, card = _bundle([dup, dup2])
    results = run_all_layers(build_view(data, card), _contract(), count_floor=2)
    assert _status(results, checks.DEDUP) is CheckStatus.FAILED
    # dups allowed => passes
    allowed = run_all_layers(
        build_view(data, card), _contract(dedup_policy="dups_allowed"), count_floor=2
    )
    assert _status(allowed, checks.DEDUP) is CheckStatus.PASSED


def test_card_hash_mismatch() -> None:
    data, _ = _bundle([_qa("rec-1")])
    stale = serialize_card(DatasetCard(record_count=1, data_sha256="0" * 64))
    results = run_all_layers(build_view(data, stale), _contract(), count_floor=1)
    assert _status(results, checks.CARD_CONSISTENCY) is CheckStatus.FAILED


def test_count_floor_and_protected() -> None:
    data, card = _bundle([_qa("rec-1")])
    results = run_all_layers(
        build_view(data, card), _contract(), protected_ids=("rec-missing",), count_floor=5
    )
    assert _status(results, checks.COUNT_FLOOR) is CheckStatus.FAILED
    assert _status(results, checks.PROTECTED_RETENTION) is CheckStatus.FAILED
