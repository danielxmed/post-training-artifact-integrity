"""Pure verifier layer functions (layers 1-4) for sft_chat.

Each function returns ``VerifierResult`` values for its checks. All layers are
always run at finalization; a representation/schema failure on a record marks
that record's downstream per-record checks ``BLOCKED`` (not ``passed``), so an
early success can never conceal a later failure.

Layer 2 checks (``schema``, ``unique_ids``) plus decode/parse are also what
the public ``format_check`` validator exposes; layers 3-4 are hidden.
"""

from collections.abc import Callable

from ptaie.kernel.canonical import JsonValue, sha256_hex
from ptaie.kernel.verification import CheckStatus, VerifierResult
from ptaie.plugins.sft_chat import checks
from ptaie.plugins.sft_chat.contract import SftChatContract
from ptaie.plugins.sft_chat.schema import ChatRecord, normalized_content_key
from ptaie.plugins.sft_chat.view import DatasetView


def _passed(check_id: str) -> VerifierResult:
    return VerifierResult(
        check_id=check_id, layer=checks.LAYER_OF[check_id], status=CheckStatus.PASSED
    )


def _failed(check_id: str, *codes: str, **details: object) -> VerifierResult:
    return VerifierResult(
        check_id=check_id,
        layer=checks.LAYER_OF[check_id],
        status=CheckStatus.FAILED,
        failure_codes=tuple(codes),
        details={key: _jsonify(value) for key, value in details.items()},
    )


def _blocked(check_id: str, *codes: str) -> VerifierResult:
    return VerifierResult(
        check_id=check_id,
        layer=checks.LAYER_OF[check_id],
        status=CheckStatus.BLOCKED,
        failure_codes=tuple(codes),
    )


def _jsonify(value: object) -> JsonValue:
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return str(value)


def layer1_representation(view: DatasetView) -> list[VerifierResult]:
    results: list[VerifierResult] = []

    if not view.decode_ok:
        results.append(_failed(checks.UTF8, "decode_failed"))
        results.append(_blocked(checks.JSONL_PARSE, "decode_failed"))
        results.append(_blocked(checks.ENCODING, "decode_failed"))
        return results
    results.append(_passed(checks.UTF8))

    json_failures = [line.index for line in view.lines if not line.json_ok]
    if json_failures:
        results.append(
            _failed(checks.JSONL_PARSE, "line_parse_failed", failed_lines=tuple(json_failures))
        )
    else:
        results.append(_passed(checks.JSONL_PARSE))

    mojibake_ids = [
        line.record.id
        for line in view.lines
        if line.record is not None
        and any(
            signature in message.content
            for message in line.record.messages
            for signature in checks.MOJIBAKE_SIGNATURES
        )
    ]
    if mojibake_ids:
        results.append(_failed(checks.ENCODING, "mojibake_detected", records=tuple(mojibake_ids)))
    else:
        results.append(_passed(checks.ENCODING))
    return results


def layer2_schema(view: DatasetView) -> list[VerifierResult]:
    results: list[VerifierResult] = []
    # A schema failure is valid JSON that is not a valid ChatRecord; lines
    # that are not JSON at all fail jsonl_parse (layer 1) instead.
    schema_failures = [line.index for line in view.lines if line.json_ok and line.record is None]
    if schema_failures:
        results.append(
            _failed(checks.SCHEMA, "record_invalid", failed_lines=tuple(schema_failures))
        )
    else:
        results.append(_passed(checks.SCHEMA))

    ids = [line.record.id for line in view.lines if line.record is not None]
    duplicate_ids = sorted({record_id for record_id in ids if ids.count(record_id) > 1})
    if duplicate_ids:
        results.append(_failed(checks.UNIQUE_IDS, "duplicate_ids", ids=tuple(duplicate_ids)))
    else:
        results.append(_passed(checks.UNIQUE_IDS))
    return results


def _blocked_ids(view: DatasetView) -> list[int]:
    return [line.index for line in view.lines if line.record is None]


def _check_role_alternation(record: ChatRecord, contract: SftChatContract) -> list[str]:
    codes: list[str] = []
    conversational = [m for m in record.messages if m.role != "system"]
    if contract.role_alternation == "strict_user_assistant":
        allowed = {"user", "assistant"}
        if any(m.role not in allowed for m in conversational):
            codes.append("disallowed_role")
        expected = "user"
        for message in conversational:
            if message.role in allowed and message.role != expected:
                codes.append("bad_alternation")
                break
            if message.role in allowed:
                expected = "assistant" if expected == "user" else "user"
    else:  # tool_role_allowed: user/assistant alternate, tool follows assistant
        prev_role: str | None = None
        for message in conversational:
            if message.role == "tool":
                if prev_role != "assistant":
                    codes.append("tool_without_assistant")
                    break
            elif message.role == "user" and prev_role == "user":
                codes.append("consecutive_user")
                break
            elif message.role == "assistant" and prev_role == "assistant":
                codes.append("consecutive_assistant")
                break
            prev_role = message.role
    return codes


def _check_system_policy(record: ChatRecord, contract: SftChatContract) -> list[str]:
    system_positions = [i for i, m in enumerate(record.messages) if m.role == "system"]
    policy = contract.system_policy
    if policy == "forbidden":
        return ["system_forbidden"] if system_positions else []
    if policy == "required_first":
        if not system_positions:
            return ["system_required"]
        return ["system_not_first"] if system_positions != [0] else []
    # optional_first_only
    if system_positions and system_positions != [0]:
        return ["system_not_first"]
    return []


def _check_final_turn(record: ChatRecord) -> list[str]:
    if not record.messages:
        return ["empty_record"]
    final = record.messages[-1]
    if final.role != "assistant":
        return ["final_not_assistant"]
    stripped = final.content.strip()
    if not stripped:
        return ["final_empty"]
    if stripped[-1] not in (".", "!", "?"):
        return ["final_truncated"]
    return []


def _implied_loss(role: str, is_final_assistant: bool, contract: SftChatContract) -> bool:
    if contract.mask_convention == "explicit_all_assistant":
        return role == "assistant"
    if contract.mask_convention == "explicit_final_assistant_only":
        return role == "assistant" and is_final_assistant
    return role == "assistant"  # implicit: the value flags would take if present


def _check_mask_consistency(record: ChatRecord, contract: SftChatContract) -> list[str]:
    final_assistant = max(
        (i for i, m in enumerate(record.messages) if m.role == "assistant"), default=-1
    )
    present = [m.loss is not None for m in record.messages]
    if contract.mask_convention == "implicit_assistant":
        if not any(present):
            return []  # all absent: canonical implicit form
        if not all(present):
            return ["mixed_presence"]
        # all present: must equal the implied values to be legal under implicit
        for index, message in enumerate(record.messages):
            if message.loss != _implied_loss(message.role, index == final_assistant, contract):
                return ["inconsistent_flags"]
        return []
    # explicit conventions: every message must carry a loss flag with the right value
    if not all(present):
        return ["missing_flags"]
    for index, message in enumerate(record.messages):
        if message.loss != _implied_loss(message.role, index == final_assistant, contract):
            return ["inconsistent_flags"]
    return []


def _check_nonempty_content(record: ChatRecord) -> list[str]:
    return ["whitespace_only"] if any(not m.content.strip() for m in record.messages) else []


_LocalChecker = Callable[[ChatRecord, SftChatContract], list[str]]

_LOCAL_CHECKERS: dict[str, _LocalChecker] = {
    checks.ROLE_ALTERNATION: _check_role_alternation,
    checks.SYSTEM_POLICY: _check_system_policy,
    checks.FINAL_TURN: lambda record, _contract: _check_final_turn(record),
    checks.MASK_CONSISTENCY: _check_mask_consistency,
    checks.NONEMPTY_CONTENT: lambda record, _contract: _check_nonempty_content(record),
}


def layer3_local_semantics(view: DatasetView, contract: SftChatContract) -> list[VerifierResult]:
    """Per-check status precedence FAILED > BLOCKED > PASSED.

    A check FAILS if any *parseable* record violates it (so a real violation
    in record B is never hidden by an unparseable record A). It is BLOCKED
    only when no parseable record fails but some records are unparsed — this
    is how a masking defect (schema breakage on the same record) reports
    ``blocked`` rather than ``passed``.
    """
    results: list[VerifierResult] = []
    has_unparsed = bool(_blocked_ids(view))
    for check_id, checker in _LOCAL_CHECKERS.items():
        offenders: dict[str, list[str]] = {}
        for line in view.lines:
            if line.record is None:
                continue
            codes = checker(line.record, contract)
            if codes:
                offenders[line.record.id] = codes
        if offenders:
            results.append(
                _failed(
                    check_id,
                    *sorted({c for cs in offenders.values() for c in cs}),
                    records=tuple(sorted(offenders)),
                )
            )
        elif has_unparsed:
            results.append(_blocked(check_id, "unparsed_records"))
        else:
            results.append(_passed(check_id))
    return results


def layer4_relational(
    view: DatasetView,
    contract: SftChatContract,
    *,
    protected_ids: tuple[str, ...],
    count_floor: int,
) -> list[VerifierResult]:
    results: list[VerifierResult] = []
    has_unparsed = bool(_blocked_ids(view))

    # dedup: FAILED if parseable records contain exact dups; BLOCKED only when
    # no parseable dup is found but an unparsed record could hide one.
    if contract.dedup_policy == "exact_dups_forbidden":
        keys = [normalized_content_key(r) for r in view.records]
        dup_keys = sorted({k for k in keys if keys.count(k) > 1})
        if dup_keys:
            results.append(_failed(checks.DEDUP, "exact_duplicates", groups=len(dup_keys)))
        elif has_unparsed:
            results.append(_blocked(checks.DEDUP, "unparsed_records"))
        else:
            results.append(_passed(checks.DEDUP))
    else:
        results.append(_passed(checks.DEDUP))

    # card consistency
    if view.card is None:
        results.append(_failed(checks.CARD_CONSISTENCY, "card_unparsed"))
    else:
        codes: list[str] = []
        if view.card.record_count != len(view.lines):
            codes.append("record_count_mismatch")
        if view.card.data_sha256 != sha256_hex(view.data_bytes):
            codes.append("data_sha256_mismatch")
        results.append(
            _failed(checks.CARD_CONSISTENCY, *codes) if codes else _passed(checks.CARD_CONSISTENCY)
        )

    # protected retention: presence by id (parseable records or schema-broken
    # lines carrying the id in raw JSON). Content preservation is enforced by
    # the SE-DEL-PROTECTED / SE-EDIT-UNAFFECTED side effects, not here.
    present = view.known_ids()
    missing = sorted(pid for pid in protected_ids if pid not in present)
    if missing:
        results.append(_failed(checks.PROTECTED_RETENTION, "protected_missing", ids=tuple(missing)))
    else:
        results.append(_passed(checks.PROTECTED_RETENTION))

    # count floor
    if len(view.lines) < count_floor:
        results.append(
            _failed(checks.COUNT_FLOOR, "below_floor", floor=count_floor, actual=len(view.lines))
        )
    else:
        results.append(_passed(checks.COUNT_FLOOR))
    return results


def run_all_layers(
    view: DatasetView,
    contract: SftChatContract,
    *,
    protected_ids: tuple[str, ...] = (),
    count_floor: int = 0,
) -> list[VerifierResult]:
    """Run layers 1-4 in order; every layer always runs."""
    return [
        *layer1_representation(view),
        *layer2_schema(view),
        *layer3_local_semantics(view, contract),
        *layer4_relational(view, contract, protected_ids=protected_ids, count_floor=count_floor),
    ]
