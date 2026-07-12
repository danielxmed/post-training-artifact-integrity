"""A parsed-or-raw view of a bundle, shared by verifiers and repairs.

Parsing is best-effort per line and distinguishes two failure modes so the
representation and schema layers stay independent:

- ``json_ok=False`` — the line is not a JSON object (a representation failure
  that fails ``jsonl_parse``);
- ``json_ok=True`` but ``record is None`` — valid JSON that is not a valid
  ``ChatRecord`` (a schema failure).

Either way ``record is None`` blocks the per-record semantic checks on that
record — they report ``blocked``, never silently ``passed``.
"""

import json
from dataclasses import dataclass

from ptaie.plugins.sft_chat.schema import (
    ChatRecord,
    DatasetCard,
    parse_card,
    split_data_lines,
)


@dataclass(frozen=True)
class ParsedLine:
    index: int
    raw: str
    record: ChatRecord | None
    json_ok: bool
    error: str | None


@dataclass(frozen=True)
class DatasetView:
    data_bytes: bytes
    card_bytes: bytes
    lines: tuple[ParsedLine, ...]
    decode_ok: bool
    card: DatasetCard | None
    card_error: str | None

    @property
    def records(self) -> list[ChatRecord]:
        return [line.record for line in self.lines if line.record is not None]

    @property
    def all_parsed(self) -> bool:
        return all(line.record is not None for line in self.lines)

    def record_by_id(self, record_id: str) -> ChatRecord | None:
        for line in self.lines:
            if line.record is not None and line.record.id == record_id:
                return line.record
        return None

    def known_ids(self) -> set[str]:
        """Record ids present in the data, including the raw-JSON ``id`` of
        schema-broken (unparseable-as-record but valid-JSON) lines. Used to
        tell a repaired-in-place record from a fabricated one."""
        ids: set[str] = set()
        for line in self.lines:
            if line.record is not None:
                ids.add(line.record.id)
            elif line.json_ok:
                try:
                    payload = json.loads(line.raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict) and isinstance(payload.get("id"), str):
                    ids.add(payload["id"])
        return ids


def build_view(data_bytes: bytes, card_bytes: bytes) -> DatasetView:
    decode_ok = True
    parsed: list[ParsedLine] = []
    try:
        raw_lines = split_data_lines(data_bytes)
    except UnicodeDecodeError:
        decode_ok = False
        raw_lines = []
    for index, raw in enumerate(raw_lines):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            parsed.append(
                ParsedLine(index=index, raw=raw, record=None, json_ok=False, error=str(exc)[:120])
            )
            continue
        try:
            record = ChatRecord.model_validate(payload)
            parsed.append(ParsedLine(index=index, raw=raw, record=record, json_ok=True, error=None))
        except ValueError as exc:
            parsed.append(
                ParsedLine(index=index, raw=raw, record=None, json_ok=True, error=str(exc)[:120])
            )

    card: DatasetCard | None = None
    card_error: str | None = None
    try:
        card = parse_card(card_bytes)
    except (ValueError, UnicodeDecodeError) as exc:
        card_error = str(exc)[:120]

    return DatasetView(
        data_bytes=data_bytes,
        card_bytes=card_bytes,
        lines=tuple(parsed),
        decode_ok=decode_ok,
        card=card,
        card_error=card_error,
    )
