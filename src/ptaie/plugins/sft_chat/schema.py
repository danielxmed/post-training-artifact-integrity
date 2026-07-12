"""Record and card schemas for SFT chat bundles, plus line-level JSONL codecs.

The artifact under repair is a JSONL file the agent edits at the raw-line
level (so schema-broken lines remain editable). Serialization is canonical
per line: sorted keys, no ASCII escaping, ``loss: None`` means the key is
absent (the implicit-mask convention).
"""

import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ptaie.kernel.ids import require_hex256

KNOWN_ROLES = ("system", "user", "assistant", "tool")

CARD_SCHEMA_VERSION = "sftchat/v1"


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: str
    content: str
    loss: bool | None = None


class ChatRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    messages: tuple[ChatMessage, ...]
    meta: dict[str, str] = Field(default_factory=dict)


class DatasetCard(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["sftchat/v1"] = "sftchat/v1"
    declared: dict[str, str] = Field(default_factory=dict)
    record_count: Annotated[int, Field(ge=0)]
    data_sha256: str
    notes: tuple[str, ...] = ()

    @field_validator("data_sha256")
    @classmethod
    def _valid_hash(cls, value: str) -> str:
        return require_hex256(value, what="data_sha256")


def serialize_record(record: ChatRecord) -> str:
    """One canonical JSONL line (no trailing newline)."""
    return json.dumps(
        record.model_dump(mode="json", exclude_none=True),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def parse_record_line(line: str) -> ChatRecord:
    """Parse one JSONL line strictly. Raises ``ValueError`` subclasses
    (json / pydantic) — layer functions convert failures into check results,
    never exceptions to the agent."""
    return ChatRecord.model_validate(json.loads(line))


def split_data_lines(data: bytes) -> list[str]:
    """Split JSONL bytes into lines (without newline characters).

    A single trailing newline is the serialization convention and is not a
    line; anything else (empty interior lines, missing trailing newline) is
    surfaced by the representation layer, not here.
    """
    text = data.decode("utf-8")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def join_data_lines(lines: list[str]) -> bytes:
    return ("\n".join(lines) + "\n").encode("utf-8") if lines else b""


def serialize_dataset(records: list[ChatRecord]) -> bytes:
    return join_data_lines([serialize_record(record) for record in records])


def serialize_card(card: DatasetCard) -> bytes:
    return (
        json.dumps(
            card.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def parse_card(data: bytes) -> DatasetCard:
    return DatasetCard.model_validate(json.loads(data.decode("utf-8")))


def normalized_content_key(record: ChatRecord) -> str:
    """Duplicate detection key: the record's content identity, id excluded."""
    return json.dumps(
        [
            {"role": message.role, "content": message.content, "loss": message.loss}
            for message in record.messages
        ],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
