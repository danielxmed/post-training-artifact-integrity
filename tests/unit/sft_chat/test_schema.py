import pytest
from pydantic import ValidationError

from ptaie.plugins.sft_chat.schema import (
    ChatMessage,
    ChatRecord,
    join_data_lines,
    normalized_content_key,
    parse_record_line,
    serialize_dataset,
    serialize_record,
    split_data_lines,
)


def _record() -> ChatRecord:
    return ChatRecord(
        id="rec-0001",
        messages=(
            ChatMessage(role="user", content="Explain café logistics?"),
            ChatMessage(role="assistant", content="Here is a summary of café logistics."),
        ),
    )


def test_record_line_roundtrip() -> None:
    record = _record()
    assert parse_record_line(serialize_record(record)) == record


def test_loss_none_is_omitted() -> None:
    line = serialize_record(_record())
    assert '"loss"' not in line
    with_loss = _record().model_copy(
        update={"messages": (_record().messages[0].model_copy(update={"loss": False}),)}
    )
    assert '"loss":false' in serialize_record(with_loss)


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        ChatRecord.model_validate({"id": "x", "messages": [], "surprise": 1})


def test_split_join_roundtrip() -> None:
    records = [_record(), _record().model_copy(update={"id": "rec-0002"})]
    data = serialize_dataset(records)
    assert data.endswith(b"\n")
    lines = split_data_lines(data)
    assert len(lines) == 2
    assert join_data_lines(lines) == data


def test_normalized_key_ignores_id() -> None:
    a = _record()
    b = _record().model_copy(update={"id": "rec-9999"})
    assert normalized_content_key(a) == normalized_content_key(b)
    c = _record().model_copy(
        update={"messages": (_record().messages[0].model_copy(update={"content": "different"}),)}
    )
    assert normalized_content_key(a) != normalized_content_key(c)


def test_serialization_is_deterministic() -> None:
    record = _record()
    assert serialize_record(record) == serialize_record(record)
