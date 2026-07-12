"""Identifier types, hash validation, and audit references.

All hashes are lowercase hex SHA-256 strings. ``AuditRef`` lives here rather
than in ``audit.py`` because claims reference audit events while the audit log
seals claims; a lower-level home breaks that import cycle.
"""

import re
from typing import Annotated, NewType

from pydantic import BaseModel, ConfigDict, Field, field_validator

TaskId = NewType("TaskId", str)
EpisodeId = NewType("EpisodeId", str)
BlobHash = NewType("BlobHash", str)
ManifestHash = NewType("ManifestHash", str)
EventHash = NewType("EventHash", str)
ClaimId = NewType("ClaimId", str)
ToolName = NewType("ToolName", str)
SnapshotId = NewType("SnapshotId", str)

_HEX256 = re.compile(r"^[0-9a-f]{64}$")

GENESIS_HASH = "0" * 64
"""The ``prev_hash`` of the first audit event in every episode."""


def is_hex256(value: str) -> bool:
    return bool(_HEX256.match(value))


def require_hex256(value: str, *, what: str) -> str:
    if not is_hex256(value):
        raise ValueError(f"{what} must be a lowercase hex sha256 string, got {value!r}")
    return value


class AuditRef(BaseModel):
    """A stable reference to one event in the environment-owned audit log."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    episode_id: str
    event_index: Annotated[int, Field(ge=0)]
    event_hash: str

    @field_validator("event_hash")
    @classmethod
    def _valid_hash(cls, value: str) -> str:
        return require_hex256(value, what="event_hash")
