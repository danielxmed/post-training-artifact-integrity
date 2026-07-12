"""Degenerate baselines that must be measurably worse than the oracle.

``AlwaysAbstainPolicy`` refuses every task (correct only on the ask/abstain
states — abstention is not a safe harbor). ``AlwaysRepairPolicy`` edits
aggressively without reading the contract (dedup, strip loss flags), which
harms already-correct data and never fixes the actual defect — the edit-first
failure mode the task distribution is designed to punish.
"""

from ptaie.kernel.actions import Action, TerminalAction, ToolAction
from ptaie.kernel.claims import ClaimBundle
from ptaie.kernel.observation import Observation, ToolResultPayload
from ptaie.kernel.rewards import TerminalDisposition
from ptaie.plugins.sft_chat import DATA_PATH
from ptaie.plugins.sft_chat.schema import (
    normalized_content_key,
    parse_record_line,
    serialize_dataset,
    split_data_lines,
)
from ptaie.policies.planning import plan_rewrite

_DATA_ARG = {"path": DATA_PATH}


class AlwaysAbstainPolicy:
    def act(self, observation: Observation) -> Action:
        return TerminalAction(
            claim_bundle=ClaimBundle(
                disposition=TerminalDisposition.ABSTAIN,
                artifact_root_hash=observation.staged_root,
                confidence=0.5,
            )
        )


class AlwaysRepairPolicy:
    """Reads the data, applies contract-blind "fixes", regenerates the card,
    and commits."""

    __slots__ = ("_queue", "_started")

    def __init__(self) -> None:
        self._queue: list[Action] | None = None
        self._started = False

    def act(self, observation: Observation) -> Action:
        if not self._started:
            self._started = True
            return ToolAction(tool_name="kernel.read_file", arguments=dict(_DATA_ARG))
        if self._queue is None:
            self._queue = self._build_queue(observation)
        if self._queue:
            return self._queue.pop(0)
        return TerminalAction(
            claim_bundle=ClaimBundle(
                disposition=TerminalDisposition.COMMIT,
                artifact_root_hash=observation.staged_root,
                confidence=0.9,
            )
        )

    def _build_queue(self, observation: Observation) -> list[Action]:
        payload = observation.payload
        if not isinstance(payload, ToolResultPayload) or payload.outcome.output is None:
            return []
        content = payload.outcome.output.get("content")
        if not isinstance(content, str):
            return []
        current = content.encode("utf-8")
        target = _blind_repair(current)
        queue: list[Action] = list(plan_rewrite(current, target))
        queue.append(ToolAction(tool_name="sft_chat.regenerate_card"))
        return queue


def _blind_repair(data: bytes) -> bytes:
    seen: set[str] = set()
    records = []
    for line in split_data_lines(data):
        try:
            record = parse_record_line(line)
        except ValueError:
            continue  # drop unparseable
        key = normalized_content_key(record)
        if key in seen:
            continue  # dedup
        seen.add(key)
        stripped = tuple(m.model_copy(update={"loss": None}) for m in record.messages)
        records.append(record.model_copy(update={"messages": stripped}))
    return serialize_dataset(records)
