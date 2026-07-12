"""Helpers to drive an sft_chat episode end-to-end through the real finalizer.

These let PR5 tests reach a chosen terminal disposition after applying a set
of line-level edits, exercising the full engine → seal → finalize path.
"""

from ptaie.kernel.actions import TerminalAction, ToolAction
from ptaie.kernel.claims import ClaimBundle
from ptaie.kernel.engine import PtaieEnv, StepResult
from ptaie.kernel.plugin import PluginRegistry
from ptaie.kernel.rewards import TerminalDisposition
from ptaie.plugins.sft_chat.finalize import SftChatFinalizer
from ptaie.plugins.sft_chat.plugin import SftChatPlugin
from ptaie.plugins.sft_chat.schema import parse_record_line, split_data_lines
from ptaie.plugins.sft_chat.taskgen import SftTask
from ptaie.version import ENV_VERSION


def make_env() -> PtaieEnv:
    registry = PluginRegistry()
    registry.register(SftChatPlugin())
    return PtaieEnv(registry, SftChatFinalizer())


def line_index_of(data: bytes, record_id: str) -> int | None:
    for index, line in enumerate(split_data_lines(data)):
        try:
            if parse_record_line(line).id == record_id:
                return index
        except ValueError:
            continue
    return None


def rewrite_to(env: PtaieEnv, target_data: bytes, *, regenerate: bool = True) -> StepResult:
    """Drive the workspace's data file to ``target_data`` line-by-line, then
    (optionally) regenerate the card. Returns the last step result."""
    target_lines = split_data_lines(target_data)
    current = env.engine
    # read current line count
    listing = env.step(
        ToolAction(tool_name="kernel.read_file", arguments={"path": "data/train.jsonl"})
    )
    payload = listing.observation.payload
    assert payload.__class__.__name__ == "ToolResultPayload"
    current_data = payload.outcome.output["content"].encode("utf-8")  # type: ignore[union-attr,index]
    current_lines = split_data_lines(current_data)

    result = listing
    # delete extra trailing lines
    while len(current_lines) > len(target_lines):
        result = env.step(
            ToolAction(
                tool_name="sft_chat.delete_record",
                arguments={"line_index": len(current_lines) - 1},
            )
        )
        current_lines.pop()
    # replace / insert
    for index, line in enumerate(target_lines):
        if index < len(current_lines):
            if current_lines[index] != line:
                result = env.step(
                    ToolAction(
                        tool_name="sft_chat.replace_record",
                        arguments={"line_index": index, "raw_line": line},
                    )
                )
        else:
            result = env.step(
                ToolAction(
                    tool_name="sft_chat.insert_record",
                    arguments={"line_index": index, "raw_line": line},
                )
            )
    if regenerate:
        result = env.step(ToolAction(tool_name="sft_chat.regenerate_card"))
    _ = current
    return result


def commit(env: PtaieEnv, *, confidence: float = 0.8) -> StepResult:
    root = _staged_root(env)
    return env.step(
        TerminalAction(
            claim_bundle=ClaimBundle(
                disposition=TerminalDisposition.COMMIT,
                artifact_root_hash=root,
                confidence=confidence,
            )
        )
    )


def terminal(env: PtaieEnv, disposition: TerminalDisposition, **kwargs: object) -> StepResult:
    root = _staged_root(env)
    unresolved = ("residual",) if disposition is TerminalDisposition.PARTIAL_HANDOFF else ()
    bundle = ClaimBundle(
        disposition=disposition,
        artifact_root_hash=root,
        confidence=float(kwargs.get("confidence", 0.8)),  # type: ignore[arg-type]
        unresolved=unresolved,
    )
    return env.step(TerminalAction(claim_bundle=bundle))


def _staged_root(env: PtaieEnv) -> str:
    result = env.step(ToolAction(tool_name="kernel.hash_artifact"))
    payload = result.observation.payload
    assert payload.__class__.__name__ == "ToolResultPayload"
    return payload.outcome.output["hash"]  # type: ignore[union-attr,index,return-value]


def reset_to(env: PtaieEnv, task: SftTask) -> None:
    env.reset(task_seed=task.task_seed, artifact_class="sft_chat", env_version=ENV_VERSION)
