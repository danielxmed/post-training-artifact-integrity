"""The sft_chat ``ArtifactClassPlugin`` implementation.

Ties the generator, the line-level tools, the visible validators, and the
clarification oracle into the kernel plugin protocol. Hidden verification and
scoring (the finalizer) are injected into ``PtaieEnv`` separately in PR5.
"""

from ptaie.kernel.canonical import sha256_hex
from ptaie.kernel.clarify import ClarificationOracle, ScriptedOracle
from ptaie.kernel.plugin import GeneratedTask
from ptaie.kernel.store.workspace import WorkspaceReadView
from ptaie.kernel.task import TaskRecord
from ptaie.kernel.tools.base import Tool
from ptaie.kernel.verification import CheckStatus, Validator, VerifierLayer, VerifierResult
from ptaie.plugins.sft_chat import ARTIFACT_CLASS, CARD_PATH, DATA_PATH, checks
from ptaie.plugins.sft_chat.taskgen import generate_sft_task
from ptaie.plugins.sft_chat.tools import sft_chat_tools
from ptaie.plugins.sft_chat.verifiers.layers import layer1_representation, layer2_schema
from ptaie.plugins.sft_chat.view import build_view

_PUBLIC_FORMAT_CHECKS = frozenset(
    {checks.UTF8, checks.JSONL_PARSE, checks.SCHEMA, checks.UNIQUE_IDS}
)


def _view(read: WorkspaceReadView) -> object:
    data = read.read(DATA_PATH) if read.has(DATA_PATH) else b""
    card = read.read(CARD_PATH) if read.has(CARD_PATH) else b"{}"
    return build_view(data, card)


class FormatValidator:
    """Public representation + schema checks (what checks/format_check.py
    documents): UTF-8 JSONL parse, per-record schema, unique ids."""

    check_id = "format_check"
    layer = VerifierLayer.SCHEMA

    def check(self, view: WorkspaceReadView) -> tuple[VerifierResult, ...]:
        dataset = _view(view)
        results = [*layer1_representation(dataset), *layer2_schema(dataset)]  # type: ignore[arg-type]
        return tuple(r for r in results if r.check_id in _PUBLIC_FORMAT_CHECKS)


class CardValidator:
    """Public card consistency: record_count and data_sha256 match the data."""

    check_id = "card_check"
    layer = VerifierLayer.RELATIONAL

    def check(self, view: WorkspaceReadView) -> tuple[VerifierResult, ...]:
        dataset = _view(view)
        data = dataset.data_bytes  # type: ignore[attr-defined]
        card = dataset.card  # type: ignore[attr-defined]
        line_count = len(dataset.lines)  # type: ignore[attr-defined]
        codes: list[str] = []
        if card is None:
            codes.append("card_unparsed")
        else:
            if card.record_count != line_count:
                codes.append("record_count_mismatch")
            if card.data_sha256 != sha256_hex(data):
                codes.append("data_sha256_mismatch")
        status = CheckStatus.PASSED if not codes else CheckStatus.FAILED
        return (
            VerifierResult(
                check_id=checks.CARD_CONSISTENCY,
                layer=VerifierLayer.RELATIONAL,
                status=status,
                failure_codes=tuple(codes),
            ),
        )


class SftChatPlugin:
    name = ARTIFACT_CLASS
    plugin_version = "0.1.0"

    def generate(self, task_seed: int, env_version: str) -> GeneratedTask:
        task = generate_sft_task(task_seed, env_version)
        return GeneratedTask(
            task=task.task_record, store=task.store, manifest=task.initial_manifest
        )

    def tools(self) -> tuple[Tool, ...]:
        return sft_chat_tools()

    def visible_validators(self) -> tuple[Validator, ...]:
        return (FormatValidator(), CardValidator())

    def clarification_oracle(self, task: TaskRecord) -> ClarificationOracle:
        return ScriptedOracle(task.hidden.clarification_script)
