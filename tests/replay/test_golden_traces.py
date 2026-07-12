"""Replay the committed golden traces and assert bit-identical outcomes.

The golden fixtures pin the full determinism contract (terminal code, reward
totals, constraint vector, staged root, and the audit head hash). Regenerate
with ``python scripts/regenerate_golden.py``; a diff there requires an
``ENV_VERSION`` bump.
"""

import json
from pathlib import Path
from typing import Any, cast

import pytest

from ptaie.kernel.actions import ActionTrace
from ptaie.kernel.plugin import PluginRegistry
from ptaie.kernel.replay import ReplayHarness, ReplayResult
from ptaie.plugins.sft_chat.finalize import SftChatFinalizer
from ptaie.plugins.sft_chat.plugin import SftChatPlugin
from ptaie.version import ENV_VERSION

pytestmark = pytest.mark.replay

_GOLDEN = Path(__file__).parent.parent / "fixtures" / "golden_traces" / "oracle.json"


def _load() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(_GOLDEN.read_text(encoding="utf-8")))


def _harness() -> ReplayHarness:
    registry = PluginRegistry()
    registry.register(SftChatPlugin())
    return ReplayHarness(registry, SftChatFinalizer())


def test_golden_file_matches_env_version() -> None:
    payload = _load()
    assert payload["env_version"] == ENV_VERSION, (
        "golden traces are from a different ENV_VERSION; regenerate and justify"
    )


def test_all_golden_traces_replay_identically() -> None:
    payload = _load()
    harness = _harness()
    entries = payload["entries"]
    assert isinstance(entries, list) and entries
    for entry in entries:
        trace = ActionTrace.model_validate(entry["trace"])
        expected = ReplayResult.model_validate(entry["expected"])
        harness.assert_identical(trace, expected)


def test_golden_covers_every_latent_state() -> None:
    payload = _load()
    states = {entry["latent_state"] for entry in payload["entries"]}
    assert states == {
        "repairable",
        "already_correct",
        "ambiguous",
        "underdetermined",
        "inconsistent",
        "adversarial_trap",
    }
