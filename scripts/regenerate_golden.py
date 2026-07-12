"""Regenerate the committed golden replay traces.

Records the oracle's trace for one seed per latent state and writes the action
trace plus the outcome signature to ``tests/fixtures/golden_traces/``. A diff
in these fixtures during review is a semantic-change signal and requires an
``ENV_VERSION`` bump with justification.

Usage:
    python scripts/regenerate_golden.py
"""

import json
from pathlib import Path

from ptaie.kernel.plugin import PluginRegistry
from ptaie.kernel.replay import ReplayHarness
from ptaie.kernel.task import LatentTaskState
from ptaie.plugins.sft_chat.finalize import SftChatFinalizer
from ptaie.plugins.sft_chat.plugin import SftChatPlugin
from ptaie.plugins.sft_chat.taskgen import generate_sft_task
from ptaie.policies.oracle import OraclePolicy
from ptaie.version import ENV_VERSION

GOLDEN_PATH = Path(__file__).parent.parent / "tests" / "fixtures" / "golden_traces" / "oracle.json"


def _one_seed_per_state(limit: int = 2000) -> dict[str, int]:
    seen: dict[str, int] = {}
    for seed in range(limit):
        state = generate_sft_task(seed, ENV_VERSION).latent_state.value
        seen.setdefault(state, seed)
        if len(seen) == len(LatentTaskState):
            break
    return seen


def build() -> dict[str, object]:
    registry = PluginRegistry()
    registry.register(SftChatPlugin())
    harness = ReplayHarness(registry, SftChatFinalizer())
    entries: list[dict[str, object]] = []
    for state, seed in sorted(_one_seed_per_state().items()):
        task = generate_sft_task(seed, ENV_VERSION)
        trace, result = harness.record(
            OraclePolicy(task),
            task_seed=seed,
            artifact_class="sft_chat",
            env_version=ENV_VERSION,
        )
        entries.append(
            {
                "latent_state": state,
                "trace": trace.model_dump(mode="json"),
                "expected": result.model_dump(mode="json"),
            }
        )
    return {"env_version": ENV_VERSION, "entries": entries}


def main() -> int:
    payload = build()
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)
    with GOLDEN_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text + "\n")
    print(f"wrote {len(payload['entries'])} golden traces to {GOLDEN_PATH}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
