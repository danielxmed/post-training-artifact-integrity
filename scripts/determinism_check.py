"""Determinism check: run the oracle over a seed range and serialize the
outcome signature of each episode.

CI runs this twice under *different* ``PYTHONHASHSEED`` values and byte-diffs
the outputs — a mismatch reveals any dict/set-iteration-order dependence that
an identical double-run would never surface.

Usage:
    python scripts/determinism_check.py --seeds 0:100 --out run.json
"""

import argparse
import json
import sys
from pathlib import Path

from ptaie.kernel.plugin import PluginRegistry
from ptaie.kernel.replay import ReplayHarness, ReplayResult
from ptaie.plugins.sft_chat.finalize import SftChatFinalizer
from ptaie.plugins.sft_chat.plugin import SftChatPlugin
from ptaie.plugins.sft_chat.taskgen import generate_sft_task
from ptaie.policies.oracle import OraclePolicy
from ptaie.version import ENV_VERSION


def _signature(result: ReplayResult) -> dict[str, object]:
    return {
        "seed": result.task_seed,
        "terminal_code": result.terminal_code.value if result.terminal_code else None,
        "reward_totals": result.reward_totals.model_dump(mode="json"),
        "constraint": result.constraint.model_dump(mode="json"),
        "audit_head_hash": result.audit_head_hash,
        "staged_root": result.staged_root,
        "steps": result.steps,
    }


def run(start: int, stop: int) -> list[dict[str, object]]:
    registry = PluginRegistry()
    registry.register(SftChatPlugin())
    harness = ReplayHarness(registry, SftChatFinalizer())
    signatures: list[dict[str, object]] = []
    for seed in range(start, stop):
        task = generate_sft_task(seed, ENV_VERSION)
        _, result = harness.record(
            OraclePolicy(task),
            task_seed=seed,
            artifact_class="sft_chat",
            env_version=ENV_VERSION,
        )
        signatures.append(_signature(result))
    return signatures


def _parse_range(spec: str) -> tuple[int, int]:
    start_str, stop_str = spec.split(":")
    return int(start_str), int(stop_str)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0:100", help="seed range START:STOP")
    parser.add_argument("--out", default="-", help="output path, or - for stdout")
    args = parser.parse_args(argv)
    start, stop = _parse_range(args.seeds)
    payload = {"env_version": ENV_VERSION, "signatures": run(start, stop)}
    text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)
    if args.out == "-":
        sys.stdout.write(text + "\n")
    else:
        with Path(args.out).open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
