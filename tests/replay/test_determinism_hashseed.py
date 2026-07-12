"""Run the determinism check twice under different PYTHONHASHSEED values and
assert byte-identical output — the local mirror of the CI determinism job."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.replay

_SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "determinism_check.py"


def _run(hashseed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": hashseed}
    result = subprocess.run(  # noqa: S603 - fixed args, no shell
        [sys.executable, str(_SCRIPT), "--seeds", "0:40", "--out", "-"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return result.stdout


def test_determinism_is_independent_of_hash_seed() -> None:
    assert _run("0") == _run("524287")
