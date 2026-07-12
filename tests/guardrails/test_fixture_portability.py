"""Fixtures must be portable: no Windows paths, no host-specific paths."""

import re
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
DRIVE_LETTER = re.compile(r"[A-Za-z]:[\\/]")
# A bare double-backslash is NOT a signal — it appears legitimately as
# JSON escaping in nested-JSON fixtures (e.g. golden traces). The real
# host-path signals are UNC hostnames, POSIX home paths, and drive letters.
FORBIDDEN_SUBSTRINGS = ("wsl.localhost", "/home/", "C:\\")


def test_fixtures_contain_no_host_specific_paths() -> None:
    if not FIXTURES_DIR.is_dir():
        return  # no fixtures yet — trivially portable
    violations: list[str] = []
    for path in sorted(FIXTURES_DIR.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue  # binary fixture: path rules don't apply
        for needle in FORBIDDEN_SUBSTRINGS:
            if needle in text:
                violations.append(f"{path.name}: contains {needle!r}")
        if DRIVE_LETTER.search(text):
            violations.append(f"{path.name}: contains a drive-letter path")
    assert not violations, f"non-portable fixtures: {violations}"
