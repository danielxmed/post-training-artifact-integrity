"""No wall-clock time or uncontrolled randomness anywhere in kernel or plugins.

``event_index`` is the only clock; ``ptaie.kernel.rng`` is the only
randomness source (guardrail: ``random`` may be imported there and nowhere
else).
"""

import ast
from pathlib import Path

import ptaie

PACKAGE_DIR = Path(ptaie.__file__).parent
SCANNED_SUBPACKAGES = ("kernel", "plugins", "policies")
BANNED_MODULES = {"time", "datetime", "uuid", "secrets"}
RANDOM_ALLOWED_IN = {PACKAGE_DIR / "kernel" / "rng.py"}


def _iter_source_files() -> list[Path]:
    files: list[Path] = []
    for subpackage in SCANNED_SUBPACKAGES:
        subdir = PACKAGE_DIR / subpackage
        if subdir.is_dir():
            files.extend(sorted(subdir.rglob("*.py")))
    return files


def _imported_top_level(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def test_no_wall_clock_or_uncontrolled_randomness() -> None:
    violations: list[str] = []
    for path in _iter_source_files():
        imported = _imported_top_level(path)
        banned_hits = imported & BANNED_MODULES
        if banned_hits:
            violations.append(f"{path.relative_to(PACKAGE_DIR)}: {sorted(banned_hits)}")
        if "random" in imported and path not in RANDOM_ALLOWED_IN:
            violations.append(f"{path.relative_to(PACKAGE_DIR)}: bare 'random'")
    assert not violations, f"nondeterminism sources found: {violations}"
