"""No wall-clock time or uncontrolled randomness anywhere in kernel or plugins.

``event_index`` is the only clock; ``ptaie.kernel.rng`` is the only
randomness source. Beyond literal imports, the scan also flags the natural
evasions: ``os.urandom``/``os.getrandom`` calls and dynamic imports of banned
modules via ``__import__``/``importlib.import_module`` with literal names.
"""

import ast
from pathlib import Path

import ptaie

PACKAGE_DIR = Path(ptaie.__file__).parent
SCANNED_SUBPACKAGES = ("kernel", "plugins", "policies")
BANNED_MODULES = {"time", "datetime", "uuid", "secrets"}
BANNED_OS_FUNCS = {"urandom", "getrandom"}
RANDOM_ALLOWED_IN = {PACKAGE_DIR / "kernel" / "rng.py"}


def _iter_source_files() -> list[Path]:
    files: list[Path] = []
    for subpackage in SCANNED_SUBPACKAGES:
        subdir = PACKAGE_DIR / subpackage
        if subdir.is_dir():
            files.extend(sorted(subdir.rglob("*.py")))
    return files


def _imported_top_level(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def _dynamic_violations(tree: ast.AST) -> set[str]:
    hits: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # __import__("time") / importlib.import_module("time")
        dynamic_import = (isinstance(func, ast.Name) and func.id == "__import__") or (
            isinstance(func, ast.Attribute) and func.attr == "import_module"
        )
        if (
            dynamic_import
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            top = node.args[0].value.split(".")[0]
            if top in BANNED_MODULES or top == "random":
                hits.add(f"dynamic import of {node.args[0].value!r}")
        # os.urandom(...) / os.getrandom(...) / bare urandom(...) via from-import
        if (
            isinstance(func, ast.Attribute)
            and func.attr in BANNED_OS_FUNCS
            and isinstance(func.value, ast.Name)
            and func.value.id == "os"
        ):
            hits.add(f"os.{func.attr}")
        if isinstance(func, ast.Name) and func.id in BANNED_OS_FUNCS:
            hits.add(func.id)
    return hits


def test_no_wall_clock_or_uncontrolled_randomness() -> None:
    violations: list[str] = []
    for path in _iter_source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        relative = path.relative_to(PACKAGE_DIR)
        imported = _imported_top_level(tree)
        banned_hits = imported & BANNED_MODULES
        if banned_hits:
            violations.append(f"{relative}: {sorted(banned_hits)}")
        if "random" in imported and path not in RANDOM_ALLOWED_IN:
            violations.append(f"{relative}: bare 'random'")
        dynamic = _dynamic_violations(tree)
        if dynamic:
            violations.append(f"{relative}: {sorted(dynamic)}")
    assert not violations, f"nondeterminism sources found: {violations}"


def test_scanner_catches_the_evasions_it_claims_to() -> None:
    """Self-test: the scanner must flag the documented evasion paths."""
    evasive = ast.parse(
        "import importlib\n"
        "import os\n"
        "importlib.import_module('time')\n"
        "__import__('uuid')\n"
        "os.urandom(8)\n"
    )
    hits = _dynamic_violations(evasive)
    assert {"dynamic import of 'time'", "dynamic import of 'uuid'", "os.urandom"} <= hits
