"""Architecture as executable assertions: the kernel import wall.

``ptaie.kernel`` may import only the standard library, pydantic, and itself
(plus ``ptaie.version``). In particular it must never import plugins,
policies, or adapters — the kernel defines semantics, everything else plugs
into it.
"""

import ast
import sys
from pathlib import Path

import ptaie

KERNEL_DIR = Path(ptaie.__file__).parent / "kernel"
ALLOWED_THIRD_PARTY = {"pydantic"}
ALLOWED_PTAIE_PREFIXES = ("ptaie.kernel", "ptaie.version")


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                continue  # relative import: kernel-internal by construction
            if node.module is not None:
                modules.append(node.module)
    return modules


def test_kernel_imports_only_stdlib_pydantic_and_itself() -> None:
    violations: list[str] = []
    for path in sorted(KERNEL_DIR.rglob("*.py")):
        for module in _imported_modules(path):
            top = module.split(".")[0]
            if top == "ptaie":
                if not (module.startswith(ALLOWED_PTAIE_PREFIXES) or module == "ptaie"):
                    violations.append(f"{path.name}: {module}")
            elif top not in sys.stdlib_module_names and top not in ALLOWED_THIRD_PARTY:
                violations.append(f"{path.name}: {module}")
    assert not violations, f"kernel import wall violated: {violations}"


def test_kernel_never_imports_adapters_plugins_or_policies() -> None:
    for path in sorted(KERNEL_DIR.rglob("*.py")):
        for module in _imported_modules(path):
            for forbidden in ("ptaie.adapters", "ptaie.plugins", "ptaie.policies"):
                assert not module.startswith(forbidden), f"{path.name} imports {module}"
