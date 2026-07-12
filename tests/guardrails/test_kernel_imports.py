"""Architecture as executable assertions: the kernel import wall.

``ptaie.kernel`` may import only the standard library, pydantic, and itself
(plus ``ptaie.version``). In particular it must never import plugins,
policies, or adapters — the kernel defines semantics, everything else plugs
into it.

Relative imports are resolved to absolute module paths (a ``from .. import``
can escape the kernel package), and ``from ptaie import X`` surfaces
``ptaie.X`` so package-attribute aliasing cannot dodge the wall.
"""

import ast
import sys
from pathlib import Path

import ptaie

PACKAGE_DIR = Path(ptaie.__file__).parent
KERNEL_DIR = PACKAGE_DIR / "kernel"
ALLOWED_THIRD_PARTY = {"pydantic"}
ALLOWED_PTAIE_PREFIXES = ("ptaie.kernel", "ptaie.version")


def _module_package(path: Path) -> list[str]:
    """The package parts of the module at ``path``, e.g. kernel/store/x.py
    -> ["ptaie", "kernel", "store"]."""
    relative = path.relative_to(PACKAGE_DIR)
    parts = ["ptaie", *relative.parent.parts]
    if path.name == "__init__.py":
        return parts
    return parts


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = _module_package(path)
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                # Resolve the relative import: level=1 is the current
                # package, each extra level walks one package up.
                base = package[: len(package) - (node.level - 1)]
                resolved = base + ([node.module] if node.module else [])
                modules.append(".".join(resolved))
            else:
                assert node.module is not None
                modules.append(node.module)
                # `from ptaie import plugins` must surface as ptaie.plugins:
                modules.extend(f"{node.module}.{alias.name}" for alias in node.names)
    return modules


def test_kernel_imports_only_stdlib_pydantic_and_itself() -> None:
    violations: list[str] = []
    for path in sorted(KERNEL_DIR.rglob("*.py")):
        for module in _imported_modules(path):
            top = module.split(".")[0]
            if top == "ptaie":
                if not module.startswith(ALLOWED_PTAIE_PREFIXES):
                    violations.append(f"{path.name}: {module}")
            elif top not in sys.stdlib_module_names and top not in ALLOWED_THIRD_PARTY:
                violations.append(f"{path.name}: {module}")
    assert not violations, f"kernel import wall violated: {violations}"


def test_kernel_never_imports_adapters_plugins_or_policies() -> None:
    for path in sorted(KERNEL_DIR.rglob("*.py")):
        for module in _imported_modules(path):
            for forbidden in ("ptaie.adapters", "ptaie.plugins", "ptaie.policies"):
                assert not module.startswith(forbidden), f"{path.name} imports {module}"


def test_relative_import_resolution_is_correct() -> None:
    """Self-test of the wall's resolver so a refactor can't silently gut it."""
    source = "from . import blob\nfrom .. import canonical\nfrom ..rng import DerivedRng\n"
    scratch = KERNEL_DIR / "store" / "workspace.py"
    tree = ast.parse(source)
    package = _module_package(scratch)
    resolved: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level > 0:
            base = package[: len(package) - (node.level - 1)]
            resolved.append(".".join(base + ([node.module] if node.module else [])))
    assert resolved == ["ptaie.kernel.store", "ptaie.kernel", "ptaie.kernel.rng"]
