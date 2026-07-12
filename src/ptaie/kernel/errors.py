"""Kernel error hierarchy.

Errors here are environment-internal. Agent-facing failures never cross the
protocol boundary as exceptions: tool failures become deterministic
``ToolOutcome`` error values and budget exhaustion becomes a truncated
``StepResult``.
"""


class KernelError(Exception):
    """Base class for all kernel errors."""


class CanonicalizationError(KernelError):
    """A value cannot be canonically serialized (non-JSON type, non-finite float, bad key)."""


class StoreError(KernelError):
    """Base class for artifact-store errors."""


class MissingBlobError(StoreError):
    """A blob hash is not present in the store."""


class PathViolationError(StoreError):
    """A workspace path is malformed, absolute, or attempts traversal."""


class WorkspacePathNotFoundError(StoreError):
    """A well-formed path does not exist in the staged manifest."""


class UnknownSnapshotError(StoreError):
    """A snapshot id does not name a recorded snapshot."""


class ManifestError(StoreError):
    """A manifest operation is invalid (e.g. removing a path that is absent)."""


class BudgetError(KernelError):
    """Budget accounting was used incorrectly (never raised by exhaustion itself)."""


class SealViolationError(KernelError):
    """The evidence-locked-commit ordering was violated."""


class RegistryError(KernelError):
    """A tool or plugin registration is invalid."""


class ReplayMismatchError(KernelError):
    """A replayed trace diverged from its recorded result."""

    def __init__(self, field: str, expected: object, actual: object) -> None:
        self.field = field
        self.expected = expected
        self.actual = actual
        super().__init__(f"replay mismatch on {field}: expected {expected!r}, got {actual!r}")


class TaskGenerationRejectedError(KernelError):
    """A task seed failed generation acceptance checks.

    Deterministic and typed: a rejected seed is never silently resampled.
    Callers use pre-validated seed lists.
    """

    def __init__(self, seed: int, failed_checks: tuple[str, ...]) -> None:
        self.seed = seed
        self.failed_checks = failed_checks
        super().__init__(f"task generation rejected for seed {seed}: {', '.join(failed_checks)}")
