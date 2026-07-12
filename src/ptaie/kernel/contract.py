"""Latent contracts: the hidden requirements a task is scored against.

Everything in this module is hidden-side state. ``HiddenBaseModel`` redacts
``repr``/``str`` as a leakage backstop: hidden values must never reach an
observation, an error message, or a log line through string formatting.
"""

from collections.abc import Iterable
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from ptaie.kernel.canonical import JsonValue


class HiddenBaseModel(BaseModel):
    """Base for hidden-side models: frozen, forbid-extra, redacted rendering.

    ``model_dump`` still exposes values — redaction guards accidental string
    interpolation, not deliberate serialization by environment code.
    ``hide_input_in_errors`` keeps pydantic ``ValidationError`` messages from
    echoing hidden field values; the empty ``__repr_args__`` starves
    rich/devtools formatters (``__rich_repr__``, ``__pretty__``) of values.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(<redacted>)"

    def __str__(self) -> str:
        return f"{type(self).__name__}(<redacted>)"

    def __repr_args__(self) -> Iterable[tuple[str | None, Any]]:
        return ()


class Requirement(HiddenBaseModel):
    """One contract requirement, evaluated by a registered predicate."""

    requirement_id: str
    predicate: str
    params: dict[str, JsonValue] = {}
    severity: Literal["hard", "soft"] = "hard"


class ContractVariant(HiddenBaseModel):
    """One plausible interpretation of the latent contract."""

    variant_id: str
    requirements: tuple[Requirement, ...]


class InconsistencyWitness(HiddenBaseModel):
    """Machine-checkable proof that two task assertions are jointly unsatisfiable."""

    assertion_a: str
    assertion_b: str
    rule_id: str
    note: str = ""


class LatentContract(HiddenBaseModel):
    """The latent contract: one or more variants plus prohibited side effects.

    - one variant ⇒ fully specified;
    - multiple variants with ``true_variant_id`` set ⇒ ambiguous but
      resolvable (a clarification distinguishes them);
    - multiple variants with ``true_variant_id=None`` ⇒ underdetermined (no
      ground-truth interpretation exists).
    """

    contract_id: str
    variants: tuple[ContractVariant, ...]
    true_variant_id: str | None = None
    prohibited_effects: tuple[Requirement, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.variants:
            raise ValueError("contract must have at least one variant")
        variant_ids = [variant.variant_id for variant in self.variants]
        if len(set(variant_ids)) != len(variant_ids):
            raise ValueError("variant ids must be unique")
        if self.true_variant_id is not None and self.true_variant_id not in variant_ids:
            raise ValueError("true_variant_id must name one of the variants")
        return self

    @property
    def true_variant(self) -> ContractVariant | None:
        if self.true_variant_id is None:
            return None
        for variant in self.variants:
            if variant.variant_id == self.true_variant_id:
                return variant
        raise AssertionError("unreachable: validated at construction")
