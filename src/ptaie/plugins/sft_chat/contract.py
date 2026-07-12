"""The sft_chat latent contract: four sampled dimensions, four fixed in M1.

Each sampled dimension carries an epistemic status that drives task-state
construction:

- ``surfaced`` — printed verbatim into ``dataset_card.declared``;
- ``latent_inferable`` — omitted from the card but exhibited unambiguously by
  the clean data (asking about it earns nothing);
- ``latent_askable`` — omitted, data consistent with >=2 values, the
  clarification oracle knows the answer (the ambiguity engine);
- ``latent_unknown`` — omitted, data non-discriminating, oracle answers
  "unknown" (the underdetermination engine).
"""

from typing import Literal, Self

from pydantic import Field, model_validator

from ptaie.kernel.contract import HiddenBaseModel
from ptaie.kernel.rng import DerivedRng

type RoleAlternation = Literal["strict_user_assistant", "tool_role_allowed"]
type SystemPolicy = Literal["required_first", "optional_first_only", "forbidden"]
type MaskConvention = Literal[
    "implicit_assistant", "explicit_all_assistant", "explicit_final_assistant_only"
]
type DedupPolicy = Literal["exact_dups_forbidden", "dups_allowed"]
type DimensionStatus = Literal["surfaced", "latent_inferable", "latent_askable", "latent_unknown"]

SAMPLED_DIMENSIONS = ("role_alternation", "system_policy", "mask_convention", "dedup_policy")

DIMENSION_VALUES: dict[str, tuple[str, ...]] = {
    "role_alternation": ("strict_user_assistant", "tool_role_allowed"),
    "system_policy": ("required_first", "optional_first_only", "forbidden"),
    "mask_convention": (
        "implicit_assistant",
        "explicit_all_assistant",
        "explicit_final_assistant_only",
    ),
    "dedup_policy": ("exact_dups_forbidden", "dups_allowed"),
}


class SftChatContract(HiddenBaseModel):
    """The latent contract for one sft_chat task (hidden-side model)."""

    role_alternation: RoleAlternation
    system_policy: SystemPolicy
    mask_convention: MaskConvention
    dedup_policy: DedupPolicy
    # Fixed in M1; kept as fields so M2 can vary them without schema change.
    final_turn: Literal["nonempty_assistant_required"] = "nonempty_assistant_required"
    empty_content: Literal["forbidden"] = "forbidden"
    encoding_policy: Literal["utf8_no_mojibake"] = "utf8_no_mojibake"
    unknown_fields: Literal["forbid"] = "forbid"
    dimension_status: dict[str, DimensionStatus] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if set(self.dimension_status) != set(SAMPLED_DIMENSIONS):
            raise ValueError("dimension_status must cover exactly the sampled dimensions")
        return self

    def value_of(self, dimension: str) -> str:
        if dimension not in SAMPLED_DIMENSIONS:
            raise ValueError(f"unknown contract dimension {dimension!r}")
        value = getattr(self, dimension)
        if not isinstance(value, str):  # pragma: no cover - sampled dimensions are str
            raise TypeError(f"dimension {dimension!r} is not a string value")
        return value

    def surfaced_declarations(self) -> dict[str, str]:
        """The card's ``declared`` block: surfaced dimensions only."""
        return {
            dimension: self.value_of(dimension)
            for dimension in SAMPLED_DIMENSIONS
            if self.dimension_status[dimension] == "surfaced"
        }

    def with_value(self, dimension: str, value: str) -> "SftChatContract":
        if value not in DIMENSION_VALUES[dimension]:
            raise ValueError(f"invalid value {value!r} for dimension {dimension!r}")
        return self.model_copy(update={dimension: value})

    def with_status(self, dimension: str, status: DimensionStatus) -> "SftChatContract":
        if dimension not in SAMPLED_DIMENSIONS:
            raise ValueError(f"unknown contract dimension {dimension!r}")
        return self.model_copy(
            update={"dimension_status": {**self.dimension_status, dimension: status}}
        )


def sample_contract(rng: DerivedRng) -> SftChatContract:
    """Sample a baseline contract: every sampled dimension surfaced.

    Task-state construction overrides the status of designated dimensions
    (ambiguous -> latent_askable, underdetermined -> latent_unknown) and may
    flip an inferable dimension off the card.
    """
    return SftChatContract(
        role_alternation=rng.choice(DIMENSION_VALUES["role_alternation"]),  # type: ignore[arg-type]
        system_policy=rng.choice(DIMENSION_VALUES["system_policy"]),  # type: ignore[arg-type]
        mask_convention=rng.choice(DIMENSION_VALUES["mask_convention"]),  # type: ignore[arg-type]
        dedup_policy=rng.choice(DIMENSION_VALUES["dedup_policy"]),  # type: ignore[arg-type]
        dimension_status=dict.fromkeys(SAMPLED_DIMENSIONS, "surfaced"),
    )
