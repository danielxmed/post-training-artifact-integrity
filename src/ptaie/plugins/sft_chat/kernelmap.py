"""Bridge between the sft_chat native contract and the kernel LatentContract.

The kernel stays domain-agnostic, so the sft-specific ``SftChatContract`` and
generation parameters (the count floor) ride inside a kernel
``ContractVariant`` as one requirement's params. Verifiers and scoring decode
them back at finalization.
"""

from ptaie.kernel.contract import ContractVariant, LatentContract, Requirement
from ptaie.plugins.sft_chat.contract import SftChatContract

SPEC_PREDICATE = "sft_chat.spec"
SPEC_REQUIREMENT_ID = "sft_chat.spec"


def spec_variant(variant_id: str, contract: SftChatContract, count_floor: int) -> ContractVariant:
    return ContractVariant(
        variant_id=variant_id,
        requirements=(
            Requirement(
                requirement_id=SPEC_REQUIREMENT_ID,
                predicate=SPEC_PREDICATE,
                params={"contract": contract.model_dump(mode="json"), "count_floor": count_floor},
                severity="hard",
            ),
        ),
    )


def decode_spec(variant: ContractVariant) -> tuple[SftChatContract, int]:
    requirement = variant.requirements[0]
    contract = SftChatContract.model_validate(requirement.params["contract"])
    count_floor = int(requirement.params["count_floor"])  # type: ignore[arg-type]
    return contract, count_floor


def single_variant_contract(
    contract_id: str, contract: SftChatContract, count_floor: int
) -> LatentContract:
    return LatentContract(
        contract_id=contract_id,
        variants=(spec_variant("v0", contract, count_floor),),
        true_variant_id="v0",
    )
