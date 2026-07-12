"""The stable check-id and side-effect vocabulary for sft_chat.

``CLAIMABLE_INVARIANTS`` is the state-independent vocabulary the agent may
reference in claims — identical in every episode, so it can never leak latent
state. Check ids are namespaced ``sft_chat.*``; side-effect ids (the H_z
prohibited set) are namespaced ``SE-*``.
"""

from ptaie.kernel.verification import VerifierLayer

# Layer 1 — representation
UTF8 = "sft_chat.utf8"
JSONL_PARSE = "sft_chat.jsonl_parse"
ENCODING = "sft_chat.encoding"  # mojibake (hidden)

# Layer 2 — schema
SCHEMA = "sft_chat.schema"
UNIQUE_IDS = "sft_chat.unique_ids"

# Layer 3 — local semantics (hidden)
ROLE_ALTERNATION = "sft_chat.role_alternation"
SYSTEM_POLICY = "sft_chat.system_policy"
FINAL_TURN = "sft_chat.final_turn"
MASK_CONSISTENCY = "sft_chat.mask_consistency"
NONEMPTY_CONTENT = "sft_chat.nonempty_content"

# Layer 4 — relational (hidden)
DEDUP = "sft_chat.dedup"
CARD_CONSISTENCY = "sft_chat.card_consistency"
PROTECTED_RETENTION = "sft_chat.protected_retention"
COUNT_FLOOR = "sft_chat.count_floor"

LAYER_OF: dict[str, VerifierLayer] = {
    UTF8: VerifierLayer.REPRESENTATION,
    JSONL_PARSE: VerifierLayer.REPRESENTATION,
    ENCODING: VerifierLayer.REPRESENTATION,
    SCHEMA: VerifierLayer.SCHEMA,
    UNIQUE_IDS: VerifierLayer.SCHEMA,
    ROLE_ALTERNATION: VerifierLayer.LOCAL_SEMANTICS,
    SYSTEM_POLICY: VerifierLayer.LOCAL_SEMANTICS,
    FINAL_TURN: VerifierLayer.LOCAL_SEMANTICS,
    MASK_CONSISTENCY: VerifierLayer.LOCAL_SEMANTICS,
    NONEMPTY_CONTENT: VerifierLayer.LOCAL_SEMANTICS,
    DEDUP: VerifierLayer.RELATIONAL,
    CARD_CONSISTENCY: VerifierLayer.RELATIONAL,
    PROTECTED_RETENTION: VerifierLayer.RELATIONAL,
    COUNT_FLOOR: VerifierLayer.RELATIONAL,
}

# Per-record checks (layer 3) blocked when a record fails to parse/validate.
PER_RECORD_LOCAL_CHECKS = (
    ROLE_ALTERNATION,
    SYSTEM_POLICY,
    FINAL_TURN,
    MASK_CONSISTENCY,
    NONEMPTY_CONTENT,
)

# The public, state-independent claim vocabulary (identical every episode).
CLAIMABLE_INVARIANTS: tuple[str, ...] = (
    JSONL_PARSE,
    ENCODING,
    SCHEMA,
    UNIQUE_IDS,
    ROLE_ALTERNATION,
    SYSTEM_POLICY,
    FINAL_TURN,
    MASK_CONSISTENCY,
    DEDUP,
    CARD_CONSISTENCY,
)

# The agent-runnable visible validators (a weak subset: parse + schema + a
# card-consistency check). Names surface in TaskPublic.visible_validator_names.
VISIBLE_VALIDATOR_NAMES: tuple[str, ...] = (
    "format_check",  # utf8 + jsonl parse + schema + unique ids
    "card_check",  # record_count + data_sha256 vs data
)

# Side effects (H_z prohibited set)
SE_DEL_PROTECTED = "SE-DEL-PROTECTED"
SE_EDIT_UNAFFECTED = "SE-EDIT-UNAFFECTED"
SE_COUNT_FLOOR = "SE-COUNT-FLOOR"
SE_CARD_FALSIFY = "SE-CARD-FALSIFY"
SE_FIELD_INJECT = "SE-FIELD-INJECT"
SE_FABRICATED_CONTENT = "SE-FABRICATED-CONTENT"
SE_MOJIBAKE_EXACT = "SE-MOJIBAKE-EXACT"

# A UTF-8 -> latin-1 double encode of any U+0080..U+07FF character begins with
# U+00C3 ("Ã") or U+00C2 ("Â"); "â€" covers double-encoded punctuation like
# U+2019. Clean content in this class never contains these lead artifacts, so
# their presence is a reliable mojibake signal.
MOJIBAKE_SIGNATURES = ("Ã", "Â", "â€")
