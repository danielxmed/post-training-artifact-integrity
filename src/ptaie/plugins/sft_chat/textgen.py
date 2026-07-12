"""Seeded record content generation — no LLM, no network, no wall clock.

Content is composed from fixed word banks and three conversation templates
(single-turn QA, multi-turn instruction, tool-flavored dialog).

Two deliberate textgen invariants:

1. every clean assistant message ends with terminal punctuation (``.``,
   ``!``, ``?``) — this makes truncation semantically detectable at layer 3
   without hidden reference diffs;
2. word banks include accented (non-ASCII) words, so mojibake corruption has
   material to double-encode and the encoding layer has real signal.
"""

from ptaie.kernel.rng import DerivedRng
from ptaie.plugins.sft_chat.contract import SftChatContract
from ptaie.plugins.sft_chat.schema import ChatMessage, ChatRecord

_HEX = "0123456789abcdef"

_TOPICS = (
    "café logistics",
    "résumé screening",
    "naïve baselines",
    "protégé onboarding",
    "crème brûlée recipes",
    "façade rendering",
    "el niño forecasts",
    "smörgåsbord planning",
)

_QUESTION_STEMS = (
    "Could you explain",
    "What is the best approach to",
    "How would you summarize",
    "Please compare options for",
    "What should I know about",
)

_ANSWER_STEMS = (
    "Here is a concise overview of",
    "The recommended approach to",
    "A practical summary of",
    "The key considerations for",
)

_FOLLOWUP_STEMS = (
    "Can you elaborate on the details of",
    "What are the risks involved in",
    "How does this change for",
)

_DETAILS = (
    "the déjà vu edge case",
    "the naïve baseline",
    "the jalapeño variant",
    "the entrée selection",
    "the château scenario",
)

_SYSTEM_PROMPTS = (
    "You are a helpful assistant.",
    "You are a careful, concise assistant.",
    "You are an assistant that answers precisely.",
)

_TOOL_OBSERVATIONS = (
    '{"status": "ok", "rows": 3}',
    '{"status": "ok", "value": "42"}',
    '{"status": "ok", "items": ["café", "résumé"]}',
)

TERMINAL_PUNCTUATION = (".", "!", "?")


def _record_id(rng: DerivedRng) -> str:
    return "rec-" + "".join(rng.choice(_HEX) for _ in range(8))


def _question(rng: DerivedRng) -> str:
    return f"{rng.choice(_QUESTION_STEMS)} {rng.choice(_TOPICS)}?"


def _answer(rng: DerivedRng) -> str:
    sentences = [f"{rng.choice(_ANSWER_STEMS)} {rng.choice(_TOPICS)}."]
    if rng.random() < 0.5:
        sentences.append(f"Note {rng.choice(_DETAILS)} before committing to it.")
    return " ".join(sentences)


def _followup(rng: DerivedRng) -> str:
    return f"{rng.choice(_FOLLOWUP_STEMS)} {rng.choice(_DETAILS)}?"


def _mask_value(role: str, is_final_assistant: bool, contract: SftChatContract) -> bool | None:
    """The loss flag a clean record carries for this message, or None (absent)."""
    if contract.mask_convention == "implicit_assistant":
        return None
    if role == "assistant":
        if contract.mask_convention == "explicit_all_assistant":
            return True
        return is_final_assistant  # explicit_final_assistant_only
    return False


def _apply_mask(
    messages: list[tuple[str, str]], contract: SftChatContract
) -> tuple[ChatMessage, ...]:
    final_assistant_index = max(
        (index for index, (role, _) in enumerate(messages) if role == "assistant"),
        default=-1,
    )
    return tuple(
        ChatMessage(
            role=role,
            content=content,
            loss=_mask_value(role, index == final_assistant_index, contract),
        )
        for index, (role, content) in enumerate(messages)
    )


def _template_single_qa(rng: DerivedRng, contract: SftChatContract) -> list[tuple[str, str]]:
    return [("user", _question(rng)), ("assistant", _answer(rng))]


def _template_multi_turn(rng: DerivedRng, contract: SftChatContract) -> list[tuple[str, str]]:
    return [
        ("user", _question(rng)),
        ("assistant", _answer(rng)),
        ("user", _followup(rng)),
        ("assistant", _answer(rng)),
    ]


def _template_tool_dialog(rng: DerivedRng, contract: SftChatContract) -> list[tuple[str, str]]:
    if contract.role_alternation != "tool_role_allowed":
        return _template_multi_turn(rng, contract)
    return [
        ("user", _question(rng)),
        ("assistant", f"Let me look that up for {rng.choice(_DETAILS)}."),
        ("tool", rng.choice(_TOOL_OBSERVATIONS)),
        ("assistant", _answer(rng)),
    ]


_TEMPLATES = (_template_single_qa, _template_multi_turn, _template_tool_dialog)


def build_record(rng: DerivedRng, contract: SftChatContract, *, template_index: int) -> ChatRecord:
    template = _TEMPLATES[template_index % len(_TEMPLATES)]
    messages = template(rng, contract)
    if contract.system_policy == "required_first" or (
        contract.system_policy == "optional_first_only" and rng.random() < 0.5
    ):
        messages = [("system", rng.choice(_SYSTEM_PROMPTS)), *messages]
    return ChatRecord(id=_record_id(rng), messages=_apply_mask(messages, contract))


def build_records(rng: DerivedRng, contract: SftChatContract, count: int) -> list[ChatRecord]:
    """Build ``count`` clean records. At least one multi-turn record (two
    assistant turns) is always present so mask conventions are inferable and
    truncation has trim room."""
    records = []
    for index in range(count):
        # index 1 is forced multi-turn; the rest cycle through templates
        template_index = 1 if index == 1 else rng.randint(0, len(_TEMPLATES) - 1)
        records.append(
            build_record(rng.substream(f"record-{index}"), contract, template_index=template_index)
        )
    return records
