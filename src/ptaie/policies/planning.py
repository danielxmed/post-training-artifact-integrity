"""Turn a target data file into a sequence of line-level tool actions.

Shared by the oracle (and available to other scripted policies). The plan
deletes trailing extras first, then replaces changed lines and appends new
ones, so applying it in order drives the workspace data to ``target_data``.
"""

from ptaie.kernel.actions import ToolAction
from ptaie.plugins.sft_chat.schema import split_data_lines


def plan_rewrite(current_data: bytes, target_data: bytes) -> list[ToolAction]:
    current = split_data_lines(current_data)
    target = split_data_lines(target_data)
    actions: list[ToolAction] = []

    length = len(current)
    while length > len(target):
        actions.append(
            ToolAction(tool_name="sft_chat.delete_record", arguments={"line_index": length - 1})
        )
        length -= 1

    for index, line in enumerate(target):
        if index < length:
            if current[index] != line:
                actions.append(
                    ToolAction(
                        tool_name="sft_chat.replace_record",
                        arguments={"line_index": index, "raw_line": line},
                    )
                )
        else:
            actions.append(
                ToolAction(
                    tool_name="sft_chat.insert_record",
                    arguments={"line_index": index, "raw_line": line},
                )
            )
    return actions
