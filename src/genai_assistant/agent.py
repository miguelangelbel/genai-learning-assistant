"""Manual tool execution and agent orchestration."""

from genai_assistant.tools import multiply, add
from langchain_core.messages import ToolMessage
from langchain_core.messages import HumanMessage

import logging

logger = logging.getLogger(__name__)

AVAILABLE_TOOLS = [multiply,add]

TOOLS_BY_NAME = {
    tool.name: tool
    for tool in AVAILABLE_TOOLS
}


def execute_tool_call(tool_call):
    """Execute a tool requested by the LLM."""

    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    tool = TOOLS_BY_NAME.get(tool_name)

    if tool is None:
        raise ValueError(
            f"Unknown tool: {tool_name}"
        )

    return tool.invoke(tool_args)

def execute_tool_calls(ai_message):
    """Execute all requested tools and return their results as messages."""

    tool_messages = []

    for tool_call in ai_message.tool_calls:
        result = execute_tool_call(tool_call)

        logger.debug(
            "Tool %s returned: %s",
            tool_call["name"],
            result
        )

        tool_message = ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        )

        tool_messages.append(tool_message)

    return tool_messages


def run_agent_loop(llm_with_tools, question, max_tool_rounds=3):
    """Run an iterative LLM-tool cycle until a final answer is produced."""

    messages = [
        HumanMessage(content=question)
    ]

    tool_rounds = 0

    while True:
        ai_message = llm_with_tools.invoke(messages)

        messages.append(ai_message)

        if not ai_message.tool_calls:
            return ai_message

        if tool_rounds >= max_tool_rounds:
            raise RuntimeError(
                "Maximum number of tool rounds exceeded."
            )
        logger.debug(
            "Tool round %s: %s tool call(s)",
            tool_rounds + 1,
            len(ai_message.tool_calls)
        )

        for tool_call in ai_message.tool_calls:
            logger.debug(
                "Requested tool: %s",
                tool_call["name"]
            )

        tool_messages = execute_tool_calls(ai_message)

        messages.extend(tool_messages)

        tool_rounds += 1