from unittest.mock import Mock

import pytest
from langgraph.errors import GraphRecursionError
from langchain_core.messages import AIMessage

from genai_assistant.langchain_agent import (
    AgentExecutionLimitReached,
    run_langchain_agent,
)


def test_langchain_agent_handles_execution_limit():
    fake_agent = Mock()

    fake_agent.invoke.side_effect = GraphRecursionError(
        "Recursion limit exceeded"
    )

    with pytest.raises(
        AgentExecutionLimitReached,
        match="Agent stopped after reaching its execution limit"
    ):
        run_langchain_agent(
            fake_agent,
            "Multiply 12 by 5",
            recursion_limit=2
        )

    fake_agent.invoke.assert_called_once_with(
        {
            "messages": [
                {"role": "user", "content": "Multiply 12 by 5"}
            ]
        },
        config={"recursion_limit": 2}
    )

def test_langchain_agent_returns_successful_response():
    fake_agent = Mock()

    expected_result = {
        "messages": [
            AIMessage(content="The final result is 68.")
        ]
    }

    fake_agent.invoke.return_value = expected_result

    result = run_langchain_agent(
        fake_agent,
        "Multiply 12 by 5 and add 8",
        recursion_limit=10
    )

    assert result == expected_result

    fake_agent.invoke.assert_called_once_with(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Multiply 12 by 5 and add 8"
                }
            ]
        },
        config={"recursion_limit": 10}
    )