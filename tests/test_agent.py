import pytest

from genai_assistant.agent import execute_tool_call
from langchain_core.messages import AIMessage
from genai_assistant.agent import run_agent_loop

class AlwaysCallingLLM:
    """Fake LLM that always requests another tool."""

    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1

        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "multiply",
                    "args": {"a": 2, "b": 3},
                    "id": f"call_{self.calls}",
                    "type": "tool_call"
                }
            ]
        )

class DirectAnswerLLM:
    """Fake LLM that always returns a final answer."""

    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1

        return AIMessage(
            content="Hello! How can I help you?"
        )

def test_execute_valid_tool():
    tool_call = {
        "name": "multiply",
        "args": {"a": 12, "b": 5}
    }

    result = execute_tool_call(tool_call)

    assert result == 60


def test_execute_unknown_tool():
    tool_call = {
        "name": "delete_all_files",
        "args": {}
    }

    with pytest.raises(
        ValueError,
        match="Unknown tool: delete_all_files"
    ):
        execute_tool_call(tool_call)

def test_agent_respects_max_tool_rounds():
    fake_llm = AlwaysCallingLLM()

    with pytest.raises(
        RuntimeError,
        match="Maximum number of tool rounds exceeded"
    ):
        run_agent_loop(
            fake_llm,
            "Keep multiplying numbers",
            max_tool_rounds=2
        )

    assert fake_llm.calls == 3

def test_agent_returns_direct_answer():
    fake_llm = DirectAnswerLLM()

    response = run_agent_loop(
        fake_llm,
        "Hello!",
        max_tool_rounds=2
    )

    assert response.content == "Hello! How can I help you?"
    assert response.tool_calls == []
    assert fake_llm.calls == 1