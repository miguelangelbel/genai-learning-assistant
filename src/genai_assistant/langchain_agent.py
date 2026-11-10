"""LangChain agent orchestration helpers."""

from langgraph.errors import GraphRecursionError
from langchain.agents import create_agent
from genai_assistant.agent import AVAILABLE_TOOLS

class AgentExecutionLimitReached(RuntimeError):
    """Raised when an agent exceeds its execution budget."""


def build_langchain_agent(llm):
    """Create a LangChain agent with the registered tools."""

    return create_agent(
        model=llm,
        tools=AVAILABLE_TOOLS
    )

def run_langchain_agent(agent, question: str, recursion_limit: int = 10):
    """Execute a LangChain agent with a controlled execution limit."""

    try:
        return agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": question}
                ]
            },
            config={"recursion_limit": recursion_limit}
        )

    except GraphRecursionError as exc:
        raise AgentExecutionLimitReached(
            "Agent stopped after reaching its execution limit."
        ) from exc