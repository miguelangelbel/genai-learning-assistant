"""Tools available to the GenAI Learning Assistant."""

from langchain_core.tools import tool

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b