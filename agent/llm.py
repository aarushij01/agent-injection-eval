"""LLM backends.

The graph only needs two things from a model:
    llm.bind_tools(schemas) -> model that may return tool calls
    model.invoke(messages)  -> AIMessage

GroqLLM is the real one. MockLLM and ScriptedLLM need no API key and exist so
the harness and tests can run offline.
"""
from __future__ import annotations

import os

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


def make_groq_llm(model: str | None = None, temperature: float = 0.0):
    """Real model via Groq. Needs GROQ_API_KEY in the environment or .env."""
    from dotenv import load_dotenv
    from langchain_groq import ChatGroq

    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    return ChatGroq(
        model=model or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        temperature=temperature,
        max_retries=3,
    )


class MockLLM:
    """A deliberately naive stand-in: search once, then repeat what it read.

    It is NOT a model. It exists to smoke-test the pipeline without an API key.
    Because it parrots tool output, it "falls for" most injections, which is a
    handy check that the graders detect them. Never report its numbers as results.
    """

    model_name = "mock"

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        question = messages[last_human].content
        tool_outputs = [m for m in messages[last_human + 1:] if isinstance(m, ToolMessage)]
        if not tool_outputs:
            return AIMessage(
                content="",
                tool_calls=[{"name": "search_articles", "args": {"query": question}, "id": "mock-call-1", "type": "tool_call"}],
            )
        return AIMessage(content=str(tool_outputs[-1].content)[:700])


class ScriptedLLM:
    """Replays a fixed list of AIMessages. Used by the unit tests."""

    model_name = "scripted"

    def __init__(self, responses: list[AIMessage]):
        self._responses = list(responses)
        self.calls: list[list] = []  # prompts received, for assertions

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        if not self._responses:
            return AIMessage(content="(script exhausted)")
        return self._responses.pop(0)


def tool_call(name: str, call_id: str = "call-1", **args) -> AIMessage:
    """Helper for tests: an AIMessage that requests one tool call."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])
