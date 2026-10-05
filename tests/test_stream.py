"""tiered_ask.stream() event mapping, driven by stand-in LangGraph graphs
whose nodes never call a real LLM: a FakeListChatModel streams the
answer tokens, and tool calls are hand-built AIMessages. Proves the
callback→event translation, buffering of agent-turn tokens, escalation,
and the final event shape."""

import asyncio

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from tabletalk import tiered_ask


class S(TypedDict, total=False):
    question: str
    history: str
    sql: str
    result_columns: list
    result_rows: list
    error: object
    answer: str
    messages: list
    tools_used: list


def _fixed_graph(rows):
    llm = FakeListChatModel(responses=["forty-two"])

    def retrieve(s):
        return {"error": None}

    def write_sql(s):
        return {"sql": "SELECT 42"}

    def execute(s):
        return {"result_columns": ["n"], "result_rows": rows, "error": None}

    def answer(s):
        return {"answer": llm.invoke([HumanMessage("q")]).content}

    g = StateGraph(S)
    for name, fn in [("retrieve", retrieve), ("write_sql", write_sql), ("execute", execute), ("answer", answer)]:
        g.add_node(name, fn)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "write_sql")
    g.add_edge("write_sql", "execute")
    g.add_edge("execute", "answer")
    g.add_edge("answer", END)
    return g.compile()


def _agent_graph():
    llm = FakeListChatModel(responses=["the agent answer"])

    def seed(s):
        return {"messages": [], "tools_used": []}

    def agent(s):
        if not s.get("tools_used"):
            return {"messages": [AIMessage(content="", tool_calls=[{"name": "run_sql", "args": {"sql": "SELECT 1"}, "id": "c1"}])]}
        return {"messages": [AIMessage(content=llm.invoke([HumanMessage("q")]).content)]}

    def tools(s):
        return {"tools_used": ["run_sql(sql='SELECT 1')"], "sql": "SELECT 1",
                "result_columns": ["x"], "result_rows": [[1]], "error": None}

    def finalize(s):
        return {"answer": s["messages"][-1].content}

    def route(s):
        return "tools" if s["messages"][-1].tool_calls else "finalize"

    g = StateGraph(S)
    for name, fn in [("seed", seed), ("agent", agent), ("tools", tools), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "seed")
    g.add_edge("seed", "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", "finalize": "finalize"})
    g.add_edge("tools", "agent")
    g.add_edge("finalize", END)
    return g.compile()


def _collect(**kw):
    async def run():
        return [ev async for ev in tiered_ask.stream("q", **kw)]

    return asyncio.run(run())


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(tiered_ask, "_fixed", None)
    monkeypatch.setattr(tiered_ask, "_agent", None)
    monkeypatch.setattr(tiered_ask, "build_agent_graph", _agent_graph)


class TestFixedPath:
    def test_events_in_order_and_final(self, monkeypatch):
        monkeypatch.setattr(tiered_ask, "build_graph", lambda: _fixed_graph([[42]]))
        evs = _collect()
        types = [e["type"] for e in evs]
        assert types[0] == "status" and evs[0]["node"] == "retrieve"
        assert [e["node"] for e in evs if e["type"] == "status"] == ["retrieve", "write_sql", "execute", "answer"]
        assert next(e for e in evs if e["type"] == "sql")["sql"] == "SELECT 42"
        table = next(e for e in evs if e["type"] == "table")
        assert table["columns"] == ["n"] and table["rows"] == [[42]]
        assert "".join(e["text"] for e in evs if e["type"] == "token") == "forty-two"
        final = evs[-1]
        assert final["type"] == "final" and final["answer"] == "forty-two"
        assert final["escalated"] is False and final["rows"] == [[42]]

    def test_mode_fixed_never_escalates(self, monkeypatch):
        monkeypatch.setattr(tiered_ask, "build_graph", lambda: _fixed_graph([]))
        evs = _collect(mode="fixed")
        assert not any(e.get("node") == "escalate" for e in evs)
        assert evs[-1]["escalated"] is False


class TestEscalation:
    def test_zero_rows_escalates_to_agent(self, monkeypatch):
        monkeypatch.setattr(tiered_ask, "build_graph", lambda: _fixed_graph([]))
        evs = _collect()
        esc = next(e for e in evs if e.get("node") == "escalate")
        assert esc["escalated"] is True and "0 rows" in esc["reason"]
        assert any(e["type"] == "tool_call" and e["name"] == "run_sql" for e in evs)
        # tokens from the tool-calling turn must not leak; only the final prose
        assert "".join(e["text"] for e in evs if e["type"] == "token") == "the agent answer"
        final = evs[-1]
        assert final["escalated"] is True and final["answer"] == "the agent answer"
        assert final["tools_used"][0].startswith("fixed_pipeline(")

    def test_mode_agent_skips_fixed(self, monkeypatch):
        monkeypatch.setattr(tiered_ask, "build_graph", lambda: 1 / 0)
        evs = _collect(mode="agent")
        assert [e["node"] for e in evs if e["type"] == "status"][0] == "seed"
        assert evs[-1]["answer"] == "the agent answer" and evs[-1]["escalated"] is False
