"""Cost-aware orchestration: try the fixed pipeline first (2 LLM calls,
cheap) and escalate to the tool-calling agent only when it fails or
silently returns nothing — the exact failure signature of a wrong
join/column guess that the agent's exploration can catch and the fixed
pipeline can't. Most questions never need the expensive path.

    rung 1  fixed pipeline, minimal reasoning effort      (cheap)
    rung 2  same pipeline, told what went wrong, high effort
    rung 3  tool-calling agent that explores the schema   (expensive)

state["tier"] records which rung produced the answer.
"""

from . import guardrail
from .agent import build_agent_graph
from .config import AUTO_LEARN, EFFORT_RETRY
from .graph import build_graph

_fixed = None
_agent = None


def should_escalate(state: dict) -> bool:
    """Pure decision logic, no I/O — testable without an LLM. True if the
    fixed pipeline's result looks unreliable: it errored, or it ran
    successfully but returned zero rows (indistinguishable at the SQL
    level from a legitimately empty answer, so we re-verify via the
    agent's exploration rather than risk shipping a silently wrong one)."""
    if state.get("error"):
        return True
    if state.get("sql") and state.get("result_columns") and not state.get("result_rows"):
        return True
    if state.get("verify_failed"):
        return True
    return False


def retry_note(state: dict) -> str:
    """What to tell the second-rung write_sql about the first attempt."""
    why = (
        f"error: {state['error']}" if state.get("error")
        else "; ".join(state.get("caveats") or []) or "it returned 0 rows"
    )
    return f"SQL: {state.get('sql')!r} -> {why}"


_REFUSALS = {
    guardrail.INJECTION_REFUSAL,
    guardrail.SCOPE_REFUSAL,
    guardrail.ROLE_BREAK_REFUSAL,
}


def ask(question: str, history: str = "", callbacks: list | None = None) -> dict:
    """Try build_graph() first; escalate to build_agent_graph() only if
    should_escalate() says the fixed-pipeline result looks unreliable.
    `history` is prior turns as "Q: ... / A: ..." text; `callbacks` attach
    LangChain callback handlers to every graph run."""
    global _fixed, _agent
    inputs = {"question": question, "history": history}
    config = {"callbacks": callbacks} if callbacks else None

    if _fixed is None:
        _fixed = build_graph()
    state = _fixed.invoke(inputs, config=config)
    state["tier"] = "fixed"
    attempts = []

    if should_escalate(state):
        # Rung 2: same cheap pipeline, but told what went wrong and allowed
        # to think hard. Catches most wrong guesses at a fraction of the
        # agent's cost.
        attempts.append(f"fixed_pipeline({retry_note(state)})")
        state = _fixed.invoke(
            {**inputs, "retry_note": retry_note(state), "effort": EFFORT_RETRY},
            config=config,
        )
        state["tier"] = "retry"

    if should_escalate(state):
        attempts.append(f"fixed_pipeline_retry({retry_note(state)})")
        if _agent is None:
            _agent = build_agent_graph()
        agent_state = _agent.invoke(inputs, config=config)
        agent_state["tools_used"] = attempts + list(agent_state.get("tools_used", []))
        agent_state["tier"] = "agent"
        state = agent_state

    state["escalated"] = state["tier"] != "fixed"
    _finish(question, state)
    return state


def should_learn(state: dict) -> bool:
    """Pure: is this run trustworthy enough to become an auto example?
    Yes only if SQL ran without error, returned rows, wasn't a refusal,
    the self-check (when on) didn't flag it, and the cheap first rung
    produced it — an escalated answer got there by exploration and its SQL
    is not necessarily the canonical way to ask the question."""
    return bool(
        state.get("sql")
        and not state.get("error")
        and state.get("result_rows")
        and not state.get("blocked")
        and not state.get("verify_failed")
        and state.get("confidence") != "low"
        and state.get("tier", "fixed") == "fixed"
        and not state.get("history")  # a follow-up's SQL depends on context the feed won't have
    )


def _finish(question: str, state: dict) -> None:
    state["blocked"] = state.get("answer") in _REFUSALS
    if AUTO_LEARN and should_learn(state):
        try:
            from .feed import add_example

            add_example(question, state["sql"], source="auto")
            state["learned"] = True
        except Exception:  # noqa: BLE001 — learning is best-effort
            state["learned"] = False


# --- streaming ------------------------------------------------------------
#
# Same decision logic as ask(), but yields typed events as the graphs run so
# UIs can show progress and the answer as it is written:
#
#   {"type": "status",    "node": "retrieve" | "write_sql" | "execute" |
#                                 "answer" | "agent" | "tools" | "finalize",
#                         "escalated": bool}
#   {"type": "tool_call", "name": ..., "args": {...}}          (agent only)
#   {"type": "sql",       "sql": ...}
#   {"type": "table",     "columns": [...], "rows": [[...]], "error": ...}
#   {"type": "token",     "text": ...}                          answer text
#   {"type": "final",     ...same keys as ask()'s return dict}
#
# Token events come from LangChain's on_chat_model_stream callback, which
# fires for any chat model invoked inside a node — the nodes themselves are
# unchanged. In the agent graph a model turn may be either a tool call or
# the final prose, and that's only known when the turn ends, so its tokens
# are buffered per turn and released only if no tool was called.

_FIXED_NODES = ("retrieve", "write_sql", "execute", "answer")
_AGENT_NODES = ("seed", "agent", "tools", "finalize")


def _chunk_text(chunk) -> str:
    content = getattr(chunk, "content", "")
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return content or ""


async def _run_graph(graph, inputs, config, nodes, escalated: bool, may_escalate: bool = False):
    """Drive one graph with astream_events, yielding stream events and
    finally ("__state__", final_state). With may_escalate, the fixed
    pipeline's answer tokens are withheld once its execute step shows the
    result will be escalated — that answer is discarded by the caller, so
    the user must not see it streamed first."""
    buffered: list[str] = []
    final_state = None
    seen_sql = ""
    suppress_answer = False
    async for ev in graph.astream_events(inputs, version="v2", config=config):
        kind = ev["event"]
        name = ev.get("name")
        node = (ev.get("metadata") or {}).get("langgraph_node")
        if kind == "on_chain_start" and name in nodes:
            yield {"type": "status", "node": name, "escalated": escalated}
            if name in ("agent", "finalize"):
                buffered = []
        elif kind == "on_chat_model_stream":
            text = _chunk_text(ev["data"].get("chunk"))
            if not text:
                continue
            if node == "answer":
                if not suppress_answer:
                    yield {"type": "token", "text": text}
            elif node in ("agent", "finalize"):
                buffered.append(text)
        elif kind == "on_chain_end" and name in nodes:
            out = ev["data"].get("output") or {}
            if not isinstance(out, dict):
                continue
            if name == "write_sql" and out.get("sql"):
                seen_sql = out["sql"]
                yield {"type": "sql", "sql": out["sql"]}
            elif name == "execute":
                if may_escalate and should_escalate({**out, "sql": seen_sql}):
                    suppress_answer = True
                else:
                    yield {
                        "type": "table",
                        "columns": out.get("result_columns") or [],
                        "rows": out.get("result_rows") or [],
                        "error": out.get("error"),
                    }
            elif name == "agent":
                msgs = out.get("messages") or []
                last = msgs[-1] if msgs else None
                calls = getattr(last, "tool_calls", None) or []
                if calls:
                    for call in calls:
                        yield {"type": "tool_call", "name": call["name"], "args": call["args"]}
                elif buffered:
                    for text in buffered:
                        yield {"type": "token", "text": text}
                    buffered = []
            elif name == "tools":
                if out.get("sql"):
                    yield {"type": "sql", "sql": out["sql"]}
                    yield {
                        "type": "table",
                        "columns": out.get("result_columns") or [],
                        "rows": out.get("result_rows") or [],
                        "error": out.get("error"),
                    }
            elif name == "finalize" and buffered:
                for text in buffered:
                    yield {"type": "token", "text": text}
                buffered = []
        elif kind == "on_chain_end" and name == "LangGraph":
            final_state = ev["data"].get("output")
    if final_state is None:
        final_state = await graph.ainvoke(inputs, config=config)
    yield ("__state__", final_state)


async def stream(
    question: str,
    history: str = "",
    callbacks: list | None = None,
    mode: str = "tiered",
):
    """Async generator version of ask(): same tiering, but yields progress/token/table events as they happen. The last
    event is always {"type": "final", ...}. `mode` — "tiered" (default),
    "fixed" (never escalate) or "agent" (skip the fixed pipeline) — is
    what the Chainlit switch and CLI flags select."""
    global _fixed, _agent
    inputs = {"question": question, "history": history}
    config = {"callbacks": callbacks} if callbacks else None

    state = None
    if mode != "agent":
        if _fixed is None:
            _fixed = build_graph()
        async for ev in _run_graph(
            _fixed, inputs, config, _FIXED_NODES, escalated=False,
            may_escalate=(mode != "fixed"),
        ):
            if isinstance(ev, tuple):
                state = ev[1]
            else:
                yield ev

    attempts: list[str] = []
    if mode == "agent":
        if _agent is None:
            _agent = build_agent_graph()
        async for ev in _run_graph(_agent, inputs, config, _AGENT_NODES, escalated=False):
            if isinstance(ev, tuple):
                state = ev[1]
            else:
                yield ev
        state["tier"] = "agent"
    else:
        state["tier"] = "fixed"

    if mode == "tiered" and should_escalate(state):
        note = retry_note(state)
        attempts.append(f"fixed_pipeline({note})")
        yield {"type": "status", "node": "retry", "escalated": True, "reason": note}
        async for ev in _run_graph(
            _fixed, {**inputs, "retry_note": note, "effort": EFFORT_RETRY}, config,
            _FIXED_NODES, escalated=True, may_escalate=True,
        ):
            if isinstance(ev, tuple):
                state = ev[1]
            else:
                yield ev
        state["tier"] = "retry"

    if mode == "tiered" and should_escalate(state):
        note = retry_note(state)
        attempts.append(f"fixed_pipeline_retry({note})")
        if _agent is None:
            _agent = build_agent_graph()
        yield {"type": "status", "node": "escalate", "escalated": True, "reason": note}
        agent_state = None
        async for ev in _run_graph(_agent, inputs, config, _AGENT_NODES, escalated=True):
            if isinstance(ev, tuple):
                agent_state = ev[1]
            else:
                yield ev
        agent_state["tools_used"] = attempts + list(agent_state.get("tools_used", []))
        agent_state["tier"] = "agent"
        state = agent_state

    state["escalated"] = mode == "tiered" and state["tier"] != "fixed"
    _finish(question, state)
    yield {
        "type": "final",
        "answer": state.get("answer", ""),
        "sql": state.get("sql", ""),
        "error": state.get("error"),
        "columns": state.get("result_columns") or [],
        "rows": state.get("result_rows") or [],
        "escalated": state["escalated"],
        "tier": state["tier"],
        "confidence": state.get("confidence"),
        "caveats": state.get("caveats") or [],
        "blocked": state["blocked"],
        "tools_used": state.get("tools_used") or [],
    }
