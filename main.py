"""TableTalk CLI.

Usage:
    python main.py ingest                             # full rebuild of the vector feed
    python main.py sync                               # re-sync schema cards from the live DB
    python main.py ask "question"                     # one-shot question (tiered: pipeline -> retry -> agent)
    python main.py chat                               # interactive chat (in-memory history)
    python main.py tables                             # list tables in the target DB
    python main.py eval                               # execution accuracy against golden/golden.json
    python main.py feed list                          # list all feed entries
    python main.py feed add-example "question" "sql"  # add an NL->SQL example
    python main.py feed add-note "text" [table]       # add a business note
    python main.py feed delete <id>                   # delete a feed entry by id

ask, chat and eval take --fixed (fixed pipeline only, never escalate) or
--agent (tool-calling agent only) in place of the default tiered mode.
"""

import sys


HISTORY_TURNS = 3  # prior Q/A turns replayed into each chat question


def _ask(mode: str, question: str, history: str = "") -> str:
    if mode == "tiered":
        from tabletalk.tiered_ask import ask as tiered_ask

        state = tiered_ask(question, history)
        print(f"\n[answered by: {state['tier']}]")
    else:
        inputs = {"question": question, "history": history}
        state = _build_graph(agent=(mode == "agent")).invoke(inputs)
    if state.get("sql"):
        print(f"\nSQL: {state['sql']}")
    elif state.get("tools_used"):
        print("\nNo SQL was needed. Tools used:")
        for call in state["tools_used"]:
            print(f"  {call}")
    else:
        print("\nSQL: (none)")
    if state.get("error"):
        print(f"Error: {state['error']}")
    print(f"\n{state['answer']}\n")
    return state["answer"]


def cmd_ingest() -> None:
    from tabletalk.feed import rebuild

    counts = rebuild()
    print(
        f"Rebuilt feed: {counts['schema']} schema cards, "
        f"{counts['examples']} examples, {counts['notes']} notes."
    )


def cmd_sync() -> None:
    from tabletalk.feed import sync_schema

    result = sync_schema()
    print(f"Schema synced: {result['synced']} cards, {result['removed']} removed.")


def _build_graph(agent: bool = False):
    if agent:
        from tabletalk.agent import build_agent_graph

        return build_agent_graph()
    from tabletalk.graph import build_graph

    return build_graph()


def cmd_ask(question: str, mode: str = "tiered") -> None:
    _ask(mode, question)


def cmd_chat(mode: str = "tiered") -> None:
    history: list[str] = []
    print("TableTalk chat — empty line or 'exit' to quit.")
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in ("exit", "quit"):
            break
        answer = _ask(mode, question, history="\n".join(history[-HISTORY_TURNS:]))
        history.append(f"Q: {question}\nA: {answer}")


def cmd_tables() -> None:
    from tabletalk.db import list_tables, table_info

    for name in list_tables():
        info = table_info(name, samples=0)
        print(f"{name}  ({info.row_count} rows, {len(info.columns)} columns)")


def _norm_value(v) -> str:
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _row_contains(generated_row, expected_row) -> bool:
    """True if every value of the expected row appears in the generated row
    (as a multiset) — tolerates extra columns and different column order."""
    pool = [_norm_value(v) for v in generated_row]
    for v in (_norm_value(v) for v in expected_row):
        if v in pool:
            pool.remove(v)
        else:
            return False
    return True


def rows_match(generated, expected) -> bool:
    if len(generated) != len(expected):
        return False
    remaining = [list(r) for r in generated]
    for exp_row in expected:
        for i, gen_row in enumerate(remaining):
            if _row_contains(gen_row, exp_row):
                remaining.pop(i)
                break
        else:
            return False
    return True


def cmd_eval(mode: str = "tiered") -> None:
    import json
    from pathlib import Path

    from tabletalk import guardrail
    from tabletalk.db import run_sql
    from tabletalk.stats import RunStats

    golden = json.loads(
        (Path(__file__).parent / "golden" / "golden.json").read_text(encoding="utf-8")
    )
    if not golden:
        print("golden/golden.json is empty — nothing to evaluate.")
        return
    graph = None if mode == "tiered" else _build_graph(agent=(mode == "agent"))
    passed = 0
    escalated = 0
    tiers: dict[str, int] = {}
    total_ms = 0
    total_tokens = 0
    total_calls = 0
    for item in golden:
        stats = RunStats()
        if mode == "tiered":
            from tabletalk.tiered_ask import ask as tiered_ask

            state = tiered_ask(item["question"], "", callbacks=[stats])
            if state.get("escalated"):
                escalated += 1
            tiers[state.get("tier", "?")] = tiers.get(state.get("tier", "?"), 0) + 1
        else:
            state = graph.invoke(
                {"question": item["question"], "history": ""},
                config={"callbacks": [stats]},
            )
        total_ms += stats.elapsed_ms
        total_tokens += stats.total_tokens
        total_calls += stats.llm_calls
        cost = f"[{stats.elapsed_ms}ms, {stats.llm_calls} llm, {stats.total_tokens} tok]"

        expect = item.get("expect")
        if expect:
            # Refusal cases: the answer must be the refusal text and no SQL
            # may have been executed.
            refusal = {
                "out_of_scope": guardrail.SCOPE_REFUSAL,
                "blocked": guardrail.INJECTION_REFUSAL,
            }[expect]
            if state.get("answer") == refusal and not state.get("result_rows"):
                passed += 1
                print(f"pass  {item['question']}  {cost}")
            else:
                print(f"FAIL  {item['question']}  {cost}\n"
                      f"      expected {expect}; got: {state.get('answer', '')[:120]!r}")
            continue

        expected = run_sql(item["sql"])
        if state.get("error"):
            print(f"FAIL  {item['question']}  {cost}\n      error: {state['error']}")
            continue
        if rows_match(state.get("result_rows", []), expected.rows):
            passed += 1
            print(f"pass  {item['question']}  {cost}")
        else:
            print(f"FAIL  {item['question']}  {cost}\n      generated: {state.get('sql')}")
    n = len(golden)
    print(f"\nExecution accuracy: {passed}/{n} ({100 * passed // n}%)")
    print(f"Avg latency: {total_ms // n}ms   Avg LLM calls: {total_calls / n:.1f}   "
          f"Avg tokens: {total_tokens // n}   Total tokens: {total_tokens}")
    if mode == "tiered":
        by_tier = ", ".join(f"{k}={v}" for k, v in sorted(tiers.items()))
        print(f"Escalated: {escalated}/{n} ({100 * escalated // n}%)  answered by tier: {by_tier}")


def cmd_feed(args: list[str]) -> None:
    from tabletalk import feed

    if not args:
        print(__doc__)
        return
    sub, rest = args[0], args[1:]
    if sub == "list":
        for entry in feed.list_feed():
            print(f"{entry['id']:<24} [{entry['type']}] {entry['preview']}")
    elif sub == "add-example" and len(rest) == 2:
        print(f"Added {feed.add_example(rest[0], rest[1])}")
    elif sub == "add-note" and rest:
        table = rest[1] if len(rest) > 1 else ""
        print(f"Added {feed.add_note(rest[0], table)}")
    elif sub == "delete" and rest:
        feed.delete_by_id(rest[0])
        print(f"Deleted {rest[0]}")
    else:
        print(__doc__)


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    mode = "agent" if "--agent" in args else ("fixed" if "--fixed" in args else "tiered")
    args = [a for a in args if a not in ("--agent", "--fixed", "--tiered")]
    cmd, rest = args[0], args[1:]
    if cmd == "ingest":
        cmd_ingest()
    elif cmd == "sync":
        cmd_sync()
    elif cmd == "ask" and rest:
        cmd_ask(rest[0], mode=mode)
    elif cmd == "chat":
        cmd_chat(mode=mode)
    elif cmd == "tables":
        cmd_tables()
    elif cmd == "eval":
        cmd_eval(mode=mode)
    elif cmd == "feed":
        cmd_feed(rest)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
