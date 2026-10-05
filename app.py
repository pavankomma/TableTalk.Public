"""TableTalk web UI — Chainlit frontend over the tiered text-to-SQL graphs.

Run:
    python -m chainlit run app.py -w

Each chat session keeps a rolling history (the same "Q: ... / A: ..." format
the CLI chat uses). Every turn streams progress, shows the generated SQL in
a collapsible step, the result rows as a table, and a "Save as example"
action — clicking it writes the confirmed NL->SQL pair into the
live vector feed, so the agent learns from confirmed answers.
"""

import chainlit as cl
from chainlit.input_widget import Switch

TABLE_DISPLAY_ROWS = 20  # rows shown in the results step
HISTORY_TURNS = 3        # prior Q/A turns replayed as conversation context


@cl.set_starters
async def starters() -> list[cl.Starter]:
    return [
        cl.Starter(
            label="Revenue by region",
            message="What was total revenue per region in 2025, highest first?",
        ),
        cl.Starter(
            label="Best-selling products",
            message="Which 5 products sold the most units last quarter?",
        ),
        cl.Starter(
            label="Customer dossier",
            message="Show me everything that happened with customer 25.",
        ),
        cl.Starter(
            label="Explore the schema",
            message="What tables exist in this database and how are they related?",
        ),
    ]


@cl.on_chat_start
async def on_chat_start() -> None:
    # graphs are built lazily and shared by tiered_ask (module-level cache)
    cl.user_session.set("history", [])
    cl.user_session.set("use_agent", False)
    await cl.ChatSettings(
        [
            Switch(
                id="use_agent",
                label="Agent only",
                initial=False,
                description="Skip the tiered pipeline and go straight to the "
                "tool-calling agent, which explores the schema and decides its "
                "own steps.",
            )
        ]
    ).send()


@cl.on_settings_update
async def on_settings_update(settings: dict) -> None:
    cl.user_session.set("use_agent", settings["use_agent"])


def _markdown_table(columns: list, rows: list) -> str:
    def cell(v) -> str:
        return str(v).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cell(c) for c in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows[:TABLE_DISPLAY_ROWS]:
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")
    if len(rows) > TABLE_DISPLAY_ROWS:
        lines.append(
            f"\n_{len(rows) - TABLE_DISPLAY_ROWS} more rows — run the SQL above "
            "for the full set._"
        )
    return "\n".join(lines)


@cl.action_callback("save_example")
async def save_example(action: cl.Action) -> None:
    from tabletalk.feed import add_example

    doc_id = await cl.make_async(add_example)(
        action.payload["question"], action.payload["sql"]
    )
    await action.remove()
    await cl.Message(
        content=f"Saved to the feed as `{doc_id}` — future questions like this "
        "will retrieve it as an example."
    ).send()


_STATUS_LABELS = {
    "retrieve": "Retrieving schema context",
    "write_sql": "Writing SQL",
    "execute": "Running SQL",
    "answer": "Composing answer",
    "retry": "Result looked unreliable — retrying with more reasoning",
    "escalate": "Still unreliable — escalating to the tool-calling agent",
    "verify": "Checking the SQL against the question",
    "seed": "Starting agent",
    "agent": "Agent thinking",
    "tools": "Running tools",
    "finalize": "Composing answer",
}


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """One turn, streamed: a progress step that updates per graph node,
    the SQL as soon as it's written, the result table, and answer tokens
    as they arrive."""
    from tabletalk.tiered_ask import stream as tiered_stream

    use_agent = cl.user_session.get("use_agent", False)
    history: list[str] = cl.user_session.get("history") or []

    answer_msg = cl.Message(content="")
    sql_text = ""
    tools_used: list[str] = []
    final: dict | None = None
    columns: list = []
    rows: list = []

    async with cl.Step(name="Generate and run SQL", type="tool") as step:
        try:
            async for ev in tiered_stream(
                message.content,
                "\n".join(history[-HISTORY_TURNS:]),
                mode="agent" if use_agent else "tiered",
            ):
                kind = ev["type"]
                if kind == "status":
                    step.output = _STATUS_LABELS.get(ev["node"], ev["node"]) + "…"
                    await step.update()
                elif kind == "tool_call":
                    args = ", ".join(f"{k}={v!r}" for k, v in ev["args"].items())
                    tools_used.append(f"{ev['name']}({args})")
                    step.output = f"Calling `{ev['name']}`…"
                    await step.update()
                elif kind == "sql":
                    sql_text = ev["sql"]
                    step.output = f"```sql\n{sql_text}\n```"
                    await step.update()
                elif kind == "table":
                    columns, rows = ev["columns"], ev["rows"]
                elif kind == "token":
                    await answer_msg.stream_token(ev["text"])
                elif kind == "final":
                    final = ev
        except Exception as exc:  # noqa: BLE001 — surface, don't crash the session
            step.output = f"Pipeline failed: {exc}"
            await cl.Message(content=f"Something went wrong while answering: {exc}").send()
            return

        if sql_text:
            step.output = f"```sql\n{sql_text}\n```"
        elif tools_used:
            steps = "\n".join(f"- `{call}`" for call in tools_used)
            step.output = f"No SQL was needed for this answer. Tools used:\n{steps}"
        else:
            step.output = "(no SQL generated)"
        if final and final.get("escalated"):
            tier = {"retry": "a second, higher-effort attempt", "agent": "the tool-calling agent"}
            step.output = f"_Answered by {tier.get(final.get('tier'), final.get('tier'))}._\n\n" + step.output
        if final and final.get("error"):
            step.output += f"\n\n**SQL error (after fix attempts):** {final['error']}"

    final = final or {"answer": answer_msg.content, "error": None}
    actions = []
    if columns and rows and not final.get("error"):
        # plain step name: Chainlit builds an avatar URL from it, and
        # non-alphanumeric characters 400 against /avatars/<name>
        async with cl.Step(name="Results", type="tool") as results:
            results.output = f"**{len(rows)} rows**\n\n" + _markdown_table(columns, rows)
        if sql_text:
            actions.append(
                cl.Action(
                    name="save_example",
                    label="👍 Save as example",
                    tooltip="Store this question + SQL in the feed so the agent "
                    "learns from it",
                    payload={"question": message.content, "sql": sql_text},
                )
            )

    answer = final.get("answer") or answer_msg.content
    if final.get("confidence") == "low" and final.get("caveats"):
        # the self-check (TABLETALK_SELF_CHECK=1) flagged the SQL; say so plainly
        answer += "\n\n> ⚠️ **Low confidence** — " + "; ".join(final["caveats"])
    if answer_msg.content and answer_msg.content == answer:
        answer_msg.actions = actions
        await answer_msg.send()
    else:
        # nothing streamed (refusal, error) or the final text differs — send it whole
        await cl.Message(content=answer, actions=actions).send()
    history.append(f"Q: {message.content}\nA: {final.get('answer') or answer}")
    cl.user_session.set("history", history)
