"""Tool functions for the tool-calling agent (tabletalk/agent.py). Plain,
undecorated functions so tests can call them directly, the same way
tests exercise db.py/retriever.py — tabletalk/agent.py wraps each one with
langchain_core.tools.tool() at the point of use.

Thin wrappers around existing db.py/retriever.py/schema_docs.py logic —
no new business logic. Every function is safe to call with a bad
argument: none of them raise on a missing table or a failing SQL
statement, since a raised exception would crash the agent's graph
instead of giving the LLM something to react to.
"""

from typing import Literal

from . import db
from .config import (
    EXAMPLES_TOP_K,
    MAX_DOSSIER_DEPTH,
    SCHEMA_TOP_K,
    get_vectorstore,
)
from .retriever import fk_neighbors
from .retriever import search as retriever_search
from .schema_docs import render_schema_card

def list_tables() -> str:
    """List every table name in the database. Call this first if you're
    unsure what tables exist."""
    return ", ".join(db.list_tables())


def search_schema(
    query: str, kind: Literal["schema", "note", "example"] = "schema"
) -> str:
    """Semantically search the knowledge base for relevant schema cards
    ("schema"), business notes ("note"), or similar past NL->SQL example
    queries ("example"). Use this to find candidate tables/columns for a
    question, or to see how similar questions were answered before."""
    k = EXAMPLES_TOP_K if kind == "example" else SCHEMA_TOP_K
    docs = retriever_search(get_vectorstore(), query, k=k, filter={"type": kind})
    if not docs:
        return f"No matching {kind} entries found."
    if kind == "note":
        return "\n".join(f"- {d.page_content}" for d in docs)
    if kind == "example":
        return "\n\n".join(
            f"Example question: {d.page_content}\nExample SQL: {d.metadata['sql']}"
            for d in docs
        )
    return "\n\n".join(d.page_content for d in docs)


def get_table_detail(table: str) -> str:
    """Get the full schema card for one table: columns, types, primary/
    foreign keys, row count, known lookup values, and sample rows. Call
    this to confirm a table's real columns and join keys before writing
    SQL against it."""
    try:
        return render_schema_card(db.table_info(table))
    except Exception:
        return f"No table named {table!r}. Known tables: {', '.join(db.list_tables())}"


def get_fk_neighbors(table: str) -> str:
    """List the tables a given table can be joined to via foreign keys."""
    if table not in db.list_tables():
        return f"No table named {table!r}. Known tables: {', '.join(db.list_tables())}"
    neighbors = fk_neighbors().get(table)
    if not neighbors:
        return f"{table} has no foreign-key relationships to other tables."
    return f"{table} joins to (via foreign key): " + ", ".join(sorted(neighbors))


def run_sql(sql: str) -> dict:
    """Execute exactly one read-only SQL statement (SELECT or WITH...SELECT)
    against the database and return its result. This is the only way to
    get real data — never claim to have run a query without calling this.
    If it returns an error, revise the SQL based on the message and try
    again; don't repeat an identical failing statement."""
    try:
        result = db.run_sql(sql)
        return {
            "ok": True,
            "sql": db.format_sql(sql),
            "columns": list(result.columns),
            "rows": [list(r) for r in result.rows],
            "text": result.as_text(),
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "sql": sql,
            "columns": [],
            "rows": [],
            "text": "",
            "error": str(exc),
        }


def _render_rows(label: str, columns: list, rows: list) -> str:
    lines = [f"{label}:"]
    for row in rows:
        rendered = ", ".join(f"{c}={v!r}" for c, v in zip(columns, row))
        if len(rendered) > 300:
            rendered = rendered[:300] + "..."
        lines.append(f"  ({rendered})")
    return "\n".join(lines)


def get_record_history(table: str, id_column: str, id_value: str) -> str:
    """Pull a full dossier for one record: its own row(s) plus every row in
    every other table connected to it via foreign key (in either direction),
    following FK-connected identifiers outward up to a few hops (e.g. a customer
    -> their orders -> each order's line items). Use this instead
    of writing your own joins when the question is about one specific
    record/entity by its ID ("what happened to X", "show me everything
    about Y") — it's guaranteed to find every connected table, not just the
    ones you think to join. Does not sort by date; reason about
    recency/order yourself from the returned rows."""
    if table not in db.list_tables():
        return f"No table named {table!r}. Known tables: {', '.join(db.list_tables())}"
    columns = db.column_names(table)
    if id_column not in columns:
        return (
            f"No column {id_column!r} on {table}. "
            f"Known columns: {', '.join(columns)}"
        )

    edges_by_table: dict[str, list] = {}
    for fk in db.all_foreign_keys():
        edges_by_table.setdefault(fk["table"], []).append(
            (fk["column"], fk["ref_table"], fk["ref_column"])
        )
        edges_by_table.setdefault(fk["ref_table"], []).append(
            (fk["ref_column"], fk["table"], fk["column"])
        )

    seed = (table, id_column, id_value)
    visited = set()
    sections = []
    frontier = [seed]
    for depth in range(MAX_DOSSIER_DEPTH + 1):
        next_frontier = []
        for tbl, col, val in frontier:
            key = (tbl, col, val)
            if key in visited:
                continue
            visited.add(key)
            result = db.fetch_by_column(tbl, col, val)
            if not result.rows:
                continue
            label = tbl if (tbl, col, val) == seed else f"{tbl} (via {col} = {val!r})"
            sections.append(_render_rows(label, result.columns, result.rows))
            if depth < MAX_DOSSIER_DEPTH:
                col_index = {name: i for i, name in enumerate(result.columns)}
                for local_col, nb_table, nb_col in edges_by_table.get(tbl, []):
                    if local_col not in col_index:
                        continue
                    for row in result.rows:
                        v = row[col_index[local_col]]
                        if v is not None:
                            next_frontier.append((nb_table, nb_col, v))
        frontier = next_frontier
        if not frontier:
            break

    if not sections:
        return f"No rows found in {table} where {id_column} = {id_value!r}."
    return "\n\n".join(sections)

