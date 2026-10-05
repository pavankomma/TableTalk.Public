"""Builds the documents that feed the vector store: one 'schema card' per
table (introspected live from the database) plus example NL->SQL pairs and
free-text business notes.

Every document gets a stable, deterministic ID so the feed can be
incrementally added to, synced, and deleted from — see feed.py.
"""

import hashlib

from langchain_core.documents import Document

from .db import TableInfo, list_tables, table_info


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def schema_id(table: str) -> str:
    return f"schema:{table}"


def example_id(question: str) -> str:
    return f"example:{_short_hash(question)}"


def note_id(text: str) -> str:
    return f"note:{_short_hash(text)}"


def render_schema_card(info: TableInfo) -> str:
    lines = [f"Table: {info.name}", "Columns:"]
    for c in info.columns:
        flags = []
        if c["pk"]:
            flags.append("PRIMARY KEY")
        if c["notnull"]:
            flags.append("NOT NULL")
        suffix = (" " + " ".join(flags)) if flags else ""
        lines.append(f"  {c['name']} {c['type']}{suffix}")
    if info.foreign_keys:
        lines.append("Foreign keys:")
        for fk in info.foreign_keys:
            lines.append(
                f"  {fk['column']} -> {fk['ref_table']}.{fk['ref_column']}"
            )
    lines.append(f"Row count: {info.row_count}")
    if info.categorical:
        lines.append("Lookup values (use these exact strings in WHERE clauses):")
        for cat in info.categorical:
            vals = ", ".join(f"{v!r} ({n})" for v, n in cat["values"])
            lines.append(f"  {cat['column']} ({cat['distinct_count']} distinct): {vals}")
    if info.sample_rows:
        lines.append("Sample rows:")
        for row in info.sample_rows:
            rendered = ", ".join(repr(v) for v in row)
            if len(rendered) > 200:
                rendered = rendered[:200] + "..."
            lines.append(f"  ({rendered})")
    return "\n".join(lines)


def build_schema_docs() -> tuple[list[str], list[Document]]:
    """Introspect the live database and return (ids, docs), one per table."""
    ids, docs = [], []
    for table in list_tables():
        card = render_schema_card(table_info(table))
        ids.append(schema_id(table))
        docs.append(
            Document(page_content=card, metadata={"type": "schema", "table": table})
        )
    return ids, docs


def build_example_doc(
    question: str, sql: str, source: str = "human", user_id: str = "", ts: str = ""
) -> tuple[str, Document]:
    """An example is embedded by its question; the SQL travels in metadata,
    as does its provenance: `source` is "human" (confirmed via Save-as-
    example / the feed editor / the CLI) or "auto" (captured by the
    learning loop from a run that looked reliable). Retrieval ranks human
    examples above auto ones."""
    meta = {"type": "example", "sql": sql, "source": source or "human"}
    if user_id:
        meta["user_id"] = user_id
    if ts:
        meta["ts"] = ts
    return example_id(question), Document(page_content=question, metadata=meta)


def build_note_doc(text: str, table: str = "") -> tuple[str, Document]:
    content = f"[{table}] {text}" if table else text
    return note_id(text), Document(
        page_content=content, metadata={"type": "note", "table": table}
    )
