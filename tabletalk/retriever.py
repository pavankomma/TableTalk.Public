"""Retrieval: pull the schema cards, notes, and example queries most
relevant to a question and format them as the context block for the
SQL-writing prompt.

The context always opens with a one-line list of ALL tables so the model
can name a table the retriever missed — the fix loop then surfaces it via
detailed cards on retry. Retrieved tables also pull in their foreign-key
neighbors' cards ("joined via" section), so a question that lands on
order_items automatically sees orders and products — most join mistakes come from a missing neighbor,
not a missing hit."""

from functools import lru_cache

from langchain_core.documents import Document

from .config import (
    EXAMPLES_TOP_K,
    FULL_SCHEMA,
    RERANK_OVERSAMPLE,
    SCHEMA_TOP_K,
    get_vectorstore,
)
from .db import list_tables, table_info
from .reranker import rerank
from .schema_docs import schema_id

MAX_FK_NEIGHBORS = 4  # neighbor cards appended beyond the retrieved ones


def all_tables_line() -> str:
    return "All tables in the database: " + ", ".join(list_tables())


def prefer_human(docs: list) -> list:
    """Stable reorder: human-confirmed examples before auto-captured ones
    (metadata.source). Retrieval decides *which* examples are relevant;
    this decides which the model sees first when both kinds made the cut."""
    return sorted(docs, key=lambda d: (d.metadata or {}).get("source", "human") == "auto")


def search(vs, query: str, k: int, filter: dict) -> list:
    """similarity_search, oversampled, then cross-encoder reranked down to
    k — more precise than raw embedding-similarity order alone."""
    candidates = vs.similarity_search(query, k=k * RERANK_OVERSAMPLE, filter=filter)
    docs = rerank(query, candidates, top_k=k)
    return prefer_human(docs) if filter.get("type") == "example" else docs


@lru_cache(maxsize=1)
def fk_neighbors() -> dict:
    """Undirected FK adjacency: table -> set of tables it joins to.
    Cached per process; a schema change needs a restart (or cache_clear)."""
    neighbors: dict[str, set] = {}
    for table in list_tables():
        for fk in table_info(table, samples=0).foreign_keys:
            neighbors.setdefault(table, set()).add(fk["ref_table"])
            neighbors.setdefault(fk["ref_table"], set()).add(table)
    return neighbors


def retrieve_context(question: str) -> str:
    """Context block for the SQL prompt: table list, relevant schema cards,
    their FK neighbors, business notes, and similar example queries."""
    vs = get_vectorstore()
    tables = list_tables()
    if FULL_SCHEMA:
        # Every card, in a stable order: a fixed prompt prefix that provider
        # prompt caching can reuse across questions, and no "missed table".
        found = vs.get(ids=[schema_id(t) for t in tables])
        schema_docs = [
            Document(page_content=doc, metadata={"type": "schema", "table": t})
            for t, doc in zip(tables, found["documents"] or [])
        ]
    else:
        schema_docs = search(vs, question, k=SCHEMA_TOP_K, filter={"type": "schema"})
    note_docs = search(vs, question, k=SCHEMA_TOP_K, filter={"type": "note"})
    example_docs = search(vs, question, k=EXAMPLES_TOP_K, filter={"type": "example"})

    retrieved = {d.metadata.get("table") for d in schema_docs}
    wanted = []
    for d in schema_docs:
        for neighbor in sorted(fk_neighbors().get(d.metadata.get("table"), ())):
            if neighbor not in retrieved and neighbor not in wanted:
                wanted.append(neighbor)
    neighbor_cards = []
    if wanted:
        found = vs.get(ids=[schema_id(t) for t in wanted[:MAX_FK_NEIGHBORS]])
        neighbor_cards = found["documents"] or []

    parts = [all_tables_line() + "\n"]
    if schema_docs:
        parts.append("## Relevant tables\n")
        for d in schema_docs:
            parts.append(d.page_content + "\n")
    if neighbor_cards:
        parts.append("## Joined via foreign keys\n")
        for card in neighbor_cards:
            parts.append(card + "\n")
    if note_docs:
        parts.append("## Business notes\n")
        for d in note_docs:
            parts.append(f"- {d.page_content}")
        parts.append("")
    if example_docs:
        parts.append("## Similar example queries\n")
        # "Example ..." labels, not a bare "Question:": the real question
        # follows this block as "Question: ...", and reusing that label
        # made the model treat an example as the question it was asked.
        for d in example_docs:
            parts.append(
                f"Example question: {d.page_content}\nExample SQL: {d.metadata['sql']}\n"
            )
    return "\n".join(parts).strip()
