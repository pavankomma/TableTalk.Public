"""Live management of the vector-store feed.

The feed has three kinds of entries, all with stable deterministic IDs:
  schema:<table>   - introspected schema card (synced from the live DB)
  example:<hash>   - NL->SQL example pair (embedded by question)
  note:<hash>      - free-text business note, optionally tied to a table

Examples and notes are durable: they live in examples/examples.json and
examples/notes.json, and every add/delete updates both the JSON file and
Chroma, so a full rebuild never loses runtime additions.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from .config import EXAMPLES_FILE, get_vectorstore
from .schema_docs import (
    build_example_doc,
    build_note_doc,
    build_schema_docs,
    example_id,
    note_id,
)

NOTES_FILE = EXAMPLES_FILE.parent / "notes.json"


def _load_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


# --- schema cards ---------------------------------------------------------


def sync_schema() -> dict:
    """Upsert one schema card per live table; drop cards for removed tables."""
    vs = get_vectorstore()
    ids, docs = build_schema_docs()
    existing = vs.get(where={"type": "schema"})
    stale = [i for i in existing["ids"] if i not in set(ids)]
    if stale:
        vs.delete(ids=stale)
    if docs:
        vs.add_documents(docs, ids=ids)
    return {"synced": len(docs), "removed": len(stale)}


# --- examples -------------------------------------------------------------


def add_example(
    question: str, sql: str, source: str = "human", user_id: str = ""
) -> str:
    """Add (or replace, by question) an NL->SQL example in both stores.
    `source` is "human" or "auto" (see schema_docs.build_example_doc); an
    auto capture never overwrites a human-confirmed entry for the same
    question."""
    items = _load_json(EXAMPLES_FILE)
    existing = next((i for i in items if i["question"] == question), None)
    if existing and source == "auto" and existing.get("source", "human") == "human":
        return example_id(question)
    items = [i for i in items if i["question"] != question]
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry = {"question": question, "sql": sql, "source": source or "human", "ts": ts}
    if user_id:
        entry["user_id"] = user_id
    items.append(entry)
    _save_json(EXAMPLES_FILE, items)
    doc_id, doc = build_example_doc(question, sql, source, user_id, ts)
    get_vectorstore().add_documents([doc], ids=[doc_id])
    return doc_id


def list_examples(source: str | None = None) -> list[dict]:
    items = _load_json(EXAMPLES_FILE)
    if source:
        items = [i for i in items if i.get("source", "human") == source]
    return items


def delete_example(question: str) -> bool:
    items = _load_json(EXAMPLES_FILE)
    kept = [i for i in items if i["question"] != question]
    _save_json(EXAMPLES_FILE, kept)
    get_vectorstore().delete(ids=[example_id(question)])
    return len(kept) < len(items)


# --- notes ----------------------------------------------------------------


def add_note(text: str, table: str = "") -> str:
    items = _load_json(NOTES_FILE)
    items = [i for i in items if i["text"] != text]
    items.append({"text": text, "table": table})
    _save_json(NOTES_FILE, items)
    doc_id, doc = build_note_doc(text, table)
    get_vectorstore().add_documents([doc], ids=[doc_id])
    return doc_id


def delete_note(text: str) -> bool:
    items = _load_json(NOTES_FILE)
    kept = [i for i in items if i["text"] != text]
    _save_json(NOTES_FILE, kept)
    get_vectorstore().delete(ids=[note_id(text)])
    return len(kept) < len(items)


# --- generic --------------------------------------------------------------


def delete_by_id(doc_id: str) -> None:
    """Delete any feed entry by its ID, keeping the JSON stores in sync."""
    vs = get_vectorstore()
    entry = vs.get(ids=[doc_id])
    if not entry["ids"]:
        raise KeyError(f"No feed entry with id {doc_id}")
    if doc_id.startswith("example:"):
        # page_content of an example IS the question
        delete_example(entry["documents"][0])
    elif doc_id.startswith("note:"):
        notes = _load_json(NOTES_FILE)
        kept = [n for n in notes if note_id(n["text"]) != doc_id]
        _save_json(NOTES_FILE, kept)
        vs.delete(ids=[doc_id])
    else:
        vs.delete(ids=[doc_id])


def export_feed() -> list[dict]:
    """Full feed contents for UIs: id, type, content, and extra metadata
    (an example's SQL, a note's table)."""
    data = get_vectorstore().get()
    entries = []
    for doc_id, content, meta in zip(
        data["ids"], data["documents"], data["metadatas"]
    ):
        meta = meta or {}
        entries.append(
            {
                "id": doc_id,
                "type": meta.get("type", "?"),
                "content": content or "",
                "metadata": {k: v for k, v in meta.items() if k != "type"},
            }
        )
    return sorted(entries, key=lambda e: (e["type"], e["id"]))


def list_feed() -> list[dict]:
    data = get_vectorstore().get()
    entries = []
    for doc_id, content, meta in zip(
        data["ids"], data["documents"], data["metadatas"]
    ):
        preview = content.splitlines()[0][:80] if content else ""
        entries.append({"id": doc_id, "type": meta.get("type", "?"), "preview": preview})
    return sorted(entries, key=lambda e: (e["type"], e["id"]))


def rebuild() -> dict:
    """Full rebuild: wipe the collection, then re-add schema cards plus all
    durable examples and notes."""
    vs = get_vectorstore()
    existing = vs.get()
    if existing["ids"]:
        vs.delete(ids=existing["ids"])

    ids, docs = build_schema_docs()
    for item in _load_json(EXAMPLES_FILE):
        doc_id, doc = build_example_doc(
            item["question"], item["sql"], item.get("source", "human"),
            item.get("user_id", ""), item.get("ts", ""),
        )
        ids.append(doc_id)
        docs.append(doc)
    for item in _load_json(NOTES_FILE):
        doc_id, doc = build_note_doc(item["text"], item.get("table", ""))
        ids.append(doc_id)
        docs.append(doc)
    if docs:
        vs.add_documents(docs, ids=ids)
    n_schema = sum(1 for i in ids if i.startswith("schema:"))
    n_examples = sum(1 for i in ids if i.startswith("example:"))
    n_notes = sum(1 for i in ids if i.startswith("note:"))
    return {"schema": n_schema, "examples": n_examples, "notes": n_notes}
