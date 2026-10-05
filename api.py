"""TableTalk REST API — the tiered text-to-SQL engine behind a JSON endpoint,
so bots, BI tools, and other services can ask questions programmatically.
Also serves the Vue-based feed editor at /feed.

Run:
    python -m uvicorn api:app --port 8001

    POST /ask {"question": "...", "history": ""}
        -> {"answer", "sql", "error", "columns", "rows", "escalated", "tier", ...}
    POST /ask/stream  (same body) -> text/event-stream of status/sql/table/
        token/tool_call events, ending with `final` (see tiered_ask.stream)
    GET /health
    GET /feed                       # feed editor UI (Vue)
    GET/POST/DELETE /api/feed...    # feed CRUD + sync/rebuild backing the UI

The API has no authentication: run it on a trusted network, or put it
behind your own auth proxy.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tabletalk.tiered_ask import ask as tiered_ask

ADMIN_DIR = Path(__file__).parent / "admin"

app = FastAPI(title="TableTalk API", version="1.0")
app.mount("/admin", StaticFiles(directory=str(ADMIN_DIR)), name="admin")


# --- ask ------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str
    history: str = ""  # prior turns as "Q: ... / A: ..." text


class AskResponse(BaseModel):
    answer: str
    sql: str | None = None
    error: str | None = None
    columns: list[str] = []
    rows: list[list] = []
    escalated: bool = False
    tier: str | None = None
    confidence: str | None = None
    caveats: list[str] = []


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    state = tiered_ask(req.question, req.history)
    return AskResponse(
        answer=state.get("answer", ""),
        sql=state.get("sql"),
        error=state.get("error"),
        columns=state.get("result_columns") or [],
        rows=state.get("result_rows") or [],
        escalated=bool(state.get("escalated")),
        tier=state.get("tier"),
        confidence=state.get("confidence"),
        caveats=state.get("caveats") or [],
    )


@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    """Server-sent events: one `event: <type>` per tiered_ask.stream()
    event (status, sql, table, token, tool_call, final), each with a JSON
    `data` payload. A failure mid-stream arrives as `event: error`."""
    import json

    from sse_starlette.sse import EventSourceResponse

    from tabletalk.tiered_ask import stream as tiered_stream

    async def events():
        try:
            async for ev in tiered_stream(req.question, req.history):
                yield {"event": ev["type"], "data": json.dumps(ev, default=str)}
        except Exception as exc:  # noqa: BLE001 — report, don't drop the connection silently
            yield {"event": "error", "data": json.dumps({"detail": f"{type(exc).__name__}: {exc}"})}

    return EventSourceResponse(events())


# --- feed editor ----------------------------------------------------------


class ExampleUpsert(BaseModel):
    question: str
    sql: str
    source: str = "human"          # "human" (default) or "auto"; Approve re-posts as human
    replace_id: str | None = None  # set when editing an existing entry


class NoteUpsert(BaseModel):
    text: str
    table: str = ""
    replace_id: str | None = None


@app.get("/feed", include_in_schema=False)
def feed_page() -> FileResponse:
    return FileResponse(ADMIN_DIR / "feed.html")


@app.get("/api/feed")
def feed_data() -> dict:
    from tabletalk.feed import export_feed

    return {"entries": export_feed()}


def _replace(replace_id: str | None) -> None:
    if replace_id:
        from tabletalk.feed import delete_by_id

        try:
            delete_by_id(replace_id)
        except KeyError:
            pass  # already gone; the add below still applies


@app.post("/api/feed/example")
def upsert_example(req: ExampleUpsert) -> dict:
    from tabletalk.feed import add_example

    if not req.question.strip() or not req.sql.strip():
        raise HTTPException(422, "Question and SQL are both required.")
    _replace(req.replace_id)
    source = "auto" if req.source == "auto" else "human"
    return {"id": add_example(req.question.strip(), req.sql.strip(), source=source)}


@app.post("/api/feed/note")
def upsert_note(req: NoteUpsert) -> dict:
    from tabletalk.feed import add_note

    if not req.text.strip():
        raise HTTPException(422, "Note text is required.")
    _replace(req.replace_id)
    return {"id": add_note(req.text.strip(), req.table.strip())}


@app.delete("/api/feed/{doc_id}")
def delete_entry(doc_id: str) -> dict:
    from tabletalk.feed import delete_by_id

    try:
        delete_by_id(doc_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"deleted": doc_id}


@app.post("/api/feed/sync")
def sync() -> dict:
    from tabletalk.feed import sync_schema

    return sync_schema()


@app.post("/api/feed/rebuild")
def rebuild() -> dict:
    from tabletalk.feed import rebuild as rebuild_feed

    return rebuild_feed()
