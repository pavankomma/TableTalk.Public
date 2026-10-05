<div align="center">

<img src="public/logo_light.svg" alt="TableTalk" height="56">

# TableTalk (SQL RAG)

**Ask your database questions in plain English. Get answers grounded in real rows, with the SQL that produced them.**

Retrieval-augmented text-to-SQL on LangChain + LangGraph, with an agent that escalates only when needed.

![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-agentic-7c3aed)
![RAG](https://img.shields.io/badge/RAG-Chroma%20%2B%20rerank-2563eb)
![Read-only](https://img.shields.io/badge/SQL-read--only-0f766e)
![Tests](https://img.shields.io/badge/tests-135%20passing-16a34a)

<img src="docs/images/chat-answer.png" alt="TableTalk answering a revenue question with the generated SQL shown above the answer" width="820">

</div>

---

## What it does

- **Answers questions over any SQL database.** Each answer shows the SQL it ran, so you can check it.
- **Retrieves before it writes.** A vector feed of schema cards, business rules and worked examples tells the model what the tables *mean*, not just what they're called.
- **Escalates when it's unsure.** A cheap two-call pipeline answers most questions. If the result looks wrong, TableTalk retries with more reasoning, then hands the question to a tool-calling agent.
- **Is read-only by construction.** It runs an AST-validated `SELECT` over a read-only connection, so it can't modify your data.
- **Learns from feedback.** One click saves a confirmed answer as an example that future questions retrieve.

## How it works

<img src="docs/images/architecture.svg" alt="Architecture: question, input guardrail, three escalation rungs, RAG feed, read-only layer" width="100%">

**Rung 1: fixed RAG pipeline.** `retrieve → write_sql → execute → answer`, with up to two automatic fixes when the SQL errors. It takes two LLM calls at minimal reasoning effort.

**Rung 2: informed retry.** If rung 1 errored, returned zero rows, or failed the optional self-check, the same pipeline runs again with the failed SQL and the reason it failed in the prompt, at high effort.

**Rung 3: tool-calling agent.** If the retry is still unreliable, a LangGraph agent explores the schema with tools (`list_tables`, `search_schema`, `get_table_detail`, `get_fk_neighbors`, `get_record_history`, `run_sql`) instead of guessing joins. `state["tier"]` records which rung answered.

<div align="center">
<img src="docs/images/chat-agent.png" alt="The agent answering 'show me everything about customer 25' by calling list_tables, get_table_detail and get_record_history" width="820">
<br><sub>Agent mode: the agent finds the customer's tables itself and walks foreign keys for a complete history.</sub>
</div>

### The RAG feed

The vector store (Chroma) holds three kinds of documents, each with a stable ID:

| Document | What it holds | Where it comes from |
|---|---|---|
| `schema:<table>` | columns, keys, row count, real lookup values, sample rows | introspected from the live DB (`main.py sync`) |
| `example:<hash>` | a question → SQL pair, embedded by question | `examples/examples.json`, plus 👍 in the UI |
| `note:<hash>` | a business rule, e.g. *"net revenue excludes cancelled and returned orders"* | `examples/notes.json` |

For each question, retrieval oversamples by similarity, reranks with a local cross-encoder (ONNX, no API call), adds the cards of foreign-key neighbors so joins aren't missed, and assembles a single context block. Every feed edit is written to both the JSON files and Chroma, so a full rebuild never loses anything.

<div align="center">
<img src="docs/images/feed-editor.png" alt="The feed editor listing examples with their SQL, and edit/delete actions" width="820">
<br><sub>The feed editor (<code>/feed</code>): browse, edit and re-sync everything the model retrieves.</sub>
</div>

## Quickstart

```bash
git clone https://github.com/pavankomma/TableTalk.Public.git && cd TableTalk.Public
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/build_demo_db.py      # builds data/demo.db (about 1 MB, deterministic)
cp .env.example .env                 # then add your LLM + embedding credentials
python main.py ingest                # embeds schema cards, examples and notes
python main.py ask "Which 5 products sold the most units last quarter?"
```

The demo database is a synthetic outdoor-gear retailer: 8 tables of regions, sales reps, customers, categories, products, orders, order line items and support tickets. Its schema includes the things that trip up text-to-SQL in practice: multi-hop joins, revenue that exists only at line-item level, nullable foreign keys, and status codes.

## Ways to use it

| Surface | Command | Notes |
|---|---|---|
| **Web chat** | `python -m chainlit run app.py` | Streams progress, SQL, results; 👍 *Save as example*; an "Agent only" switch in settings |
| **REST API** | `python -m uvicorn api:app --port 8001` | `POST /ask` (JSON) and `POST /ask/stream` (server-sent events) |
| **Feed editor** | open `http://localhost:8001/feed` | Vue app served by the API |
| **CLI** | `python main.py ask "…"` / `chat` | Add `--fixed` or `--agent` to force a mode |

```bash
curl -s localhost:8001/ask -H 'content-type: application/json' \
     -d '{"question": "How many support tickets are currently open?"}'
```

```json
{ "answer": "There are 16 open support tickets…", "sql": "SELECT COUNT(*) …",
  "columns": ["open_tickets"], "rows": [[16]], "tier": "fixed", "escalated": false }
```

## Evaluation

`golden/golden.json` holds 31 questions with reference SQL, plus 3 that must be refused (off-topic, prompt injection). `main.py eval` compares the result **rows** (execution accuracy, not string matching) and reports latency and token cost per question:

```text
$ python main.py eval
pass  What was net revenue per region in 2025, highest first?  [5571ms, 2 llm, 5371 tok]
pass  How many orders has customer 25 placed?  [4571ms, 2 llm, 4773 tok]
...
Execution accuracy: 34/34 (100%)
Avg latency: 4909ms   Avg LLM calls: 1.9   Avg tokens: 4674   Total tokens: 158938
Escalated: 0/34 (0%)  answered by tier: fixed=34
```

That run used `gpt-5.4` with the demo database. Treat it as a smoke test, not a benchmark: the set is small, written for this dataset, and three ambiguous questions were reworded after an earlier run. On the demo data every question was answered by the cheap first rung; the retry and agent rungs are for questions where that rung guesses wrong, such as unfamiliar schemas or record-level digging (see the agent screenshot above).

Use it to compare prompt, model, or retrieval changes. Pass `--fixed` or `--agent` to evaluate a single rung.

## Use your own database

Point `TABLETALK_DB` at a SQLite file or any SQLAlchemy URL, then sync and add a few examples:

```bash
TABLETALK_DB=postgresql+psycopg://readonly:pass@host/db   # in .env; use a read-only login
python main.py ingest
python main.py feed add-note "Active customers have status = 'A'" customers
python main.py feed add-example "How many active customers?" "SELECT COUNT(*) FROM customers WHERE status = 'A'"
```

Prompts and SQL validation adapt to the dialect (SQLite, PostgreSQL, MySQL, SQL Server, Snowflake, and others).

## Configuration

All settings are optional except the LLM credentials. See [`.env.example`](.env.example).

| Variable | Default | Purpose |
|---|---|---|
| `AZURE_OPENAI_*` | — | Default provider: chat + embeddings via an Azure OpenAI v1 endpoint |
| `TABLETALK_LLM_PROVIDER` / `_LLM_MODEL` | `azure` | Any LangChain provider: `openai`, `anthropic`, `ollama`, … |
| `TABLETALK_DB` | `data/demo.db` | SQLite path or SQLAlchemy URL |
| `TABLETALK_EFFORT_FAST` / `_RETRY` / `_AGENT` | `minimal` / `high` / `medium` | Reasoning effort per rung |
| `TABLETALK_SELF_CHECK` | `0` | Adds a cheap "does this SQL answer the question?" check that can trigger escalation |
| `TABLETALK_FULL_SCHEMA` | `0` | Inline every schema card (small DBs; prompt-cache friendly) |
| `TABLETALK_AUTO_LEARN` | `0` | Save reliable first-try answers as `auto` examples |

## Safety

- **Read-only, enforced twice:** [sqlglot](https://github.com/tobymao/sqlglot) parses every statement and rejects anything but a single `SELECT`/`UNION`/CTE, and SQLite files are opened with `mode=ro` (for server databases, use a read-only login). Results are capped at 50 rows.
- **Guardrails without an LLM:** regex input checks refuse prompt-injection attempts before retrieval, and output checks replace answers that come from the model's general knowledge instead of the query results. Off-topic questions are refused by the system prompt.

## Project layout

```text
tabletalk/
  graph.py        rung 1/2: retrieve → write_sql → execute → [verify] → answer
  agent.py        rung 3: tool-calling agent (seed → agent ⇄ tools → finalize)
  tiered_ask.py   escalation logic + streaming events
  tools.py        the agent's tools
  retriever.py    similarity search → rerank → FK neighbors → context
  feed.py         the only writer to the vector store
  db.py           SQLAlchemy access, SQL validation, introspection
  guardrail.py    input/output checks
app.py  api.py  main.py           Chainlit UI, REST API, CLI
scripts/build_demo_db.py          deterministic demo database
examples/  golden/                feed seed data, evaluation set
```

## Tests

```bash
python -m pytest tests/ -q
```

The tests make no LLM calls and need no credentials. They run against the demo database (built automatically if missing) and derive their expectations from it.

## License

[MIT](LICENSE)
