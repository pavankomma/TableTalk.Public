"""Read-only access to the target database via SQLAlchemy: guarded query
execution and dialect-agnostic schema introspection.

TABLETALK_DB accepts either a SQLite file path (the default, data/demo.db)
or a full SQLAlchemy URL — postgresql://, mysql+pymysql://, mssql+pyodbc://,
snowflake://, etc. (the matching driver package must be installed).

The read-only guarantee is layered. validate_sql() parses the statement
with sqlglot and rejects anything that is not a single SELECT/UNION (or
WITH...SELECT) — an AST check, so a semicolon inside a string literal is
fine but a second statement is not. On top of that, SQLite files are opened
with mode=ro so even a bug in the guard cannot mutate them; for server
databases, point the URL at a read-only login to keep the second layer.
"""

import re
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import sqlglot
from sqlglot import exp
from sqlalchemy import MetaData, Table, create_engine, func, inspect, select

from .config import DB_SETTING, MAX_DOSSIER_ROWS, MAX_RESULT_ROWS

_COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)

# SQLAlchemy dialect name -> sqlglot dialect name, where they differ.
_SQLGLOT_DIALECTS = {"postgresql": "postgres", "mssql": "tsql"}

# SQLAlchemy dialect name -> name to use in LLM prompts.
_PROMPT_DIALECTS = {
    "sqlite": "SQLite",
    "postgresql": "PostgreSQL",
    "mysql": "MySQL",
    "mssql": "SQL Server (T-SQL)",
    "snowflake": "Snowflake",
    "bigquery": "BigQuery",
    "duckdb": "DuckDB",
    "oracle": "Oracle",
}


@lru_cache(maxsize=1)
def get_engine():
    if "://" in DB_SETTING:
        return create_engine(DB_SETTING, pool_pre_ping=True)
    path = Path(DB_SETTING)
    if not path.exists():
        raise RuntimeError(f"Database not found: {path}")
    uri = f"file:{path.as_posix()}?mode=ro"
    return create_engine("sqlite://", creator=lambda: sqlite3.connect(uri, uri=True))


def dialect_name() -> str:
    """SQLAlchemy dialect name of the target DB, e.g. 'sqlite'."""
    return get_engine().dialect.name


def prompt_dialect() -> str:
    """Human-readable dialect name for LLM prompts, e.g. 'PostgreSQL'."""
    name = dialect_name()
    return _PROMPT_DIALECTS.get(name, name.capitalize())


def _sqlglot_dialect() -> str:
    name = dialect_name()
    return _SQLGLOT_DIALECTS.get(name, name)


def _strip(sql: str) -> str:
    """Remove comments and trailing semicolon for statement inspection."""
    sql = _COMMENT_RE.sub(" ", sql).strip()
    return sql.rstrip(";").strip()


def parse_sql(sql: str) -> tuple[str, exp.Expression]:
    """Return (cleaned text, parsed AST) or raise ValueError if it is not a
    single read-only SELECT (or WITH...SELECT / UNION) statement."""
    cleaned = _strip(sql)
    if not cleaned:
        raise ValueError("Empty SQL statement.")
    try:
        statements = [
            s for s in sqlglot.parse(cleaned, read=_sqlglot_dialect()) if s is not None
        ]
    except sqlglot.errors.ParseError as exc:
        raise ValueError(f"SQL could not be parsed: {exc}") from exc
    if not statements:
        raise ValueError("Empty SQL statement.")
    if len(statements) > 1:
        raise ValueError("Multiple SQL statements are not allowed.")
    root = statements[0]
    if not isinstance(root, (exp.Select, exp.Union)):
        kind = root.key.upper() if root.key else "UNKNOWN"
        raise ValueError(f"Only SELECT queries are allowed, got: {kind}")
    return cleaned, root


def validate_sql(sql: str) -> str:
    """Return the cleaned statement or raise ValueError if it is not a
    single read-only SELECT (or WITH...SELECT / UNION) statement."""
    return parse_sql(sql)[0]


def format_sql(sql: str) -> str:
    """Pretty-print a SQL statement for display — never used for execution.
    LLM-generated SQL sometimes arrives as one long line (most often when
    it's a tool-call argument rather than free structured output); fall
    back to the original text if it doesn't parse for any reason, since
    this is a cosmetic step only."""
    try:
        return sqlglot.parse_one(sql, read=_sqlglot_dialect()).sql(
            dialect=_sqlglot_dialect(), pretty=True
        )
    except Exception:
        return sql


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    truncated: bool

    def as_text(self) -> str:
        if not self.rows:
            return "(no rows)"
        header = " | ".join(self.columns)
        lines = [header, "-" * len(header)]
        lines += [" | ".join(str(v) for v in row) for row in self.rows]
        if self.truncated:
            lines.append(f"... (truncated at {len(self.rows)} rows)")
        return "\n".join(lines)


def run_sql(sql: str, max_rows: int = MAX_RESULT_ROWS) -> QueryResult:
    """Execute one read-only statement (validated first; see validate_sql)."""
    cleaned = validate_sql(sql)
    with get_engine().connect() as conn:
        # exec_driver_sql: no bindparam parsing, so ':' inside literals is safe
        cursor = conn.exec_driver_sql(cleaned)
        columns = list(cursor.keys())
        rows = [tuple(r) for r in cursor.fetchmany(max_rows + 1)]
    truncated = len(rows) > max_rows
    return QueryResult(columns=columns, rows=rows[:max_rows], truncated=truncated)


# --- schema introspection -------------------------------------------------


# FTS5 (and other virtual-table modules) register these as ordinary tables,
# but they're internal storage, not queryable schema — hide them the same
# way sqlite hides sqlite_master's own bookkeeping tables.
_FTS5_SHADOW_SUFFIXES = ("_data", "_idx", "_docsize", "_config", "_content")


def _is_fts5_shadow(name: str, all_names: set[str]) -> bool:
    for suffix in _FTS5_SHADOW_SUFFIXES:
        if name.endswith(suffix) and name[: -len(suffix)] in all_names:
            return True
    return False


def list_tables() -> list[str]:
    names = set(inspect(get_engine()).get_table_names())
    return sorted(n for n in names if not _is_fts5_shadow(n, names))


def column_names(table: str) -> list[str]:
    """Cheap column-name lookup (DDL introspection only, no data scan) —
    unlike table_info(), which also computes categorical value counts."""
    return [c["name"] for c in inspect(get_engine()).get_columns(table)]


@dataclass
class TableInfo:
    name: str
    columns: list[dict]        # {name, type, pk, notnull}
    foreign_keys: list[dict]   # {column, ref_table, ref_column}
    row_count: int
    sample_rows: list[tuple]
    categorical: list[dict]    # {column, distinct_count, values: [(value, count), ...]}


# Columns with at most this many distinct values get their full value/count
# breakdown embedded in the schema card, so the agent knows real lookup
# values (status strings, categories, regions) instead of
# guessing from a handful of sample rows.
MAX_CATEGORICAL_DISTINCT = 60


def _categorical_columns(engine, table: Table, pk_cols: set[str]) -> list[dict]:
    results = []
    with engine.connect() as conn:
        for col in table.columns:
            if col.name in pk_cols:
                continue
            distinct_count = conn.execute(
                select(func.count(func.distinct(col)))
            ).scalar()
            if not distinct_count or distinct_count > MAX_CATEGORICAL_DISTINCT:
                continue
            rows = conn.execute(
                select(col, func.count().label("n"))
                .group_by(col)
                .order_by(func.count().desc())
            ).fetchall()
            results.append(
                {
                    "column": col.name,
                    "distinct_count": distinct_count,
                    "values": [(r[0], r[1]) for r in rows],
                }
            )
    return results


def table_info(name: str, samples: int = 3) -> TableInfo:
    engine = get_engine()
    insp = inspect(engine)
    pk_cols = set(insp.get_pk_constraint(name).get("constrained_columns") or [])
    cols = [
        {
            "name": c["name"],
            "type": str(c["type"]),
            "notnull": not c.get("nullable", True),
            "pk": c["name"] in pk_cols,
        }
        for c in insp.get_columns(name)
    ]
    fks = []
    for fk in insp.get_foreign_keys(name):
        for col, ref in zip(fk["constrained_columns"], fk["referred_columns"]):
            fks.append(
                {"column": col, "ref_table": fk["referred_table"], "ref_column": ref}
            )
    table = Table(name, MetaData(), autoload_with=engine)
    with engine.connect() as conn:
        count = conn.execute(select(func.count()).select_from(table)).scalar() or 0
        rows = (
            [tuple(r) for r in conn.execute(select(table).limit(samples))]
            if samples
            else []
        )
    categorical = _categorical_columns(engine, table, pk_cols)
    return TableInfo(
        name=name,
        columns=cols,
        foreign_keys=fks,
        row_count=count,
        sample_rows=rows,
        categorical=categorical,
    )


@lru_cache(maxsize=1)
def all_foreign_keys() -> list[dict]:
    """Every declared foreign key across every table, flattened to
    {table, column, ref_table, ref_column}. Introspection only (no data
    scan) — deliberately doesn't go through table_info(), which also
    computes categorical value counts and is much more expensive. Cached
    per process; a schema change needs a restart (or cache_clear())."""
    engine = get_engine()
    insp = inspect(engine)
    keys = []
    for table in list_tables():
        for fk in insp.get_foreign_keys(table):
            for col, ref in zip(fk["constrained_columns"], fk["referred_columns"]):
                keys.append(
                    {
                        "table": table,
                        "column": col,
                        "ref_table": fk["referred_table"],
                        "ref_column": ref,
                    }
                )
    return keys


def fetch_by_column(
    table: str,
    column: str,
    value,
    limit: int = MAX_DOSSIER_ROWS,
) -> QueryResult:
    """Fetch up to `limit` rows from `table` where `column` == `value`,
    safely parameterized via SQLAlchemy Core. Never touches raw SQL text,
    so it needs no validate_sql pass — but it also does no existence
    checking, so callers must validate `table`/`column` against real
    introspected names first (tools.get_record_history does this)."""
    engine = get_engine()
    tbl = Table(table, MetaData(), autoload_with=engine)
    col = tbl.c[column]
    stmt = select(tbl).where(col == value)
    with engine.connect() as conn:
        cursor = conn.execute(stmt.limit(limit))
        columns = list(cursor.keys())
        rows = [tuple(r) for r in cursor.fetchall()]
    return QueryResult(columns=columns, rows=rows, truncated=len(rows) >= limit)
