"""No-LLM tests: SQL guards, introspection, schema cards, and the seed
examples file. Expectations are derived from the live database at runtime
rather than hard-coded."""

import json

import pytest

from tabletalk.config import EXAMPLES_FILE
from tabletalk.db import list_tables, run_sql, table_info, validate_sql
from tabletalk.retriever import all_tables_line
from tabletalk.schema_docs import (
    build_schema_docs,
    example_id,
    note_id,
    render_schema_card,
    schema_id,
)


class TestValidateSql:
    def test_select_ok(self):
        assert validate_sql("SELECT 1") == "SELECT 1"

    def test_cte_ok(self):
        assert validate_sql("WITH x AS (SELECT 1) SELECT * FROM x").startswith("WITH")

    def test_trailing_semicolon_ok(self):
        assert validate_sql("SELECT 1;") == "SELECT 1"

    def test_comments_stripped(self):
        assert validate_sql("-- comment\nSELECT 1 /* inline */") == "SELECT 1"

    @pytest.mark.parametrize(
        "sql",
        [
            "INSERT INTO orders VALUES (1)",
            "UPDATE orders SET status = 'x'",
            "DELETE FROM orders",
            "DROP TABLE orders",
            "PRAGMA table_info(orders)",
            "SELECT 1; SELECT 2",
            "",
            "-- only a comment",
        ],
    )
    def test_rejected(self, sql):
        with pytest.raises(ValueError):
            validate_sql(sql)


class TestRunSql:
    def test_simple_select(self):
        result = run_sql("SELECT COUNT(*) AS n FROM orders")
        assert result.columns == ["n"]
        assert result.rows[0][0] > 0

    def test_max_rows_truncation(self):
        result = run_sql("SELECT order_id FROM orders", max_rows=5)
        assert len(result.rows) == 5
        assert result.truncated
        assert "truncated" in result.as_text()

    def test_write_rejected(self):
        with pytest.raises(ValueError):
            run_sql("DELETE FROM orders")


class TestIntrospection:
    def test_tables_exist(self):
        tables = list_tables()
        assert len(tables) > 0
        assert "orders" in tables

    def test_table_info_matches_db(self):
        for name in list_tables():
            info = table_info(name)
            assert info.columns, f"{name} has no columns"
            count = run_sql(f'SELECT COUNT(*) FROM "{name}"').rows[0][0]
            assert info.row_count == count

    def test_schema_card_contains_columns(self):
        info = table_info(list_tables()[0])
        card = render_schema_card(info)
        for col in info.columns:
            assert col["name"] in card

    def test_one_doc_per_table(self):
        ids, docs = build_schema_docs()
        assert len(ids) == len(list_tables())
        assert all(i.startswith("schema:") for i in ids)
        assert all(d.metadata["type"] == "schema" for d in docs)


class TestFeedIds:
    def test_stable_and_distinct(self):
        assert schema_id("orders") == "schema:orders"
        assert example_id("q1") == example_id("q1")
        assert example_id("q1") != example_id("q2")
        assert note_id("t") != example_id("t")


class TestAllTablesLine:
    def test_lists_every_table(self):
        line = all_tables_line()
        for table in list_tables():
            assert table in line


class TestExamplesFile:
    def test_seed_examples_valid(self):
        items = json.loads(EXAMPLES_FILE.read_text(encoding="utf-8"))
        assert items, "examples.json is empty"
        for item in items:
            assert item["question"].strip()
            validate_sql(item["sql"])  # every seed SQL must pass the guard

    def test_seed_examples_run(self):
        items = json.loads(EXAMPLES_FILE.read_text(encoding="utf-8"))
        for item in items:
            run_sql(item["sql"], max_rows=1)  # must execute against the DB
