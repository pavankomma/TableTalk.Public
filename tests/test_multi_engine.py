"""No-LLM tests for the SQLAlchemy/sqlglot layer: AST-based SQL validation,
dialect helpers, FK-neighbor graph, the golden eval set, and the eval row
matcher. Expectations derive from the live database at runtime."""

import json
from pathlib import Path

import pytest

from main import rows_match
from tabletalk.db import (
    dialect_name,
    list_tables,
    prompt_dialect,
    run_sql,
    validate_sql,
)
from tabletalk.retriever import fk_neighbors

GOLDEN = json.loads(
    (Path(__file__).parent.parent / "golden" / "golden.json").read_text(
        encoding="utf-8"
    )
)


class TestAstValidation:
    def test_semicolon_inside_string_literal_ok(self):
        # the old regex guard rejected this; the AST guard must not
        sql = "SELECT ';' AS s"
        assert validate_sql(sql) == sql

    def test_union_ok(self):
        assert validate_sql("SELECT 1 UNION SELECT 2").startswith("SELECT")

    def test_subquery_ok(self):
        validate_sql("SELECT * FROM (SELECT 1 AS x) WHERE x = 1")

    @pytest.mark.parametrize(
        "sql",
        [
            "VACUUM",
            "ATTACH DATABASE 'x.db' AS x",
            "CREATE TABLE t (x INT)",
            "REPLACE INTO orders VALUES (1)",
        ],
    )
    def test_more_statements_rejected(self, sql):
        with pytest.raises(ValueError):
            validate_sql(sql)


class TestDialect:
    def test_default_is_sqlite(self):
        assert dialect_name() == "sqlite"
        assert prompt_dialect() == "SQLite"


class TestFkNeighbors:
    def test_order_items_join_orders_both_ways(self):
        neighbors = fk_neighbors()
        assert "orders" in neighbors["order_items"]
        assert "order_items" in neighbors["orders"]

    def test_all_neighbor_tables_exist(self):
        tables = set(list_tables())
        for table, joined in fk_neighbors().items():
            assert table in tables
            assert joined <= tables


GOLDEN_SQL = [g for g in GOLDEN if "sql" in g]
GOLDEN_REFUSALS = [g for g in GOLDEN if "expect" in g]


class TestGoldenSet:
    def test_golden_set_is_populated(self):
        # main.py eval divides by len(golden); an empty set is a silent no-op
        assert len(GOLDEN_SQL) >= 30
        assert GOLDEN_REFUSALS, "need at least one out_of_scope/blocked case"

    def test_golden_items_are_well_formed(self):
        for g in GOLDEN:
            assert g["question"].strip()
            assert ("sql" in g) != ("expect" in g), g
            if "expect" in g:
                assert g["expect"] in ("out_of_scope", "blocked"), g

    def test_golden_questions_unique(self):
        questions = [g["question"] for g in GOLDEN]
        assert len(questions) == len(set(questions))

    def test_golden_does_not_duplicate_feed_examples(self):
        # examples.json is retrieved as few-shot context, so overlap would
        # make the eval measure recall of the feed rather than SQL writing
        examples = json.loads(
            (Path(__file__).parent.parent / "examples" / "examples.json").read_text(
                encoding="utf-8"
            )
        )
        feed_questions = {e["question"] for e in examples}
        assert not feed_questions & {g["question"] for g in GOLDEN}

    @pytest.mark.parametrize(
        "item", GOLDEN_SQL, ids=[g["question"][:40] for g in GOLDEN_SQL]
    )
    def test_golden_sql_validates_and_executes(self, item):
        validate_sql(item["sql"])
        result = run_sql(item["sql"])
        assert result.rows, f"golden SQL returned no rows: {item['sql']}"


class TestRowsMatch:
    def test_exact_match(self):
        assert rows_match([("a", 1)], [("a", 1)])

    def test_extra_columns_and_reorder_tolerated(self):
        assert rows_match([(1, "a", "junk")], [("a", 1)])

    def test_row_count_mismatch_fails(self):
        assert not rows_match([("a", 1)], [("a", 1), ("b", 2)])

    def test_wrong_value_fails(self):
        assert not rows_match([("a", 2)], [("a", 1)])

    def test_float_normalization(self):
        assert rows_match([("US", 523.0600000001)], [("US", 523.06)])
