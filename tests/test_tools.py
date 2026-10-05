"""No-LLM tests for tabletalk/tools.py: expectations derived from the live
database at runtime, same style as test_units.py.

search_schema is intentionally not covered here — it calls
get_vectorstore(), which needs embedding credentials, the same reason
retriever.retrieve_context() is untested today.
"""

import pytest

from tabletalk import db, tools
from tabletalk.db import list_tables, run_sql, table_info
from tabletalk.retriever import fk_neighbors
from tabletalk.schema_docs import render_schema_card


class TestListTables:
    def test_matches_db(self):
        rendered = tools.list_tables()
        for name in list_tables():
            assert name in rendered


class TestGetTableDetail:
    def test_known_table_matches_render_schema_card(self):
        name = list_tables()[0]
        assert tools.get_table_detail(name) == render_schema_card(table_info(name))

    def test_unknown_table_does_not_raise(self):
        result = tools.get_table_detail("NotARealTable123")
        assert isinstance(result, str)
        assert "NotARealTable123" in result


class TestGetFkNeighbors:
    def test_known_table_with_neighbors(self):
        neighbors_by_table = fk_neighbors()
        table = next(t for t, n in neighbors_by_table.items() if n)
        result = tools.get_fk_neighbors(table)
        for neighbor in neighbors_by_table[table]:
            assert neighbor in result

    def test_table_without_neighbors(self):
        neighbors_by_table = fk_neighbors()
        no_fk_table = next(
            (t for t in list_tables() if not neighbors_by_table.get(t)), None
        )
        if no_fk_table is None:
            return  # every table in this DB has a foreign key; nothing to assert
        result = tools.get_fk_neighbors(no_fk_table)
        assert "no foreign-key relationships" in result

    def test_unknown_table_does_not_raise(self):
        result = tools.get_fk_neighbors("NotARealTable123")
        assert isinstance(result, str)
        assert "NotARealTable123" in result


class TestRunSql:
    def test_ok(self):
        result = tools.run_sql("SELECT COUNT(*) AS n FROM orders")
        assert result["ok"] is True
        assert result["columns"] == ["n"]
        assert result["rows"][0][0] > 0
        assert "n" in result["text"]
        assert result["error"] is None

    def test_write_statement_returns_error_not_raise(self):
        result = tools.run_sql("DELETE FROM orders")
        assert result["ok"] is False
        assert result["error"]

    def test_bad_table_returns_error(self):
        result = tools.run_sql("SELECT * FROM NotATable")
        assert result["ok"] is False
        assert result["error"]


class TestAllForeignKeys:
    def test_known_relationship(self):
        keys = db.all_foreign_keys()
        assert any(
            fk["table"] == "order_items"
            and fk["column"] == "order_id"
            and fk["ref_table"] == "orders"
            and fk["ref_column"] == "order_id"
            for fk in keys
        )


class TestFetchByColumn:
    def test_basic_match(self):
        customer = run_sql("SELECT customer_id FROM orders LIMIT 1").rows[0][0]
        result = db.fetch_by_column("orders", "customer_id", customer)
        assert result.rows
        idx = result.columns.index("customer_id")
        assert all(row[idx] == customer for row in result.rows)

    def test_limit_enforced(self):
        result = db.fetch_by_column("orders", "status", "delivered", limit=3)
        assert len(result.rows) <= 3

    def test_no_match(self):
        result = db.fetch_by_column("orders", "order_id", -1)
        assert result.rows == []


class TestGetRecordHistory:
    def test_customer_dossier_traverses(self):
        # a customer with both orders and support tickets: the dossier must
        # reach orders and tickets (1 hop) and line items (2 hops)
        customer = run_sql(
            "SELECT customer_id FROM support_tickets WHERE order_id IS NOT NULL LIMIT 1"
        ).rows[0][0]
        result = tools.get_record_history("customers", "customer_id", str(customer))
        assert result.startswith("customers:")
        assert "orders (via customer_id" in result
        assert "support_tickets (via customer_id" in result
        assert "order_items (via order_id" in result

    def test_order_reaches_product(self):
        order = run_sql("SELECT order_id FROM order_items LIMIT 1").rows[0][0]
        result = tools.get_record_history("orders", "order_id", str(order))
        assert "order_items" in result
        assert "products (via product_id" in result

    def test_unknown_table_does_not_raise(self):
        result = tools.get_record_history("NotARealTable123", "id", "1")
        assert isinstance(result, str)
        assert "NotARealTable123" in result

    def test_unknown_column_does_not_raise(self):
        result = tools.get_record_history("orders", "NotARealColumn123", "1")
        assert "NotARealColumn123" in result

    def test_unknown_value_does_not_raise(self):
        result = tools.get_record_history("orders", "order_id", "-1")
        assert "No rows found" in result
