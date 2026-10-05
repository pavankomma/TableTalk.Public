"""graph.py prompt assembly and routing, without an LLM."""

from tabletalk import graph
from tabletalk.graph import _route_after_execute, write_sql_messages


class TestWriteSqlMessages:
    BASE = {"context": "CTX", "question": "how many?"}

    def test_order_is_context_then_question(self):
        human = write_sql_messages(self.BASE)[1].content
        assert human.index("Database context") < human.index("Question: how many?")
        assert "SQLite" in write_sql_messages(self.BASE)[0].content

    def test_retry_note_included(self):
        human = write_sql_messages({**self.BASE, "retry_note": "SQL: 'x' -> 0 rows"}).content \
            if False else write_sql_messages({**self.BASE, "retry_note": "SQL: 'x' -> 0 rows"})[1].content
        assert "judged unreliable" in human and "SQL: 'x' -> 0 rows" in human

    def test_error_fix_included(self):
        human = write_sql_messages({**self.BASE, "error": "boom", "sql": "SELECT z"})[1].content
        assert "previous attempt failed" in human and "boom" in human

    def test_history_included(self):
        human = write_sql_messages({**self.BASE, "history": "Q: a\nA: b"})[1].content
        assert "Conversation so far" in human


class TestRouting:
    def test_error_retries_within_cap(self):
        assert _route_after_execute({"error": "x", "attempts": 1}) == "write_sql"

    def test_error_past_cap_goes_on(self):
        out = _route_after_execute({"error": "x", "attempts": graph.MAX_SQL_FIXES + 1})
        assert out in ("verify", "answer")

    def test_self_check_flag(self, monkeypatch):
        monkeypatch.setattr(graph, "SELF_CHECK", True)
        assert _route_after_execute({"error": None}) == "verify"
        monkeypatch.setattr(graph, "SELF_CHECK", False)
        assert _route_after_execute({"error": None}) == "answer"


class TestVerifyNoLlm:
    def test_skips_when_no_sql_or_error(self):
        assert graph.verify({"error": "x", "sql": "S"})["verify_failed"] is False
        assert graph.verify({"sql": ""})["verify_failed"] is False
