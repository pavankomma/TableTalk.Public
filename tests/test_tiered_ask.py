"""No-LLM tests for tabletalk/tiered_ask.py's should_escalate() — pure
decision logic over a plain state dict. ask() itself isn't tested here
(calls the LLM, same boundary as graph.py/agent.py today)."""

from tabletalk.tiered_ask import should_escalate


class TestShouldEscalate:
    def test_error_escalates(self):
        assert should_escalate({"error": "some SQL error"}) is True

    def test_zero_rows_no_error_escalates(self):
        state = {
            "sql": "SELECT COUNT(*) AS n FROM orders",
            "result_columns": ["n"],
            "result_rows": [],
            "error": None,
        }
        assert should_escalate(state) is True

    def test_successful_rows_do_not_escalate(self):
        state = {
            "sql": "SELECT COUNT(*) AS n FROM orders",
            "result_columns": ["n"],
            "result_rows": [[5]],
            "error": None,
        }
        assert should_escalate(state) is False

    def test_empty_state_does_not_misfire(self):
        assert should_escalate({}) is False

    def test_verify_failed_escalates(self):
        state = {"sql": "SELECT 1", "result_columns": ["x"], "result_rows": [[1]],
                 "error": None, "verify_failed": True}
        assert should_escalate(state) is True


class TestRetryNote:
    def test_error(self):
        from tabletalk.tiered_ask import retry_note

        assert retry_note({"sql": "SELECT x", "error": "no such column"}) == (
            "SQL: 'SELECT x' -> error: no such column"
        )

    def test_zero_rows_and_caveats(self):
        from tabletalk.tiered_ask import retry_note

        assert retry_note({"sql": "S", "result_rows": []}).endswith("it returned 0 rows")
        assert retry_note({"sql": "S", "caveats": ["wrong table"]}).endswith("wrong table")
