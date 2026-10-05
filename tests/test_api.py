"""REST surface: response shape, SSE streaming, and feed-route validation.
tiered_ask is stubbed, so no LLM is called."""

import json

import pytest
from fastapi.testclient import TestClient

import api


@pytest.fixture
def client():
    return TestClient(api.app)


def _fake_ask(question, history=""):
    return {"answer": f"echo: {question}", "sql": "SELECT 1",
            "result_columns": ["x"], "result_rows": [[1]],
            "escalated": True, "tier": "retry", "history_seen": history}


class TestAsk:
    def test_health(self, client):
        assert client.get("/health").json() == {"status": "ok"}

    def test_ask_maps_state_to_response(self, client, monkeypatch):
        monkeypatch.setattr(api, "tiered_ask", _fake_ask)
        r = client.post("/ask", json={"question": "how many?"})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == "echo: how many?"
        assert body["sql"] == "SELECT 1"
        assert body["columns"] == ["x"] and body["rows"] == [[1]]
        assert body["escalated"] is True and body["tier"] == "retry"

    def test_question_required(self, client):
        assert client.post("/ask", json={}).status_code == 422


class TestStream:
    def test_events_end_with_final(self, client, monkeypatch):
        from tabletalk import tiered_ask

        async def fake_stream(question, history=""):
            yield {"type": "status", "node": "retrieve", "escalated": False}
            yield {"type": "token", "text": "hi"}
            yield {"type": "final", "answer": "hi"}

        monkeypatch.setattr(tiered_ask, "stream", fake_stream)
        r = client.post("/ask/stream", json={"question": "q"})
        events = [line[len("event: "):].strip() for line in r.text.splitlines()
                  if line.startswith("event: ")]
        assert events == ["status", "token", "final"]
        last = [line for line in r.text.splitlines() if line.startswith("data: ")][-1]
        assert json.loads(last[len("data: "):])["answer"] == "hi"

    def test_failure_becomes_error_event(self, client, monkeypatch):
        from tabletalk import tiered_ask

        async def broken(question, history=""):
            raise RuntimeError("boom")
            yield  # pragma: no cover — makes this an async generator

        monkeypatch.setattr(tiered_ask, "stream", broken)
        r = client.post("/ask/stream", json={"question": "q"})
        assert "event: error" in r.text and "boom" in r.text


class TestFeedValidation:
    def test_example_needs_question_and_sql(self, client):
        r = client.post("/api/feed/example", json={"question": " ", "sql": "SELECT 1"})
        assert r.status_code == 422

    def test_note_needs_text(self, client):
        assert client.post("/api/feed/note", json={"text": ""}).status_code == 422

    def test_feed_page_served(self, client):
        r = client.get("/feed")
        assert r.status_code == 200 and "TableTalk" in r.text
