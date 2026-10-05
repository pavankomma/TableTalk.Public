"""Tests for tabletalk/reranker.py. Unlike search_schema/retrieve_context,
the reranker itself needs no LLM/API credentials — it's a local ONNX
classifier, not a hosted API. First run downloads the ~90MB model from
Hugging Face if not already cached under .cache/reranker/.
"""

from langchain_core.documents import Document

from tabletalk import reranker
from tabletalk.db import table_info
from tabletalk.schema_docs import render_schema_card


class TestRerank:
    def test_ranks_relevant_doc_first(self):
        ticket_card = Document(page_content=render_schema_card(table_info("support_tickets")))
        unrelated_card = Document(page_content=render_schema_card(table_info("categories")))
        result = reranker.rerank(
            "How many urgent support tickets are still open?",
            [unrelated_card, ticket_card],
            top_k=1,
        )
        assert result[0] is ticket_card

    def test_empty_docs_returns_empty(self):
        assert reranker.rerank("anything", [], top_k=5) == []

    def test_top_k_truncates(self):
        docs = [Document(page_content=f"Table: T{i}.") for i in range(5)]
        result = reranker.rerank("a query", docs, top_k=2)
        assert len(result) == 2


class TestRerankFallback:
    def test_falls_back_to_original_order_on_load_failure(self, monkeypatch):
        def broken_load():
            raise RuntimeError("simulated model load failure")

        monkeypatch.setattr(reranker, "_load", broken_load)
        docs = [Document(page_content=f"Table: T{i}.") for i in range(3)]
        result = reranker.rerank("a query", docs, top_k=2)
        assert result == docs[:2]
