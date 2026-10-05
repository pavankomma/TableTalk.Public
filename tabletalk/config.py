"""Central configuration: env vars, paths, and factory functions for the
LLM, embeddings, and vector store. Clients are built lazily so importing
this module never requires credentials.

Both chat and embeddings run against an Azure OpenAI (Azure AI Foundry)
resource through its OpenAI-compatible v1 endpoint:
    https://<resource>.openai.azure.com/openai/v1
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
# Either a SQLite file path (default) or a full SQLAlchemy URL
# (postgresql://..., mysql+pymysql://..., mssql+pyodbc://..., snowflake://...)
DB_SETTING = os.getenv("TABLETALK_DB", str(DATA_DIR / "demo.db"))
EXAMPLES_FILE = ROOT / "examples" / "examples.json"
CHROMA_DIR = ROOT / ".chroma"
COLLECTION_NAME = "tabletalk"

# Agent limits
MAX_SQL_FIXES = 2      # LLM retries after a failed SQL execution
MAX_RESULT_ROWS = 50   # rows fetched from any query
SCHEMA_TOP_K = 4       # schema cards retrieved per question
EXAMPLES_TOP_K = 3     # sample queries retrieved per question
MAX_AGENT_STEPS = 6    # tool-calling agent: agent<->tool round trips before forcing an answer
MAX_DOSSIER_ROWS = 20  # rows fetched per connected table in get_record_history
MAX_DOSSIER_DEPTH = 2  # FK hops from the seed record in get_record_history

# Cross-encoder reranking (schema/note/example retrieval)
# Any HF repo with onnx/model.onnx + tokenizer.json works, e.g. the larger
# cross-encoder/ms-marco-MiniLM-L12-v2 or BAAI/bge-reranker-base. Compare
# on the golden set (main.py eval) before changing the default.
RERANKER_MODEL = os.getenv("TABLETALK_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
RERANKER_CACHE_DIR = ROOT / ".cache" / "reranker"
RERANK_OVERSAMPLE = 3   # fetch k * this many candidates before reranking down to k

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-5.4")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
)
AZURE_OPENAI_EMBEDDING_DIMENSIONS = int(
    os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "1536")
)


def _v1_base_url() -> str:
    """Accept either the bare resource URL or the full /openai/v1 endpoint."""
    url = AZURE_OPENAI_ENDPOINT.rstrip("/")
    if not url:
        raise RuntimeError(
            "Azure OpenAI is not configured. Set AZURE_OPENAI_API_KEY and "
            "AZURE_OPENAI_ENDPOINT in .env (see .env.example)."
        )
    if not url.endswith("/openai/v1"):
        url += "/openai/v1"
    return url


# Provider switch. "azure" (default) keeps the Azure OpenAI v1 path below;
# anything else goes through LangChain's init_chat_model/init_embeddings
# with the provider's own env vars (OPENAI_API_KEY, ANTHROPIC_API_KEY,
# OLLAMA host, ...) and the TABLETALK_LLM_MODEL / TABLETALK_EMBEDDING_MODEL names.
LLM_PROVIDER = os.getenv("TABLETALK_LLM_PROVIDER", "azure").strip().lower()
LLM_MODEL = os.getenv("TABLETALK_LLM_MODEL", "")
EMBEDDING_PROVIDER = os.getenv("TABLETALK_EMBEDDING_PROVIDER", LLM_PROVIDER)
EMBEDDING_MODEL = os.getenv("TABLETALK_EMBEDDING_MODEL", "")


def get_embeddings():
    if EMBEDDING_PROVIDER != "azure":
        from langchain.embeddings import init_embeddings

        if not EMBEDDING_MODEL:
            raise RuntimeError("Set TABLETALK_EMBEDDING_MODEL for a non-Azure embedding provider.")
        return init_embeddings(EMBEDDING_MODEL, provider=EMBEDDING_PROVIDER)

    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        base_url=_v1_base_url(),
        api_key=AZURE_OPENAI_API_KEY,
        model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        dimensions=AZURE_OPENAI_EMBEDDING_DIMENSIONS,
        max_retries=5,
    )


def get_vectorstore():
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


# Reasoning effort per role (gpt-5 family: minimal | low | medium | high).
# The fixed pipeline runs cheap; the retry rung and the agent think harder.
# Empty string = don't send the parameter (for deployments that reject it).
EFFORT_FAST = os.getenv("TABLETALK_EFFORT_FAST", "minimal")     # write_sql / answer / verify
EFFORT_RETRY = os.getenv("TABLETALK_EFFORT_RETRY", "high")      # second write_sql attempt
EFFORT_AGENT = os.getenv("TABLETALK_EFFORT_AGENT", "medium")    # tool-calling agent
SELF_CHECK = os.getenv("TABLETALK_SELF_CHECK", "0") == "1"      # graph.verify node
FULL_SCHEMA = os.getenv("TABLETALK_FULL_SCHEMA", "0") == "1"    # inline every schema card
# Learning loop: after a run that looked reliable, save the question->SQL pair
# to the feed as an "auto" example (ranked below human-confirmed ones).
AUTO_LEARN = os.getenv("TABLETALK_AUTO_LEARN", "0") == "1"


def get_llm(effort: str | None = None):
    """Chat model. `effort` is the reasoning effort for this call site
    (see EFFORT_*); None sends nothing extra."""
    if LLM_PROVIDER != "azure":
        from langchain.chat_models import init_chat_model

        if not LLM_MODEL:
            raise RuntimeError("Set TABLETALK_LLM_MODEL for a non-Azure LLM provider.")
        kwargs = {"max_retries": 5}
        if effort and LLM_PROVIDER == "openai":
            kwargs["reasoning_effort"] = effort
        return init_chat_model(LLM_MODEL, model_provider=LLM_PROVIDER, **kwargs)

    from langchain_openai import ChatOpenAI

    # gpt-5-family deployments reject non-default temperature and the legacy
    # max_tokens parameter, so we deliberately pass neither.
    kwargs = {}
    if effort:
        kwargs["reasoning_effort"] = effort
    return ChatOpenAI(
        base_url=_v1_base_url(),
        api_key=AZURE_OPENAI_API_KEY,
        model=AZURE_OPENAI_CHAT_DEPLOYMENT,
        max_retries=5,
        **kwargs,
    )
