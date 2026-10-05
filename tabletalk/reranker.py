"""Cross-encoder reranking for retrieved schema cards/notes/examples — a
small (22.7M param), fully local ONNX model that jointly scores each
(query, candidate) pair, more precise than the bi-encoder similarity
search that finds the candidates in the first place.

Never raises: any failure (no network on first download, corrupt cache,
bad input) falls back to the original similarity-search order, since
reranking is a quality improvement, not a correctness dependency.
"""

from functools import lru_cache

from langchain_core.documents import Document

from .config import RERANKER_CACHE_DIR, RERANKER_MODEL


@lru_cache(maxsize=1)
def _load():
    import numpy as np  # noqa: F401 (imported for side effect check; used below)
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    model_path = hf_hub_download(
        RERANKER_MODEL,
        filename="onnx/model.onnx",
        cache_dir=str(RERANKER_CACHE_DIR),
    )
    tokenizer_path = hf_hub_download(
        RERANKER_MODEL, filename="tokenizer.json", cache_dir=str(RERANKER_CACHE_DIR)
    )
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_padding()
    tokenizer.enable_truncation(max_length=512)
    input_names = {inp.name for inp in session.get_inputs()}
    return session, tokenizer, input_names


def rerank(query: str, docs: list[Document], top_k: int) -> list[Document]:
    """Return the top_k docs, reordered by cross-encoder relevance to
    query. Falls back to docs[:top_k] in original order on any failure."""
    if not docs:
        return docs
    try:
        import numpy as np

        session, tokenizer, input_names = _load()
        encodings = tokenizer.encode_batch([(query, d.page_content) for d in docs])
        feed = {}
        if "input_ids" in input_names:
            feed["input_ids"] = np.array([e.ids for e in encodings], dtype=np.int64)
        if "attention_mask" in input_names:
            feed["attention_mask"] = np.array(
                [e.attention_mask for e in encodings], dtype=np.int64
            )
        if "token_type_ids" in input_names:
            feed["token_type_ids"] = np.array(
                [e.type_ids for e in encodings], dtype=np.int64
            )
        scores = session.run(None, feed)[0].reshape(-1)
        order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        return [docs[i] for i in order[:top_k]]
    except Exception:
        return docs[:top_k]
