"""RunStats: a LangChain callback that totals tokens and LLM calls across one
question, so `main.py eval` can report the cost of each answer alongside its
accuracy and you can compare prompt, model, or retrieval changes."""

import time
from dataclasses import dataclass, field

from langchain_core.callbacks import BaseCallbackHandler


@dataclass(eq=False)  # callbacks must stay hashable: LangChain merges them as a set
class RunStats(BaseCallbackHandler):
    """Totals token usage and LLM calls across one graph invocation.
    Pass as `config={"callbacks": [stats]}` to graph.invoke()."""

    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    started: float = field(default_factory=time.perf_counter)

    def on_llm_end(self, response, **kwargs) -> None:  # noqa: D401
        self.llm_calls += 1
        usage = (response.llm_output or {}).get("token_usage") or {}
        if not usage:
            for gen_list in response.generations:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    meta = getattr(msg, "usage_metadata", None) or {}
                    if meta:
                        usage = {
                            "prompt_tokens": meta.get("input_tokens", 0),
                            "completion_tokens": meta.get("output_tokens", 0),
                        }
                        break
                if usage:
                    break
        self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self.started) * 1000)
