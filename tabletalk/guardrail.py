"""Prompt guardrails, ported from a proven Go implementation
(LegalDocsAiAgent's internal/components/input_guardrail.go and
output_guardrail.go): fast, regex-only, no LLM.

Deliberately does NOT try to detect topical scope by regex — enumerating
every possible off-topic subject (coding, weather, jokes, ...) here is a
losing battle. That's the system prompt's job (see WRITE_SQL_SYSTEM in
graph.py and AGENT_SYSTEM in agent.py), using the constants below.
"""

import re

INJECTION_REFUSAL = "This request cannot be processed."  # deliberately vague
SCOPE_REFUSAL = (
    "I can only answer questions about the data in this database. "
    "That doesn't look like something I can look up here."
)
ROLE_BREAK_REFUSAL = (
    "I can only answer using the results of a query against this database, "
    "and this response didn't do that. Please try rephrasing your question."
)
OUT_OF_SCOPE_SENTINEL = "OUT_OF_SCOPE"  # what write_sql's LLM call emits as `sql`

_INJECTION_PATTERNS = [
    re.compile(
        r"ignore\s+(all\s+)?(previous|prior|above|your)\s+"
        r"(instructions?|prompts?|rules?|context|constraints?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(forget|disregard|override)\s+(your|all|the|previous)\s+"
        r"(instructions?|role|rules?|constraints?|guidelines?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(you\s+are\s+now|from\s+now\s+on\s+(you\s+are|be)|"
        r"your\s+new\s+(role|persona|identity)\s+is)",
        re.IGNORECASE,
    ),
    re.compile(r"\bpretend\s+(you\s+are|to\s+be)\b", re.IGNORECASE),
    re.compile(
        r"\b(jailbreak|DAN\s+mode|prompt\s+injection|"
        r"bypass\s+(safety|guardrail|restriction|filter))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(reveal|show|print|output|repeat|tell\s+me)\s+(your|the)\s+"
        r"(system\s+)?prompt",
        re.IGNORECASE,
    ),
]

_ROLE_BREAK_PATTERNS = [
    re.compile(r"\bAs an AI( language)? (model|assistant)[,\s]", re.IGNORECASE),
    re.compile(r"\bBased on my (training data|general knowledge)\b", re.IGNORECASE),
    re.compile(r"\bFrom my (training data|training)\b", re.IGNORECASE),
    re.compile(r"\bIn my training data\b", re.IGNORECASE),
]


def check_input(text: str) -> str | None:
    """Return INJECTION_REFUSAL if text matches a known prompt-injection or
    jailbreak pattern, else None. Regex-only, no LLM call."""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return INJECTION_REFUSAL
    return None


def check_output(text: str) -> bool:
    """True if text contains a role-break phrase indicating the LLM
    answered from general/training knowledge instead of query results."""
    return any(p.search(text) for p in _ROLE_BREAK_PATTERNS)
