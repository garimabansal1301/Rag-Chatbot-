"""When a chat turn must pause for a human (see the review nodes in app/chat/engine.py).

Three kinds of review:
  - unsupported_answer: excerpts were retrieved, but the answer cites none of them (or cites excerpts that do not
    exist), so its support by the documents could not be confirmed. This is a structural check of the answer's
    citations, not a confidence score: no numeric confidence is computed or shown.
  - sensitive_request: the user's message matches a sensitive support topic (refunds, account closure, legal,
    personal data, safety), or the model itself asked for a reviewer with the request_human_review tool.
  - consequential_action: the model wants to call a tool marked with `mark_requires_approval` (for example a future
    tool that changes customer records). The call is held until a reviewer approves it; no such tool ships today.

The model can only *ask* for a review. Decisions come from the reviewer through the server (app/services.py),
never from the model or from a tool.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool

REVIEW_TOOL_NAME = "request_human_review"
REQUIRES_APPROVAL = "requires_approval"  # key in a tool's metadata
MAX_REASON_CHARS = 300
MAX_REASONS = 5

REVIEW_PROMPT = (
    "A support agent reviews some answers before the user sees them. Call request_human_review (then still write "
    "the reply you propose) when the request needs a person: refunds or compensation, account or record changes, "
    "complaints, legal or safety issues, personal data, or a question about the company's own policies that the "
    "excerpts do not answer."
)

# Sensitive support topics: (label shown to the reviewer, pattern on the user's message).
SENSITIVE_TOPICS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (label, re.compile(pattern, re.IGNORECASE))
    for label, pattern in (
        ("refunds and billing disputes", r"\b(refund\w*|charge-?backs?|money back|reimburs\w+|overcharg\w+)\b"),
        (
            "account closure or changes",
            r"\b(close|delete|cancel|deactivate)\s+(my|our|the customer'?s?)\s+account\b"
            r"|\bchange\s+(my|the customer'?s?)\s+(bank|billing|email|address|password)",
        ),
        ("legal matters", r"\b(lawyer|attorney|lawsuit|legal action|sue|suing|court)\b"),
        (
            "personal data",
            r"\b(gdpr|data breach|(delete|erase|export)\s+(my|all my)\s+(personal\s+)?data|my personal data)\b",
        ),
        (
            "safety or harassment",
            r"\b(suicid\w*|self-?harm|kill myself|being harass\w*|harassing me|threatened me|threatening me)\b",
        ),
        ("formal complaints", r"\b(formal complaint|file a complaint|escalate (this|my))\b"),
    )
)

_CITATION = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]")


@dataclass(frozen=True)
class ReviewPolicy:
    """Which review checks run (consequential actions always need approval, whatever the policy)."""

    check_support: bool = True  # unsupported_answer for answers that cite no retrieved excerpt
    sensitive_topics: bool = True  # sensitive_request for messages matching SENSITIVE_TOPICS
    model_can_request: bool = True  # bind the request_human_review tool


def mark_requires_approval(target: BaseTool) -> BaseTool:
    """Mark a tool as consequential: the graph pauses for approval before every call of it."""
    target.metadata = {**(target.metadata or {}), REQUIRES_APPROVAL: True}
    return target


def requires_approval(target: BaseTool) -> bool:
    return bool((target.metadata or {}).get(REQUIRES_APPROVAL))


@tool(REVIEW_TOOL_NAME)
def request_human_review(
    category: Literal["sensitive_request", "unsupported_answer"],
    reason: str,
) -> str:
    """Ask a human support agent to review your answer before the user sees it.

    Use it for refunds or compensation, account or record changes, complaints, legal or safety issues,
    personal data, or company-policy questions that the document excerpts do not answer.
    category: "sensitive_request" or "unsupported_answer". reason: one short sentence, no personal data.
    After calling it, write the reply you propose; the reviewer approves, edits or rejects it.
    """
    return "A support agent will review your reply before the user sees it. Now write the reply you propose."


def _clean(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= MAX_REASON_CHARS else text[: MAX_REASON_CHARS - 1].rstrip() + "…"


def current_turn(messages: list[AnyMessage]) -> list[AnyMessage]:
    """The messages after the newest user message (this turn's model calls and tool results)."""
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return messages[index + 1 :]
    return list(messages)


def sensitive_reasons(question: str) -> list[str]:
    return [
        f"The message is about {label}, which a support agent must check."
        for label, pattern in SENSITIVE_TOPICS
        if pattern.search(question)
    ]


def model_requests(turn: list[AnyMessage]) -> list[tuple[str, str]]:
    """(category, reason) of each request_human_review call the model made this turn."""
    found = []
    for message in turn:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                if call["name"] == REVIEW_TOOL_NAME:
                    args = call.get("args") or {}
                    category = args.get("category")
                    category = (
                        category if category in ("sensitive_request", "unsupported_answer") else "sensitive_request"
                    )
                    found.append((category, _clean(args.get("reason") or "No reason given.")))
    return found


def used_other_tools(turn: list[AnyMessage]) -> bool:
    """True if a tool other than request_human_review returned a result this turn (e.g. a dataset analysis)."""
    return any(isinstance(m, ToolMessage) and m.name != REVIEW_TOOL_NAME for m in turn)


def support_reasons(answer: str, excerpt_count: int) -> list[str]:
    """Why an answer is not confirmed by the retrieved excerpts ([] when it cites at least one valid excerpt)."""
    if excerpt_count == 0:
        return []
    cited = {int(n) for group in _CITATION.findall(answer) for n in group.split(",")}
    valid, invalid = (
        {n for n in cited if 1 <= n <= excerpt_count},
        sorted(n for n in cited if not 1 <= n <= excerpt_count),
    )
    reasons = []
    if invalid:
        listed = ", ".join(f"[{n}]" for n in invalid[:5])
        reasons.append(f"The answer cites {listed}, but only {excerpt_count} excerpt(s) were retrieved.")
    if not valid:
        reasons.append(
            f"The answer does not cite any of the {excerpt_count} retrieved document excerpt(s), "
            "so its support by the documents could not be confirmed."
        )
    return reasons
