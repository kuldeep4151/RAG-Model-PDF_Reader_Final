from enum import Enum


class Intent(Enum):
    SUMMARY = "summary"
    BOOLEAN_PRESENCE = "boolean_presence"
    SEMANTIC_QA = "semantic_qa"


def route_intent(question: str) -> Intent:
    q = question.lower().strip()

    # ---- SUMMARY ----
    if any(w in q for w in ["summary", "summarize", "overview", "brief"]):
        return Intent.SUMMARY

    # ---- BOOLEAN PRESENCE ----
    if q.startswith((
        "is ",
        "are ",
        "was ",
        "were ",
        "does ",
        "do ",
        "did "
    )):
        return Intent.BOOLEAN_PRESENCE

    # ---- EVERYTHING ELSE (LLM HANDLES METADATA & QA) ----
    return Intent.SEMANTIC_QA
