from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Rewrite the user's question into a fully self-contained question
using the conversation history.

Rules:
- Do NOT answer the question
- Do NOT add new information
- Preserve all entities, constraints, and intent
- Only rewrite if the question depends on conversation context
"""
    ),
    ("human", "Conversation:\n{history}\n\nQuestion:\n{question}")
])


def build_question_rewriter(llm):
    return REWRITE_PROMPT | llm | StrOutputParser()


def rewrite_if_needed(llm, question: str, history: str) -> str:
    """
    Rewrite ONLY if the question is conversational.
    Factual / list / lookup questions must NOT be rewritten.
    """

    factual_triggers = [
        "list",
        "name",
        "names",
        "who",
        "when",
        "where",
        "which",
        "how many",
        "define",
        "explain",
        "give",
        "show"
    ]

    q_lower = question.lower()

    # DO NOT rewrite factual or lookup questions
    if any(trigger in q_lower for trigger in factual_triggers):
        return question

    # Rewrite only if there is history and the question is ambiguous
    if not history or len(history.strip()) == 0:
        return question

    return build_question_rewriter(llm).invoke({
        "history": history,
        "question": question
    })
