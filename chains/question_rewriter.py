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
- Only rewrite if necessary
"""
    ),
    ("human", "Conversation:\n{history}\n\nQuestion:\n{question}")
])

def build_question_rewriter(llm):
    return REWRITE_PROMPT | llm | StrOutputParser()
