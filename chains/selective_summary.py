from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os

# LLM (cheap + fast)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Summarize the following text concisely.\n"
        "Keep only key ideas and facts.\n"
        "Do NOT add interpretation or examples."
    ),
    ("human", "{text}")
])

summarizer = SUMMARY_PROMPT | llm | StrOutputParser()


def selective_summarize(vectorstore, docs, k=5, max_chars=1200):

    # 1. Ask FAISS for representative chunks
    representative_docs = vectorstore.similarity_search(
        "What is this document mainly about?",
        k=k
    )

    # 2. Build compact context
    context = ""
    for d in representative_docs:
        if len(context) + len(d.page_content) > max_chars:
            break
        context += d.page_content + "\n"

    # 3. Summarize only selected context
    summary = summarizer.invoke({"text": context})
    return summary
