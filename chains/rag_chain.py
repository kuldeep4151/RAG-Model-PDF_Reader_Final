from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from chains.prompts import RAG_PROMPT
import os
from utils.memory_utils import is_summary_question
from chains.question_rewriter import rewrite_if_needed
from utils.retrieval_utils import retrieve_relevant_docs
from utils.context_compression import compress_docs

def build_rag_chain(llm):
    """
    This chain ONLY formats prompt + calls LLM.
    NO retrieval here.
    """

    return (
        RAG_PROMPT
        | llm
        | StrOutputParser()
    )


def run_rag(vectorstore, llm, question, history):
    from chains.question_rewriter import rewrite_if_needed
    from utils.retrieval_utils import retrieve_relevant_docs
    from utils.context_compression import compress_docs

    rewritten = rewrite_if_needed(llm, question, history)

    docs = retrieve_relevant_docs(vectorstore, rewritten)
    if not docs:
        return "I could not find relevant information in the document."

    context, _ = compress_docs(
        docs,
        query=rewritten,
        max_chars=4000,
        mode="qa"
    )
    
    if not context.strip():
        # Compression removed everything → fallback to raw chunks
        fallback_docs = vectorstore.similarity_search(
            rewritten,
            k=5
        )

        context = "\n".join(
            d.page_content[:800]
            for d in fallback_docs
        )

        if not context.strip():
            return "I cannot answer this based on the provided document."

    rag_chain = build_rag_chain(llm)

    return rag_chain.invoke({
        "context": context,
        "question": rewritten,
        "history": history
    })
