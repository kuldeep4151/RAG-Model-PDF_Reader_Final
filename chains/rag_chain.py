from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from chains.prompts import RAG_PROMPT
import os

def build_rag_chain(vectorstore):
    llm = ChatGroq(
        model="groq/compound",
        api_key=os.getenv("GROQ_API_KEY")
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 15}  # 🔥 IMPORTANT
    )

    rag_chain = (
        {
            "context": lambda x: retriever.invoke(x["question"]),
            "question": lambda x: x["question"]
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return rag_chain
