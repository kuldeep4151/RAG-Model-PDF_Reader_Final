from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You must answer ONLY using the information from the provided PDF context.\n"
        "limit response size to 30 words(Maximum).\n"
        "If the answer is not contained in the PDF, reply:\n"
        "'I cannot answer this based on the PDF.'\n\n"
        "PDF Context:\n{context}"
    ),
    (
        "human",
        "Question: {question}"
    )
])
