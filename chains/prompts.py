from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a factual assistant.\n"
        "Answer ONLY using the provided PDF context.\n"
        "Do NOT use outside knowledge.\n"
        "- Maximum 50 words (EXCEPT when explicitly asked for a summary use 300 words)\n"
        "If the answer is not found in the PDF, reply exactly:\n"
        "'I cannot answer this based on the PDF.'\n\n"
        "PDF Context:\n{context}"
    ),
    (
        "human",
        "Question: {question}"
    )
])

