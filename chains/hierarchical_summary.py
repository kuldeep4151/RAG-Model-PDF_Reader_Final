import time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# ---------------- LLM ----------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# ---------------- PROMPTS ----------------

CHUNK_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
        "Summarize the following text concisely.\n"
        "Keep only factual, important information.\n"
        "Do NOT add interpretation or examples."),
    ("human", "{text}")
])

MERGE_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
        "Merge the following summaries into one coherent summary.\n"
        "Remove redundancy.\n"
        "Keep only key ideas."),
    ("human", "{summaries}")
])

chunk_summarizer = CHUNK_SUMMARY_PROMPT | llm | StrOutputParser()
merge_summarizer = MERGE_SUMMARY_PROMPT | llm | StrOutputParser()


# ---------------- MAP ----------------
def summarize_chunks(docs, max_chunks=20, max_chars=1200):
    """
    MAP step: summarize chunks independently (throttled)
    """
    summaries = []

    for doc in docs[:max_chunks]:
        text = doc.page_content[:max_chars]  # HARD CAP
        summary = chunk_summarizer.invoke({"text": text})
        summaries.append(summary)

        time.sleep(1.2)  #wait for 1.2 min

    return summaries


# ---------------- REDUCE ----------------
def reduce_summaries(summaries, batch_size=4):
    """
    REDUCE step: recursively merge summaries
    """
    current = summaries

    while len(current) > 1:
        next_round = []

        for i in range(0, len(current), batch_size):
            batch = current[i:i + batch_size]

            merged = merge_summarizer.invoke({
                "summaries": "\n".join(batch)
            })

            next_round.append(merged)
            time.sleep(1.2) 

        current = next_round

    return current[0]


# ---------------- PIPELINE ----------------
def hierarchical_summarize(docs):
    """
    Full Map-Reduce summarization pipeline
    """
    chunk_summaries = summarize_chunks(docs)
    final_summary = reduce_summaries(chunk_summaries)
    return final_summary
