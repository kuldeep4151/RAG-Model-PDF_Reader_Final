from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import os

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Summarize the conversation briefly, keeping only key facts."),
    ("human", "{conversation}")
])

llm = ChatGroq(
    model= "llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

summarizer = SUMMARY_PROMPT | llm | StrOutputParser()

conversation_summary = ""

def update_summary(history_messages):
    global conversation_summary
    
    convo_text = "\n".join(
        f"{m.type}: {m.content}" for m in history_messages
    )
    
    conversation_summary = summarizer.invoke({
        "conversation": convo_text
    })
    
def get_summary():
    return conversation_summary
