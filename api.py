from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, UploadFile, File
import os
import tempfile

from loaders.pdf_loader import load_pdf
from vectorstore.store import build_vector_store
from chains.memory_chain import get_memory_module
from chains.rag_chain import run_rag
from langchain_groq import ChatGroq
from pydantic import BaseModel

app = FastAPI()

# Global RAG state
RAG_STATE = {
    "docs": None,
    "vectorstore": None,
    "memory": None,
    "llm": None
}

def build_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        suffix = os.path.splitext(file.filename)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
            
        docs = load_pdf(tmp_path)
        if not docs:
            return {"status": "error", "message": "No content extracted from PDF"}
        
        vectorstore = build_vector_store(docs)
        memory = get_memory_module()
        llm = build_llm()
        
        # Store in global RAG state
        RAG_STATE["docs"] = docs
        RAG_STATE["vectorstore"] = vectorstore
        RAG_STATE["memory"] = memory
        RAG_STATE["llm"] = llm
        
        return {
            "status": "success",
            "pages": len(docs),
            "message": "PDF uploaded and processed successfully."
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

class QuestionRequest(BaseModel):
    question: str
    
@app.post("/ask")
def ask_question(request: QuestionRequest):
    if RAG_STATE["vectorstore"] is None:
        return {"error": "No PDF uploaded. Call /upload first."}
    
    question = request.question
    memory = RAG_STATE["memory"]
    
    history_msgs = memory.load_memory_variables({})["history"]
    raw_history = "\n".join(
        f"{msg.type}: {msg.content}"
        for msg in history_msgs[-4:]
    )

    history_text = raw_history
    
    answer = run_rag(
        vectorstore=RAG_STATE["vectorstore"],
        llm=RAG_STATE["llm"],
        question=question,
        history=history_text
    )
    memory.chat_memory.add_user_message(question)
    memory.chat_memory.add_ai_message(answer)

    return {
        "question": question,
        "answer": answer
    }