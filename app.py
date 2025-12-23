import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from loaders.pdf_loader import load_pdf
from vectorstore.store import build_vector_store

from utils.memory_utils import needs_raw_history
from chains.memory_chain import get_memory_module
from chains.rag_chain import run_rag
from utils.boolean_search import boolean_presence_search
from utils.intent_router import route_intent, Intent

from chains.hierarchical_summary import chunk_summarizer, merge_summarizer

# -------------------------
# Environment
# -------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found")

def build_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
        temperature=0
    )

# -------------------------
# MAIN
# -------------------------

def main():
    pdf_path = input("Enter PDF file path or directory path:\n> ").strip()

    if not os.path.exists(pdf_path):
        print("Error: Path does not exist.")
        return

    print("\nLoading PDF...")
    docs = load_pdf(pdf_path)

    print("Building vector store...")
    vectorstore = build_vector_store(docs)

    print("Initializing memory...")
    memory = get_memory_module()

    llm = build_llm()

    print("\nChatbot is ready! Type 'exit' to quit.\n")

    while True:
        user_input = input("🧔: ").strip()

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # --------- LOAD MEMORY ---------
        history_msgs = memory.load_memory_variables({})["history"]

        raw_history = "\n".join(
            f"{msg.type}: {msg.content}"
            for msg in history_msgs[-4:]
        )

        history_text = ""
        if needs_raw_history(user_input):
            history_text = raw_history

        intent = route_intent(user_input)
        # YOUR HYBRID SELECTIVE + MAP-REDUCE SUMMARY
        if intent == Intent.SUMMARY:
            print("\n[INFO] Running summarization Loop...\n")

            summary_query = f"Summary focus: {user_input}"
            k = 20

            summary_docs = vectorstore.similarity_search(
                summary_query,
                k=k
            )

            partial_summaries = []
            for doc in summary_docs:
                summary = chunk_summarizer.invoke({
                    "text": doc.page_content[:800]
                })
                partial_summaries.append(summary)

            final_summary = merge_summarizer.invoke({
                "summaries": "\n".join(partial_summaries)
            })

            print("\nBot:", final_summary, "\n")
            continue
        
        if intent == Intent.BOOLEAN_PRESENCE:

            found = boolean_presence_search(docs, user_input)

            if found:
                print("\nBot: Yes — the document contains relevant mentions.\n")
            else:
                print("\nBot: No — the document does not mention this.\n")

            continue
        
        response = run_rag(
            vectorstore=vectorstore,
            llm=llm,
            question=user_input,
            history=history_text
        )

        print("\n🤖:", response, "\n")

        memory.save_context(
            {"input": user_input},
            {"output": response}
        )

# ENTRY

if __name__ == "__main__":
    main()


#  /Users/kuldeeppatel/Documents/LangChain/PDF_Reader/data/AI_Module.pdf