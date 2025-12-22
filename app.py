import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
load_dotenv()

from loaders.pdf_loader import load_pdf
from vectorstore.store import build_vector_store
from chains.memory_chain import get_memory_module
from chains.rag_chain import build_rag_chain

from utils.memory_utils import needs_raw_history, is_summary_question
from chains.summary_memory import update_summary
from vectorstore.store import build_vector_store
from chains.hierarchical_summary import chunk_summarizer, merge_summarizer



#  CONFIG 
MAX_CONTEXT_CHARS = 1000

# ---------------- MAIN ----------------
def main():
    pdf_path = input("Enter the full path of your PDF file:\n> ").strip()

    if not os.path.isfile(pdf_path):
        print("Error: File not found.")
        return

    print("\nLoading PDF...")
    docs = load_pdf(pdf_path)

    print("Building vector store...")
    vectorstore = build_vector_store(docs)

    print("Initializing memory...")
    memory = get_memory_module()

    print("Building RAG chain...")
    rag_chain = build_rag_chain(vectorstore)

    print("\nChatbot is ready! Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # --------- LOAD MEMORY ---------
        history_msgs = memory.load_memory_variables({})["history"]

        raw_history = "\n".join(
            f"{msg.type}: {msg.content}"
            for msg in history_msgs[-4:]   # last 2 turns only
        )

        # DEFAULT: no history
        history_text = ""

        # ONLY include history if explicitly required
        if needs_raw_history(user_input):
            history_text = raw_history

        #--------Tried selective_summarize technique ----
        
        if is_summary_question(user_input):
            print("\n[INFO] Running selective summarization...\n")

            # Retrieve top-k important chunks
            k = 20
            docs = vectorstore.similarity_search(
                "summary of the paper",
                k=k
            )

            # Summarize EACH chunk (MAP step)
            partial_summaries = []
            for doc in docs:
                summary = chunk_summarizer.invoke({
                    "text": doc.page_content[:800]   # hard cap
                })
                partial_summaries.append(summary)

            # Merge summaries (REDUCE step)
            final_summary = merge_summarizer.invoke({
                "summaries": "\n".join(partial_summaries)
            })

            print("\nBot:", final_summary)
            continue
        
        else:
            response = rag_chain.invoke(
            {
                "question": user_input,
                "history": history_text
            })

        print("\nBot:", response, "\n")

        # --------- SAVE MEMORY ---------
        memory.save_context(
            {"input": user_input},
            {"output": response}
        )

        update_summary(
            memory.load_memory_variables({})["history"]
        )


# ---------------- ENTRY ----------------
if __name__ == "__main__":
    main()


#  /Users/kuldeeppatel/Documents/LangChain/PDF_Reader/data/AI_Module.pdf