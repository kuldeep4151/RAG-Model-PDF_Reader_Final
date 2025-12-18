import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
load_dotenv()

from loaders.pdf_loader import load_pdf
from vectorstore.store import build_vector_store
from chains.memory_chain import get_memory_module
from chains.rag_chain import build_rag_chain

from utils.memory_utils import needs_raw_history, is_summary_question
from chains.summary_memory import get_summary, update_summary
from utils.context_compression import compress_docs
from vectorstore.store import build_vector_store




# ---------------- CONFIG ----------------
MAX_CONTEXT_CHARS = 1000
SCORE_THRESHOLD = 0.75


# ---------------- HELPERS ----------------
def get_dynamic_k(question: str) -> int:
    q = question.lower()

    if any(word in q for word in ["list", "all", "models", "names", "mentioned"]):
        return 10  # high recall
    elif any(word in q for word in ["is", "does", "are", "was"]):
        return 4   # precise
    else:
        return 6   # default


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
            for msg in history_msgs[-4:]
        )

        summary_history = get_summary()

        history_text = (
            raw_history if needs_raw_history(user_input)
            else summary_history
        )

        # --------- ROUTED RETRIEVAL ---------
        if is_summary_question(user_input):
            # VERY small context, no history
            docs = vectorstore.similarity_search(user_input, k=2)
            context = docs[0].page_content[:600]
            history_text = ""

        else:
            k = get_dynamic_k(user_input)

            docs_with_scores = vectorstore.similarity_search_with_score(
                user_input,
                k=k
            )

            context, _ = compress_docs(
                docs_with_scores,
                max_chars=MAX_CONTEXT_CHARS,
                score_threshold=SCORE_THRESHOLD
            )

        # --------- DEBUG (optional) ---------
        # print(f"[DEBUG] Context chars: {len(context)} | k={k if not is_summary_question(user_input) else 2}")

        # --------- LLM CALL ---------
        response = rag_chain.invoke({
            "context": context,
            "history": history_text,
            "question": user_input
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