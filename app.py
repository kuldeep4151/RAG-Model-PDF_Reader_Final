import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
load_dotenv()

from loaders.pdf_loader import load_pdf
from vectorstore.store import build_vector_store
from chains.memory_chain import get_memory_module
from chains.rag_chain import build_rag_chain
from utils.memory_utils import needs_raw_history
from chains.summary_memory import get_summary, update_summary

MAX_CONTEXT_CHARS = 1000


def build_context(docs):
    context = ""
    for d in docs:
        if len(context) + len(d.page_content) > MAX_CONTEXT_CHARS:
            break
        context += d.page_content + "\n"
    return context


def main():
    pdf_path = input("Enter the full path of your PDF file:\n> ")

    if not os.path.isfile(pdf_path):
        print("Error: File not found.")
        return

    print("Loading PDF...")
    docs = load_pdf(pdf_path)

    print("Building Vector store...")
    vectorstore = build_vector_store(docs)

    print("Loading memory...")
    memory = get_memory_module()

    print("Building RAG Chain...")
    rag_chain = build_rag_chain(vectorstore)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    print("Chatbot is ready! Type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break

        # -------------------------------
        # 1️⃣ LOAD HISTORY
        # -------------------------------
        history_msgs = memory.load_memory_variables({})["history"]

        raw_history = "\n".join(
            f"{m.type}: {m.content}"
            for m in history_msgs[-4:]  # last 2 turns only
        )

        summary_history = get_summary()

        # -------------------------------
        # 2️⃣ HYBRID DECISION
        # -------------------------------
        if needs_raw_history(user_input):
            history_text = raw_history
        else:
            history_text = summary_history

        # -------------------------------
        # 3️⃣ RETRIEVAL (NO HISTORY HERE)
        # -------------------------------
        retrieved_docs = retriever.invoke(user_input)
        context = build_context(retrieved_docs)

        # -------------------------------
        # 4️⃣ LLM CALL
        # -------------------------------
        response = rag_chain.invoke({
            "context": context,
            "question": f"{history_text}\n\nUser: {user_input}".strip()
        })

        print("\nBot:", response)

        # -------------------------------
        # 5️⃣ SAVE + UPDATE SUMMARY
        # -------------------------------
        memory.save_context(
            {"input": user_input},
            {"output": response}
        )

        update_summary(memory.load_memory_variables({})["history"])


if __name__ == "__main__":
    main()
