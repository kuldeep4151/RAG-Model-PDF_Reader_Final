from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def build_vector_store(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=190
    )

    chunks = splitter.split_documents(docs)
    print("Chunks created:", len(chunks))

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2"
    )

    return FAISS.from_documents(chunks, embeddings)
