from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
import numpy as np
from utils.embeddings import embed_document


class ExplicitVectorStore:
    def __init__(self):
        self.index = None
        self.docs = []   # chunked docs aligned with vectors

    def add_documents(self, docs):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )

        chunked_docs = splitter.split_documents(docs)

        vectors = []
        for doc in chunked_docs:
            v = embed_document(doc.page_content)
            vectors.append(v)
            self.docs.append(doc)

        dim = len(vectors[0])
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(vectors).astype("float32"))

    def similarity_search(self, query_vector, k=10):
        D, I = self.index.search(
            np.array([query_vector]).astype("float32"),
            k
        )
        return [self.docs[i] for i in I[0]]


def build_vector_store(docs):
    store = ExplicitVectorStore()
    store.add_documents(docs)
    return store
