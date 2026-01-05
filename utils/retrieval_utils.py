from utils.embeddings import embed_query
import numpy as np
def retrieve_relevant_docs(vectorstore, query, k=15, threshold=None):
    # Explicitly embed query
    query_vector = embed_query(query)

    #  FAISS similarity search
    distances, indices = vectorstore.index.search(
        np.array([query_vector]).astype("float32"),
        k
    )

    docs = []

    for i, score in zip(indices[0], distances[0]):
        if threshold is None or score <= threshold:
            docs.append(vectorstore.docs[i])

    return docs
