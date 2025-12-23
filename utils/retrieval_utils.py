def retrieve_relevant_docs(vectorstore, query, k=15, threshold=None):
    results = vectorstore.similarity_search_with_score(query, k=k)

    docs = []
    for doc, score in results:
        if threshold is None:
            docs.append(doc)
        else:
            # FAISS-style: lower is better
            if score <= threshold:
                docs.append(doc)

    return docs
