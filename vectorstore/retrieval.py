def retrieve_relevant_docs(vectorstore, query, k=15, threshold=0.65):
    """
    Returns only high-confidence documents based on similarity score.
    Lower score = more similar.
    """
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)

    return [
        doc for doc, score in docs_with_scores
        if score <= threshold
    ]
