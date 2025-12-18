def compress_docs(
    docs_with_scores,
    max_chars=1000,
    score_threshold=0.75
):
    """
    Selectively builds context from retrieved docs.
    Priority:
    1. High similarity
    2. Shorter chunks first
    3. Hard char limit
    """

    context = ""
    selected_docs = []

    for doc, score in docs_with_scores:
        if score > score_threshold:
            continue

        if len(context) + len(doc.page_content) > max_chars:
            break

        context += doc.page_content + "\n"
        selected_docs.append(doc)

    return context, selected_docs
