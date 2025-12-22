def compress_docs(docs_with_scores, query: str, max_chars=700):
    """
    Score-aware, rank-based compression.
    Works for:
    - QA
    - Enumeration
    - Conceptual questions
    """
    context = ""
    selected = []

    q_tokens = set(
        t for t in query.lower().split()
        if len(t) > 2
    )

    for idx, (doc, score) in enumerate(docs_with_scores):

        # Keep top-ranked chunks first (FAISS already sorted)
        if len(context) + len(doc.page_content) > max_chars:
            break

        text = doc.page_content.lower()
        lexical_overlap = any(t in text for t in q_tokens)

        # Selection logic:
        # 1. Always keep first chunk
        # 2. Keep chunks with lexical overlap
        if idx == 0 or lexical_overlap:
            context += doc.page_content + "\n"
            selected.append(doc)

    return context.strip(), selected
