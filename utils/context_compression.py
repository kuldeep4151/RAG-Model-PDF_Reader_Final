def compress_docs(docs, query=None, max_chars=4000, mode="qa"):
    if mode == "qa":
        # DO NOT compress for QA
        text = "\n\n".join(doc.page_content for doc in docs)
        return text[:max_chars], docs

    # summary mode
    selected = []
    total = 0

    for doc in docs:
        chunk = doc.page_content
        if total + len(chunk) > max_chars:
            break
        selected.append(doc)
        total += len(chunk)

    return "\n\n".join(d.page_content for d in selected), selected
