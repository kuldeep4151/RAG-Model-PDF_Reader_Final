import re

def boolean_presence_search(docs, question: str):
    q = question.lower().strip()

    match = re.match(
        r"\b(is|are|does|do|was|were)\s+"
        r"([a-zA-Z][a-zA-Z0-9\- ]{1,50}?)\s+"
        r"(mentioned|present|used|in)\b",
        q
    )

    if not match:
        return False, None

    entity = match.group(2).strip().lower()

    if len(entity) < 3:
        return False, None

    # Pre-build a single lowercase corpus (much faster)
    full_text = "\n".join(d.page_content for d in docs).lower()

    return (entity in full_text), entity
