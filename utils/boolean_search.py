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

    entity = match.group(2).strip()

    if len(entity) < 3:
        return False, None

    entity = entity.lower()

    for doc in docs:
        if entity in doc.page_content.lower():
            return True, entity

    return False, entity
