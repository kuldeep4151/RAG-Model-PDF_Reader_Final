import re


def boolean_presence_search(docs, question: str) -> bool:
    """
    STRICT boolean presence:
    - Extracts the entity being asked about
    - Checks exact / near-exact match
    """

    q = question.lower().strip()

    # Extract phrase after "is / are / does / do"
    match = re.match(
    r"\b(is|are|does|do|was|were)\s+([a-zA-Z][a-zA-Z\- ]{1,50}?)\s+(mentioned|present|used|in)\b",
    q
    )


    if not match:
        return False

    entity = match.group(2).strip()

    if len(entity) < 3:
        return False

    entity = entity.lower()

    for doc in docs:
        if entity in doc.page_content.lower():
            return True

    return False
