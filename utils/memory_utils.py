def needs_raw_history(user_input: str) -> bool:
    q = user_input.lower()

    triggers = [
        "as mentioned",
        "as discussed",
        "earlier",
        "before",
        "previous",
        "that",
        "those",
        "them",
        "this model",
        "these models",
        "from above",
        "you said",
        "we discussed"
    ]

    return any(t in q for t in triggers)



def is_entity_query(query: str) -> bool:
    keywords = [
        "mentioned", "contain", "contains",
        "is there", "does the paper",
        "name", "list", "organization", "company"
    ]
    return any(k in query.lower() for k in keywords)

def is_summary_question(q: str) -> bool:
    q = q.lower()
    return any(phrase in q for phrase in [
        "what is this paper about",
        "summary",
        "overview",
        "describe the paper",
        "what does the paper discuss"
    ])
