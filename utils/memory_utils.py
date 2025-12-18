TRIGGERS = [
    "as mentioned",
    "earlier",
    "before",
    "you said",
    "based on",
    "continue",
    "why",
    "explain in detail",
    "then"
]

def needs_raw_history(user_input: str) -> bool:
    return any(trigger in user_input.lower() for trigger in TRIGGERS)


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
