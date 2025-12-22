# utils/retrieval_utils.py

def get_dynamic_k(question: str) -> int:
    q = question.lower().strip()

    # 1. Precise yes/no or fact lookup
    if q.startswith((
        "is ", "are ", "was ", "were ",
        "does ", "do ", "did ",
        "has ", "have "
    )):
        return 10

    # 2. Enumerative / listing intent
    if any(p in q for p in [
        "list all",
        "list the",
        "enumerate",
        "all the",
        "names of",
        "which models",
        "which llms",
        "what models"
    ]):
        return 10

    # 3. Conceptual / descriptive
    if any(p in q for p in [
        "what is",
        "what are",
        "explain",
        "describe",
        "purpose",
        "how does",
        "why does"
    ]):
        return 9

    # 4. Safe default
    return 10
