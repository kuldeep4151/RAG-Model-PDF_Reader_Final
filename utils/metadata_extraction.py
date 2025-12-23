import re
from collections import Counter


# -------------------------
# PEOPLE / AUTHORS
# -------------------------
def extract_people(docs, max_pages=2):
    text = "\n".join(doc.page_content for doc in docs[:max_pages])

    name_pattern = re.compile(
        r"\b[A-Z][a-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-z]+)+\b"
    )

    blacklist = {
        "Figure", "Table", "Section", "Model", "Learning"
    }

    people = Counter()

    for match in name_pattern.findall(text):
        if match.split()[0] in blacklist:
            continue
        people[match] += 1

    return [p for p, _ in people.most_common()]


# -------------------------
# ORGANIZATIONS / AFFILIATIONS
# -------------------------
def extract_organizations(docs, max_pages=2):
    org_suffixes = [
        "university", "institute", "laboratory", "lab",
        "college", "school", "research", "corporation",
        "company", "inc", "ltd"
    ]

    text = "\n".join(doc.page_content for doc in docs[:max_pages]).lower()
    orgs = set()

    for line in text.split("\n"):
        if any(s in line for s in org_suffixes):
            orgs.add(line.strip())

    return sorted(orgs)


# -------------------------
# MODELS / SYSTEMS (ROBUST)
# -------------------------
def extract_models_or_systems(docs, min_freq=2):
    """
    Extract model/system-like entities without hard-coding names.
    """

    model_pattern = re.compile(
        r"\b("
        r"[A-Z]{2,}[A-Za-z0-9\-]*|"          # acronyms / ALL CAPS
        r"[A-Z][A-Za-z]+[-][A-Za-z0-9\-]+|"  # hyphenated names
        r"[A-Z][A-Za-z]*\d+[A-Za-z0-9\-]*"   # contains digits
        r")\b"
    )

    context_words = {
        "model", "llm", "architecture", "parameter",
        "training", "trained", "pretrained", "inference"
    }

    candidates = Counter()

    for doc in docs:
        lines = doc.page_content.split("\n")

        for line in lines:
            if not any(w in line.lower() for w in context_words):
                continue

            for match in model_pattern.findall(line):
                if len(match) < 3:
                    continue
                candidates[match] += 1

    return [m for m, c in candidates.items() if c >= min_freq]


# -------------------------
# MAIN ENTRY FOR METADATA
# -------------------------
def run_metadata_extraction(docs, user_query: str):
    """
    General metadata extraction router.
    """

    q = user_query.lower()

    if any(w in q for w in ["author", "authors", "who"]):
        return "people", extract_people(docs)

    if any(w in q for w in ["organization", "institution", "affiliation", "company"]):
        return "organizations", extract_organizations(docs)

    if any(w in q for w in ["model", "models", "llm", "system", "method"]):
        return "models", extract_models_or_systems(docs)

    return "unknown", []
