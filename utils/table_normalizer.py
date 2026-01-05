import re

def normalize_tables(text: str):

    # Replace multi-spaces with a separator
    text = re.sub(r"[ \t]{2,}", " | ", text)

    # Merge broken lines in tables
    lines = text.split("\n")
    fixed = []

    buffer = ""

    for line in lines:
        if "|" in line:
            # Likely table row
            if buffer:
                fixed.append(buffer)
                buffer = ""
            fixed.append(line)
        else:
            # Broken continuation
            if buffer:
                buffer += " " + line.strip()
            else:
                buffer = line

    if buffer:
        fixed.append(buffer)

    return "\n".join(fixed)
