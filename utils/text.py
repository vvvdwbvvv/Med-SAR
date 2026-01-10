import re
from typing import Optional


def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)

    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # collapse non-newline whitespace sequences to single spaces
    s = re.compile(r"[ \t\f\v]+").sub(" ", s)
    # strip trailing spaces on each line
    s = "\n".join(line.rstrip(" ") for line in s.split("\n"))
    return s.strip()
