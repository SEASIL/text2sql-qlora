"""Read-only SQL guard.

Generated SQL is untrusted input. Before running anything we allow ONLY a single
SELECT (or WITH ... SELECT) statement and reject anything that could modify data
or the database. This is defence in depth: the database connection should also be
opened read-only (see exec_eval.open_readonly).
"""
import re
from dataclasses import dataclass

FORBIDDEN_KEYWORDS = {
    "insert", "update", "delete", "drop", "alter", "create", "truncate",
    "attach", "detach", "pragma", "vacuum", "reindex", "grant", "revoke",
    "merge", "call", "exec", "execute", "load_extension", "into",
}


@dataclass
class GuardResult:
    allowed: bool
    reason: str = ""


class _Unterminated(Exception):
    pass


def _strip_literals_and_comments(sql: str) -> str:
    """Replace comments and quoted literals/identifiers with spaces so that
    keyword scanning cannot be fooled by (or false-positive on) their contents."""
    out: list[str] = []
    i, n = 0, len(sql)
    while i < n:
        c = sql[i]
        if sql.startswith("--", i):
            j = sql.find("\n", i)
            i = n if j == -1 else j
            out.append(" ")
        elif sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            if j == -1:
                raise _Unterminated("unterminated block comment")
            i = j + 2
            out.append(" ")
        elif c in ("'", '"', "`"):
            j = i + 1
            while True:
                j = sql.find(c, j)
                if j == -1:
                    raise _Unterminated("unterminated quoted string")
                if j + 1 < n and sql[j + 1] == c:  # doubled quote = escaped quote
                    j += 2
                    continue
                break
            i = j + 1
            out.append(" ")
        elif c == "[":
            j = sql.find("]", i + 1)
            if j == -1:
                raise _Unterminated("unterminated bracket identifier")
            i = j + 1
            out.append(" ")
        else:
            out.append(c)
            i += 1
    return "".join(out)


def check_sql(sql: str) -> GuardResult:
    if not sql or not sql.strip():
        return GuardResult(False, "empty query")
    try:
        stripped = _strip_literals_and_comments(sql)
    except _Unterminated as exc:
        return GuardResult(False, str(exc))

    body = stripped.strip()
    if body.endswith(";"):
        body = body[:-1].rstrip()
    if ";" in body:
        return GuardResult(False, "multiple statements are not allowed")

    tokens = re.findall(r"[a-z_]+", body.lower())
    if not tokens:
        return GuardResult(False, "no SQL tokens found")
    if tokens[0] not in ("select", "with"):
        return GuardResult(False, f"query must start with SELECT or WITH, got '{tokens[0]}'")
    bad = sorted(set(tokens) & FORBIDDEN_KEYWORDS)
    if bad:
        return GuardResult(False, f"forbidden keyword(s): {', '.join(bad)}")
    return GuardResult(True)
