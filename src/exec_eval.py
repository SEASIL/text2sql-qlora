"""Execute SQL against SQLite databases and compare result sets (execution accuracy)."""
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExecResult:
    ok: bool
    rows: list = field(default_factory=list)
    error: str = ""


def open_readonly(db_path: str | Path) -> sqlite3.Connection:
    uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.text_factory = lambda b: b.decode("utf-8", errors="ignore")
    return conn


def run_query(conn: sqlite3.Connection, sql: str, timeout_s: float = 10.0,
              max_rows: int = 10_000) -> ExecResult:
    deadline = time.monotonic() + timeout_s
    # A non-zero return from the progress handler interrupts the running query.
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cur = conn.execute(sql)
        return ExecResult(ok=True, rows=cur.fetchmany(max_rows))
    except sqlite3.Error as exc:
        return ExecResult(ok=False, error=str(exc))
    finally:
        conn.set_progress_handler(None, 0)


def get_schema(conn: sqlite3.Connection) -> str:
    """CREATE TABLE statements of the database, used as the schema in the prompt."""
    rows = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return "\n".join(r[0].strip() + ";" for r in rows)


def _norm_value(v):
    if isinstance(v, float):
        return repr(round(v, 6))
    return repr(v)


def _norm_row(row) -> tuple:
    # Sorted values per row => column order does not matter (lenient on purpose).
    return tuple(sorted(_norm_value(v) for v in row))


def results_match(gold_rows: list, pred_rows: list, order_matters: bool) -> bool:
    g = [_norm_row(r) for r in gold_rows]
    p = [_norm_row(r) for r in pred_rows]
    if order_matters:
        return g == p
    return Counter(g) == Counter(p)


def gold_has_order_by(gold_sql: str) -> bool:
    return "order by" in gold_sql.lower()
