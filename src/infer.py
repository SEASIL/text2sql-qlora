"""Ask a question about a SQLite database in plain English.

    python -m src.infer --db data/spider/database/concert_singer/concert_singer.sqlite \
        --adapter outputs/qwen2.5-1.5b-text2sql-lora --question "How many singers are there?"

The generated SQL goes through the read-only guard before it is executed on a
read-only connection.
"""
import argparse

from src.evaluate import generate_batch, load_model
from src.exec_eval import get_schema, open_readonly, run_query
from src.prompting import build_messages, extract_sql
from src.sql_guard import check_sql


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="path to a .sqlite file")
    ap.add_argument("--question", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--load-4bit", action="store_true")
    args = ap.parse_args()

    conn = open_readonly(args.db)
    schema = get_schema(conn)
    tok, model = load_model(args.base_model, args.adapter, args.load_4bit)
    prompt = tok.apply_chat_template(build_messages(schema, args.question),
                                     tokenize=False, add_generation_prompt=True)
    sql = extract_sql(generate_batch(tok, model, [prompt], 256)[0])
    print("SQL:", sql)

    guard = check_sql(sql)
    if not guard.allowed:
        print("BLOCKED by read-only guard:", guard.reason)
        return
    res = run_query(conn, sql)
    if not res.ok:
        print("Execution error:", res.error)
        return
    print(f"{len(res.rows)} row(s):")
    for row in res.rows[:20]:
        print(row)


if __name__ == "__main__":
    main()
