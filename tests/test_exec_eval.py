import sqlite3
import pytest
from src.exec_eval import (get_schema, gold_has_order_by, open_readonly,
                           results_match, run_query)


@pytest.fixture()
def db_path(tmp_path):
    path = tmp_path / "toy.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE singer (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
    conn.executemany("INSERT INTO singer VALUES (?,?,?)",
                     [(1, "Ann", 30), (2, "Bob", 25), (3, "Cy", 41)])
    conn.commit()
    conn.close()
    return path


def test_schema_extraction(db_path):
    schema = get_schema(open_readonly(db_path))
    assert "CREATE TABLE singer" in schema


def test_run_query_and_match_ignores_column_and_row_order(db_path):
    conn = open_readonly(db_path)
    gold = run_query(conn, "SELECT name, age FROM singer")
    pred = run_query(conn, "SELECT age, name FROM singer ORDER BY age")
    assert gold.ok and pred.ok
    assert results_match(gold.rows, pred.rows, order_matters=False)
    assert not results_match(gold.rows, pred.rows, order_matters=True)


def test_wrong_result_detected(db_path):
    conn = open_readonly(db_path)
    gold = run_query(conn, "SELECT name FROM singer WHERE age > 28")
    pred = run_query(conn, "SELECT name FROM singer WHERE age > 40")
    assert not results_match(gold.rows, pred.rows, order_matters=False)


def test_invalid_sql_returns_error_not_exception(db_path):
    res = run_query(open_readonly(db_path), "SELECT nope FROM singer")
    assert not res.ok and res.error


def test_connection_is_read_only(db_path):
    res = run_query(open_readonly(db_path), "DELETE FROM singer")
    assert not res.ok


def test_timeout_interrupts_runaway_query(db_path):
    conn = open_readonly(db_path)
    sql = ("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) "
           "SELECT count(*) FROM c")
    res = run_query(conn, sql, timeout_s=0.5)
    assert not res.ok


def test_order_by_detection():
    assert gold_has_order_by("SELECT a FROM t ORDER BY a")
    assert not gold_has_order_by("SELECT a FROM t")
