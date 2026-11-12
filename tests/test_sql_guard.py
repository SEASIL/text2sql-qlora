import pytest
from src.sql_guard import check_sql


@pytest.mark.parametrize("sql", [
    "SELECT * FROM users",
    "select name from t where id = 1;",
    "WITH a AS (SELECT 1) SELECT * FROM a",
    "SELECT created_at, updated_by FROM logs",          # column names containing keywords
    "SELECT 'drop table x' AS note FROM t",             # keyword only inside a string literal
    "SELECT \"update\" FROM t",                         # quoted identifier
    "SELECT replace(name, 'a', 'b') FROM t",            # REPLACE() function is fine
    "SELECT a FROM t -- delete everything",             # keyword only in a comment
])
def test_allows_safe_queries(sql):
    assert check_sql(sql).allowed, check_sql(sql).reason


@pytest.mark.parametrize("sql", [
    "",
    "   ",
    "DROP TABLE users",
    "DELETE FROM users",
    "UPDATE users SET name = 'x'",
    "INSERT INTO users VALUES (1)",
    "SELECT * FROM users; DROP TABLE users",             # stacked statement
    "SELECT 1; SELECT 2",
    "SELECT * INTO backup FROM users",
    "WITH x AS (DELETE FROM t RETURNING *) SELECT * FROM x",
    "PRAGMA table_info(users)",
    "ATTACH DATABASE 'x.db' AS x",
    "SELECT 'unterminated FROM t",
    "SELECT 1 /* unterminated",
    "CREATE TABLE t (a int)",
])
def test_blocks_unsafe_queries(sql):
    assert not check_sql(sql).allowed
