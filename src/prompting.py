"""Shared prompt format.

Training, evaluation and inference MUST all build prompts through this module,
otherwise the model sees a different format at test time than it saw in training.
"""
import re

SYSTEM_PROMPT = (
    "You are a careful SQL assistant. Given a database schema and a question, "
    "write one read-only SQL SELECT query that answers the question. "
    "Return only the SQL, with no explanation."
)


def build_user_message(schema: str, question: str) -> str:
    return f"Schema:\n{schema.strip()}\n\nQuestion: {question.strip()}"


def build_messages(schema: str, question: str, sql: str | None = None) -> list[dict]:
    """Chat messages. Pass `sql` for training examples, omit it for generation."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(schema, question)},
    ]
    if sql is not None:
        messages.append({"role": "assistant", "content": sql.strip()})
    return messages


_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Pull the SQL out of a model reply (handles ```sql fences and a trailing ';')."""
    match = _FENCE.search(text)
    if match:
        text = match.group(1)
    text = text.strip()
    if text.endswith(";"):
        text = text[:-1].rstrip()
    return text
