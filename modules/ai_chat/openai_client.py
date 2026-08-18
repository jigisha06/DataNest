"""
openai_client.py
-----------------
Optional OpenAI integration for VizBot.

Design principle (hybrid architecture): OpenAI is ONLY ever used to
help pick an INTENT + column(s) from the same fixed vocabulary the
offline rule-based classifier uses (modules.ai_chat.intent) - never to
invent the final answer. The actual number/fact still comes from
analysis.py, and the sentence still comes from responses.py. This
means the answer is equally grounded whether it was routed by the
offline rules or by OpenAI.

If no API key is configured, the `openai` package isn't installed, or
the API call fails for ANY reason (no internet, invalid key, rate
limit, malformed response), every function here returns None and the
caller falls back to the offline assistant automatically - the user
should never see an error because of this module.
"""

import os
import json
import pandas as pd

from modules.ai_chat.intent import INTENT_KEYWORDS


def _get_client():
    """
    Tries to build an OpenAI client from the OPENAI_API_KEY
    environment variable. Returns None if unavailable for any reason.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def is_available() -> bool:
    """Small helper the UI can use to show a status hint."""
    return _get_client() is not None


def classify_via_openai(question: str, df: pd.DataFrame):
    """
    Asks OpenAI to pick an intent (from the SAME fixed list the
    offline classifier uses) and, if relevant, a column name - always
    validated against the dataset's real columns before use.

    Returns
    -------
    dict {"intent": str, "column": str|None, "column2": str|None}
    or None if OpenAI is unavailable / the call fails / the response
    can't be trusted.
    """
    client = _get_client()
    if client is None:
        return None

    valid_intents = list(INTENT_KEYWORDS.keys()) + ["unknown"]
    columns_list = df.columns.tolist()
    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    system_prompt = (
        "You are a strict routing assistant for a data-analysis chatbot. Given a user's "
        "question and a dataset's column names, respond with ONLY a JSON object with keys: "
        f"'intent' (must be exactly one of {valid_intents}), "
        "'column' (an EXACT column name from the list, or null), "
        "'column2' (an EXACT column name from the list, or null - only for correlation-between-two-columns). "
        "If the question isn't answerable from this dataset, use intent 'unknown'. "
        "Respond with ONLY the JSON object - no explanation, no markdown fences."
    )
    user_prompt = (
        f"Columns: {columns_list}\n"
        f"Numeric columns: {numeric_columns}\n"
        f"Question: {question}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=150,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw_text = response.choices[0].message.content.strip()
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()

        parsed = json.loads(raw_text)
        intent = parsed.get("intent")
        column = parsed.get("column")
        column2 = parsed.get("column2")

        if intent not in valid_intents:
            return None
        if column is not None and column not in columns_list:
            column = None
        if column2 is not None and column2 not in columns_list:
            column2 = None

        return {"intent": intent, "column": column, "column2": column2}

    except Exception:
        # Any failure - bad key, no internet, rate limit, malformed
        # JSON, etc. - falls back silently. The offline assistant
        # handles everything from here; the user never sees an error.
        return None
