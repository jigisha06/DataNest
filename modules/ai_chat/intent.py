"""
modules/ai_chat/ (package)
---------------------------
Public entry point for VizBot, the AI Dataset Chat assistant.

ARCHITECTURE (deliberately split into 4 focused files):
    intent.py         -> classifies a question into a fixed intent
                         (+ extracts column names), fully offline.
    analysis.py       -> pure pandas computations - the "facts".
    responses.py      -> turns facts into human-readable sentences.
    openai_client.py  -> OPTIONAL: asks OpenAI to pick an intent when
                         the offline classifier can't (still dispatches
                         to the same responses.py functions - OpenAI
                         never invents the final answer).

HYBRID BEHAVIOR (this is the important part):
    1. Try the offline rule-based classifier first (instant, free,
       always available).
    2. If that can't confidently classify the question AND an OpenAI
       API key is configured, ask OpenAI to classify it instead.
    3. If neither can classify it, return a helpful fallback message
       (never a hard error, never a blank response).
    4. Every single step is wrapped so this function CANNOT raise -
       if anything unexpected goes wrong, the user gets a friendly
       message instead of a crash.

The user should never be able to tell whether OpenAI is configured or
not, other than possibly getting a slightly better answer to an
ambiguous question - every question type in the spec works fully
offline.
"""

import pandas as pd

from modules.ai_chat import intent as intent_module
from modules.ai_chat import responses
from modules.ai_chat import openai_client


def is_openai_available() -> bool:
    """Small helper the UI uses to show a status hint."""
    return openai_client.is_available()


# Maps each intent name to the responses.py function that handles it,
# and how many extra arguments (besides `df`) it expects.
_NO_ARG_INTENTS = {
    "overview": responses.overview,
    "row_count": responses.row_count,
    "column_count": responses.column_count,
    "duplicates": responses.duplicates,
    "numeric_columns": responses.numeric_columns,
    "categorical_columns": responses.categorical_columns,
    "memory_usage": responses.memory_usage,
    "list_columns": responses.list_columns,
    "data_types": responses.data_types,
    "explain_columns": responses.explain_columns,
    "missing_per_column": responses.missing_per_column,
    "data_quality": responses.data_quality,
    "ai_ready": responses.ai_ready,
    "target_column": responses.target_column,
    "highly_correlated": responses.highly_correlated,
    "outliers": responses.outliers,
    "recommend_preprocessing": responses.recommend_preprocessing,
    "suggest_visualizations": responses.suggest_visualizations,
    "suggest_ml_models": responses.suggest_ml_models,
    "which_graph": responses.which_graph,
    "explain_histogram": responses.explain_histogram,
    "explain_boxplot": responses.explain_boxplot,
    "explain_scatter": responses.explain_scatter,
}

# Intents that need a single `column` argument.
_COLUMN_ARG_INTENTS = {
    "missing_values": responses.missing_values,
    "unique_values": responses.unique_values,
    "most_frequent": responses.most_frequent,
    "summary_statistics": responses.summary_statistics,
    "max_value": responses.max_value,
    "min_value": responses.min_value,
    "average_value": responses.average_value,
    "sum_value": responses.sum_value,
}

# Intents that need both `column` and `column2`.
_TWO_COLUMN_ARG_INTENTS = {
    "correlation": responses.correlation,
}


def _dispatch(intent_name: str, df: pd.DataFrame, column: str, column2: str) -> str:
    """Routes a classified intent to its response-generating function."""
    if intent_name in _NO_ARG_INTENTS:
        return _NO_ARG_INTENTS[intent_name](df)
    if intent_name in _COLUMN_ARG_INTENTS:
        return _COLUMN_ARG_INTENTS[intent_name](df, column)
    if intent_name in _TWO_COLUMN_ARG_INTENTS:
        return _TWO_COLUMN_ARG_INTENTS[intent_name](df, column, column2)
    return responses.fallback()


def answer_question(question: str, df: pd.DataFrame) -> str:
    """
    Main entry point used by the app. This function is guaranteed to
    NEVER raise - any unexpected error results in a friendly message
    instead of a crash.

    Parameters
    ----------
    question : str                  the user's natural-language question
    df : pandas.DataFrame or None   the currently loaded dataset

    Returns
    -------
    answer : str   always a non-empty string.
    """
    try:
        if df is None:
            return "Please upload and load a dataset first — then I can answer questions about it."

        if not question or not question.strip():
            return "Please type a question about your dataset to get started."

        # ---- Step 1: offline rule-based classification (always available) ----
        result = intent_module.classify(question, df)

        # ---- Step 2: if offline couldn't classify it, try OpenAI (optional) ----
        if result["intent"] == "unknown":
            openai_result = openai_client.classify_via_openai(question, df)
            if openai_result is not None:
                result = openai_result

        # ---- Step 3: dispatch to the matching response generator ----
        if result["intent"] == "unknown":
            return responses.fallback()

        answer = _dispatch(result["intent"], df, result.get("column"), result.get("column2"))

        # Some response functions can legitimately return an empty-ish
        # result in edge cases - make sure we never send back nothing.
        if not answer or not str(answer).strip():
            return responses.fallback()

        return answer

    except Exception as e:
        # Absolute last resort - VizBot should never crash the chat,
        # no matter what went wrong upstream.
        return (
            "I ran into an unexpected issue answering that. Please try rephrasing your "
            "question, or ask something like \"Summarize this dataset\"."
        )


def get_example_questions_markdown() -> str:
    """
    Returns a beginner-friendly, categorized list of example questions
    and the keywords that trigger them - so users know exactly what
    phrasing works, instead of guessing. Displayed in the AI Dataset
    Chat tab (usually inside a collapsed Accordion to save space).

    NOTE: replace <column> with a real column name from YOUR dataset.
    """
    return (
        "**Dataset Overview**\n"
        "- \"Summarize this dataset\" / \"Explain my dataset\" / \"Give me an overview\"\n\n"
        "**Statistics**\n"
        "- \"How many rows/columns are there?\" · \"Missing values\" · \"Duplicate rows\"\n"
        "- \"Numeric columns\" · \"Categorical columns\" · \"Memory usage\"\n\n"
        "**Column Information**\n"
        "- \"List all columns\" · \"Explain each column\" · \"Data types\"\n"
        "- \"Unique values in <column>\" · \"Missing values per column\"\n\n"
        "**Data Quality**\n"
        "- \"Is this dataset clean?\" · \"What problems exist?\" · \"Is it AI ready?\"\n\n"
        "**Correlation & Outliers**\n"
        "- \"Show correlation\" · \"Which columns are highly correlated?\" · \"Does the dataset contain outliers?\"\n\n"
        "**Summary Statistics**\n"
        "- \"What is the average/median/max/min <column>?\" · \"Standard deviation of <column>\"\n\n"
        "**Suggestions**\n"
        "- \"Recommend preprocessing steps\" · \"Suggest visualizations\" · \"Suggest machine learning models\"\n\n"
        "**Visualization Help**\n"
        "- \"Which graph should I use?\" · \"Explain histogram\" · \"Explain box plot\" · \"Explain scatter plot\"\n\n"
        "💡 Tip: use the **exact column names** from your dataset (check the Dashboard or Metadata tab) "
        "for the most reliable answers."
    )
