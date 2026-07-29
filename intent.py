"""
intent.py
---------
Classifies a user's natural-language question into one of a fixed set
of INTENTS, and (when relevant) figures out which column(s) they mean.

This is NOT exact keyword matching on a single phrase - each intent
has a *group* of trigger phrases covering different ways people ask
the same thing, so "Summarize this dataset", "Explain my dataset",
"Give me a quick overview", and "Tell me about the uploaded file" all
resolve to the same OVERVIEW intent. It's still fully rule-based (no
external ML model, no network call), so it's instant and free, and it
never guesses at the underlying dataframe - column extraction is
always checked against the dataset's real column names.

Matching uses word-boundary regex (not plain substring search) so
short tokens like "min"/"max"/"std" only match whole words - "min"
won't accidentally fire on words like "administrator".

Order matters: more specific intents are listed first so they're
matched before a more generic intent that might share a keyword.
"""

import re
import pandas as pd


# Each intent maps to a list of substrings/phrases that, if found
# anywhere in the (lowercased) question, indicate that intent.
#
# NOTE: word-boundary matching (see _phrase_matches below) means a
# singular phrase like "outlier" will NOT match "outliers" (no boundary
# between "r" and "s"). Every phrase below that could plausibly be
# asked in either singular or plural form is listed BOTH ways - this
# was found the hard way via testing (e.g. "Does the dataset contain
# outliers?" originally fell through to the fallback message).
INTENT_KEYWORDS = {
    # --- Highly specific first ---
    "highly_correlated": ["highly correlated", "most correlated", "strongest correlation"],
    "correlation": ["correlation", "correlate"],
    "outliers": ["outlier", "outliers", "anomaly", "anomalies"],
    "ai_ready": ["ai ready", "ai-ready", "ready for ai", "ready for machine learning", "ml ready"],
    "target_column": ["target column", "best target", "which column should i predict", "prediction target", "label column"],
    "data_quality": ["is this dataset clean", "is the dataset clean", "what problems", "data quality", "any issues", "is it clean"],
    "recommend_preprocessing": ["preprocess", "preprocessing", "pre-process", "pre-processing", "clean up steps", "what should i clean", "how should i clean"],
    "suggest_visualizations": ["suggest a visualization", "suggest visualizations", "which chart", "which charts", "what chart", "what charts", "recommend a chart", "recommend visualizations"],
    "suggest_ml_models": ["which model", "what model", "suggest a model", "suggest models", "suggest machine learning", "machine learning model", "ml model", "ml models", "predict"],
    "explain_histogram": ["what is a histogram", "explain histogram", "histogram mean"],
    "explain_boxplot": ["what is a box plot", "explain box plot", "explain boxplot", "box plot mean"],
    "explain_scatter": ["what is a scatter plot", "explain scatter", "scatter plot mean"],
    "which_graph": ["which graph", "which graphs", "what graph", "what graphs", "which visualization should i use"],

    "missing_per_column": ["missing values per column", "which column", "which columns", "columns with missing"],
    "duplicates": ["duplicate", "duplicates"],
    "memory_usage": ["memory usage", "memory size", "how much memory", "file size"],
    "numeric_columns": ["numeric column", "numeric columns", "numerical column", "numerical columns", "number columns"],
    "categorical_columns": ["categorical column", "categorical columns", "text column", "text columns", "string column", "string columns"],
    "list_columns": ["list all columns", "list columns", "what columns", "column names", "show me the columns"],
    "data_types": ["data type", "data types", "dtype", "column type", "column types"],
    "unique_values": ["unique value", "unique values", "how many unique", "distinct value", "distinct values"],
    "most_frequent": ["most common", "most frequent", "mode of"],

    "row_count": ["how many rows", "number of rows", "row count", "total rows"],
    "column_count": ["how many columns", "number of columns", "column count", "total columns", "how many features"],

    "summary_statistics": ["mean", "median", "standard deviation", "std dev", "std", "summary statistic", "describe the numbers"],
    "max_value": ["maximum", "highest", "largest", "max"],
    "min_value": ["minimum", "lowest", "smallest", "min"],
    "average_value": ["average", "averages", "typical value", "avg"],
    "sum_value": ["sum of", "total of"],

    "missing_values": ["missing value", "missing values", "missing data", "null value", "null values", "nulls", "na values"],

    "explain_columns": ["explain each column", "explain the columns", "describe each column", "describe the columns"],

    # --- Broad overview last, since many words above ("describe",
    #     "explain") could otherwise shadow it if it were checked first ---
    "overview": [
        "summarize", "summary", "overview", "explain this dataset", "explain my dataset",
        "describe the dataset", "describe this dataset", "describe the data", "tell me about",
        "what is this dataset", "quick overview", "give me an overview", "about the uploaded file",
        "about this data",
    ],
}


def _phrase_matches(phrase: str, question_lower: str) -> bool:
    """
    Word-boundary match: short tokens like "min", "max", "std", "avg"
    only match whole words, so "min" doesn't fire on "administrator".
    Multi-word phrases work the same way, just bounded at both ends.
    """
    pattern = r"\b" + re.escape(phrase) + r"\b"
    return re.search(pattern, question_lower) is not None


def _find_matching_column(question: str, columns) -> str:
    """
    Finds the column name the user is most likely referring to, by
    looking for it (case-insensitively) inside the question text.
    Longer column names are checked first so "customer_id" matches
    before a shorter column simply named "id".
    """
    question_lower = question.lower()
    for col in sorted(columns, key=len, reverse=True):
        if col.lower() in question_lower:
            return col
    return None


def _find_two_matching_columns(question: str, columns) -> tuple:
    """Finds up to two distinct column names mentioned in the question
    (used for correlation-between-two-columns queries)."""
    question_lower = question.lower()
    matched = [c for c in sorted(columns, key=len, reverse=True) if c.lower() in question_lower]
    # De-duplicate while preserving order, since a short column name
    # could accidentally also be a substring of a longer matched one.
    seen = set()
    unique_matched = []
    for c in matched:
        if c not in seen:
            unique_matched.append(c)
            seen.add(c)
    if len(unique_matched) >= 2:
        return unique_matched[0], unique_matched[1]
    return None, None


def classify(question: str, df: pd.DataFrame) -> dict:
    """
    Classifies a question into an intent and extracts any relevant
    column name(s) mentioned.

    Returns
    -------
    dict with keys: "intent" (str, "unknown" if nothing matched),
                    "column" (str or None), "column2" (str or None)
    """
    q = question.lower().strip()
    all_columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    for intent, phrases in INTENT_KEYWORDS.items():
        if any(_phrase_matches(phrase, q) for phrase in phrases):
            column, column2 = None, None

            if intent == "correlation":
                column, column2 = _find_two_matching_columns(question, numeric_columns)
            elif intent in ("average_value", "sum_value", "max_value", "min_value", "summary_statistics"):
                column = _find_matching_column(question, numeric_columns)
            elif intent in ("unique_values", "most_frequent", "missing_values"):
                column = _find_matching_column(question, all_columns)

            return {"intent": intent, "column": column, "column2": column2}

    return {"intent": "unknown", "column": None, "column2": None}
