"""
analysis.py
-----------
Pure pandas computations for the AI Dataset Chat (VizBot).

This module ONLY computes facts from the dataframe - it never writes
any user-facing sentences. That job belongs to responses.py. Keeping
"what is true about the data" separate from "how we phrase it" is
what makes this chatbot modular and easy to extend later.

Where equivalent logic already exists elsewhere in the app (outlier
detection, AI readiness scoring), we import and reuse it instead of
duplicating the math.
"""

import pandas as pd

from modules.data_cleaning import detect_outliers_iqr
from modules.ai_readiness import calculate_ai_readiness


def get_shape(df: pd.DataFrame) -> tuple:
    return df.shape


def get_column_types(df: pd.DataFrame) -> dict:
    """Splits columns into numeric / categorical / datetime buckets."""
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    return {"numeric": numeric, "categorical": categorical, "datetime": datetime_cols}


def get_missing_info(df: pd.DataFrame) -> dict:
    """Overall + per-column missing value facts."""
    total_cells = df.shape[0] * df.shape[1]
    per_column = df.isnull().sum()
    total_missing = int(per_column.sum())
    missing_pct = round((total_missing / total_cells) * 100, 2) if total_cells > 0 else 0.0
    columns_with_missing = per_column[per_column > 0].sort_values(ascending=False)

    return {
        "total_missing": total_missing,
        "missing_pct": missing_pct,
        "per_column": per_column,
        "columns_with_missing": columns_with_missing,
    }


def get_duplicate_count(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


def get_memory_usage(df: pd.DataFrame) -> str:
    total_bytes = df.memory_usage(deep=True).sum()
    if total_bytes >= 1_048_576:
        return f"{total_bytes / 1_048_576:.2f} MB"
    return f"{total_bytes / 1024:.2f} KB"


def get_column_dtype_table(df: pd.DataFrame) -> list:
    """Returns a list of (column_name, dtype_string) pairs."""
    return [(col, str(df[col].dtype)) for col in df.columns]


def get_unique_value_counts(df: pd.DataFrame, column: str = None):
    """
    If a column is given, returns its unique-value count (int).
    Otherwise returns a per-column Series of unique counts for every
    column (used when the user doesn't name a specific column).
    """
    if column and column in df.columns:
        return int(df[column].nunique(dropna=True))
    return df.nunique(dropna=True)


def get_most_frequent_value(df: pd.DataFrame, column: str):
    """Returns (value, count) for the most frequent entry in a column,
    or (None, 0) if the column has no non-missing values."""
    series = df[column].dropna()
    if series.empty:
        return None, 0
    mode_value = series.mode().iloc[0]
    count = int((df[column] == mode_value).sum())
    return mode_value, count


def get_column_summary_stats(df: pd.DataFrame, column: str) -> dict:
    """Mean/median/std/min/max for a single numeric column."""
    series = df[column].dropna()
    if series.empty or not pd.api.types.is_numeric_dtype(df[column]):
        return {}
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "min": series.min(),
        "max": series.max(),
    }


def get_overall_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """pandas .describe() for all numeric columns - used when the user
    asks for summary statistics without naming a specific column."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.describe().round(2)


def get_correlation_matrix(df: pd.DataFrame):
    """Returns the correlation matrix of numeric columns, or None if
    there are fewer than 2 numeric columns to correlate."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return None
    return numeric_df.corr()


def get_highly_correlated_pairs(df: pd.DataFrame, threshold: float = 0.7) -> list:
    """
    Returns a list of (col1, col2, correlation) tuples for numeric
    column pairs whose absolute correlation is >= threshold, sorted by
    strength descending. Each pair appears once (no A-B and B-A dupes).
    """
    corr_matrix = get_correlation_matrix(df)
    if corr_matrix is None:
        return []

    pairs = []
    columns = corr_matrix.columns.tolist()
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            value = corr_matrix.iloc[i, j]
            if pd.notna(value) and abs(value) >= threshold:
                pairs.append((columns[i], columns[j], round(float(value), 3)))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs


def get_outlier_summary(df: pd.DataFrame, max_columns: int = 8) -> list:
    """
    Runs IQR-based outlier detection (reusing the same logic as the
    Data Cleaning tab) across numeric columns and returns a list of
    (column, outlier_count) for columns that actually have outliers.
    Capped at `max_columns` numeric columns checked, to keep this fast
    on wide datasets.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()[:max_columns]
    results = []
    for col in numeric_cols:
        outlier_count, _, _, _ = detect_outliers_iqr(df, col)
        if outlier_count:
            results.append((col, outlier_count))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def get_ai_readiness_info(df: pd.DataFrame) -> dict:
    """Reuses the same AI Readiness scoring shown on its own tab, so
    the chatbot's answer always matches what the user sees there."""
    return calculate_ai_readiness(df)
