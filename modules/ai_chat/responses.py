"""
responses.py
-------------
Turns the raw facts computed by analysis.py into natural, readable
sentences - e.g. "This dataset contains 2,350 rows and 14 columns.
It has only 2% missing values and no duplicate records, indicating
good data quality." instead of a bare "Rows: 2350 / Columns: 14".

Each function here corresponds to one intent from intent.py. This
separation means: if you want to change HOW something is phrased,
you only ever touch this file - the underlying math in analysis.py
never needs to change.
"""

import pandas as pd
from modules.ai_chat import analysis


def _format_number(n) -> str:
    """1234567 -> '1,234,567' - small touch, but reads much better."""
    try:
        return f"{n:,}"
    except (ValueError, TypeError):
        return str(n)


# ----------------------------------------------------------------------
# OVERVIEW / SUMMARY
# ----------------------------------------------------------------------

def overview(df: pd.DataFrame) -> str:
    rows, cols = analysis.get_shape(df)
    missing_info = analysis.get_missing_info(df)
    dup_count = analysis.get_duplicate_count(df)
    col_types = analysis.get_column_types(df)
    numeric_count = len(col_types["numeric"])
    categorical_count = len(col_types["categorical"])

    missing_pct = missing_info["missing_pct"]
    total_missing = missing_info["total_missing"]

    # ---- Quality judgement, in plain English ----
    if missing_pct == 0 and dup_count == 0:
        quality_phrase = "no missing values and no duplicate records, indicating excellent data quality"
    elif missing_pct < 5 and dup_count == 0:
        quality_phrase = f"only {missing_pct}% missing values and no duplicate records, indicating good data quality"
    elif missing_pct < 15:
        quality_phrase = (
            f"{missing_pct}% missing values and {_format_number(dup_count)} duplicate record(s), "
            "which is manageable but worth cleaning up"
        )
    else:
        quality_phrase = (
            f"a notable {missing_pct}% missing values and {_format_number(dup_count)} duplicate record(s), "
            "suggesting this dataset needs some cleaning before analysis"
        )

    # ---- Feature-mix sentence ----
    if numeric_count == 0:
        mix_phrase = "The dataset is entirely text/categorical, so it will need encoding before most machine learning models can use it."
    elif categorical_count == 0:
        mix_phrase = "Every column is numeric, making this dataset well suited to statistical analysis and predictive modeling."
    elif numeric_count >= categorical_count:
        mix_phrase = f"Most columns are numeric ({numeric_count} numeric vs {categorical_count} categorical), making the dataset suitable for predictive analytics."
    else:
        mix_phrase = f"Most columns are categorical ({categorical_count} categorical vs {numeric_count} numeric), so encoding will be an important preprocessing step."

    return (
        f"This dataset contains **{_format_number(rows)} rows** and **{cols} columns**. "
        f"It has {quality_phrase}. {mix_phrase}"
    )


# ----------------------------------------------------------------------
# BASIC STATS
# ----------------------------------------------------------------------

def row_count(df: pd.DataFrame) -> str:
    rows, _ = analysis.get_shape(df)
    return f"This dataset has **{_format_number(rows)} rows**."


def column_count(df: pd.DataFrame) -> str:
    _, cols = analysis.get_shape(df)
    return f"This dataset has **{cols} columns**."


def missing_values(df: pd.DataFrame, column: str = None) -> str:
    if column:
        count = int(df[column].isnull().sum())
        pct = round((count / len(df)) * 100, 1) if len(df) else 0
        if count == 0:
            return f"**{column}** has no missing values - it's complete."
        return f"**{column}** has **{count}** missing value(s) ({pct}% of rows)."

    info = analysis.get_missing_info(df)
    if info["total_missing"] == 0:
        return "This dataset has **no missing values** at all - it's complete. ✅"
    return (
        f"There are **{_format_number(info['total_missing'])}** missing values in total, "
        f"about **{info['missing_pct']}%** of all cells."
    )


def missing_per_column(df: pd.DataFrame) -> str:
    info = analysis.get_missing_info(df)
    columns_with_missing = info["columns_with_missing"]

    if columns_with_missing.empty:
        return "No columns have missing values - this dataset is complete. ✅"

    lines = [f"- **{col}**: {int(count)} missing" for col, count in columns_with_missing.items()]
    top_col = columns_with_missing.index[0]
    top_count = int(columns_with_missing.iloc[0])

    return (
        f"**{len(columns_with_missing)}** column(s) have missing values. "
        f"**{top_col}** has the most, with **{top_count}** missing entries:\n\n" + "\n".join(lines[:10])
    )


def duplicates(df: pd.DataFrame) -> str:
    count = analysis.get_duplicate_count(df)
    if count == 0:
        return "There are **no duplicate rows** in this dataset. ✅"
    pct = round((count / len(df)) * 100, 1) if len(df) else 0
    return f"There are **{_format_number(count)}** duplicate row(s), about **{pct}%** of the dataset."


def numeric_columns(df: pd.DataFrame) -> str:
    cols = analysis.get_column_types(df)["numeric"]
    if not cols:
        return "This dataset has **no numeric columns**."
    return f"There are **{len(cols)} numeric column(s)**: {', '.join(cols)}."


def categorical_columns(df: pd.DataFrame) -> str:
    cols = analysis.get_column_types(df)["categorical"]
    if not cols:
        return "This dataset has **no categorical (text) columns**."
    return f"There are **{len(cols)} categorical column(s)**: {', '.join(cols)}."


def memory_usage(df: pd.DataFrame) -> str:
    return f"This dataset is using **{analysis.get_memory_usage(df)}** of memory."


def list_columns(df: pd.DataFrame) -> str:
    cols = df.columns.tolist()
    return f"This dataset has **{len(cols)} columns**:\n\n" + ", ".join(f"`{c}`" for c in cols)


def data_types(df: pd.DataFrame) -> str:
    dtype_pairs = analysis.get_column_dtype_table(df)
    lines = [f"- **{col}**: {dtype}" for col, dtype in dtype_pairs]
    return "Here are the data types for each column:\n\n" + "\n".join(lines)


def explain_columns(df: pd.DataFrame) -> str:
    """A slightly richer per-column breakdown than plain dtypes -
    includes missing count and unique count too."""
    lines = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing = int(df[col].isnull().sum())
        unique = int(df[col].nunique(dropna=True))
        lines.append(f"- **{col}** ({dtype}) — {unique} unique value(s), {missing} missing")
    return "Here's a breakdown of every column:\n\n" + "\n".join(lines)


def unique_values(df: pd.DataFrame, column: str = None) -> str:
    if column:
        count = analysis.get_unique_value_counts(df, column)
        return f"**{column}** has **{count}** unique value(s)."

    counts = analysis.get_unique_value_counts(df)
    lines = [f"- **{col}**: {int(count)} unique" for col, count in counts.items()]
    return "Unique value counts per column:\n\n" + "\n".join(lines[:12])


def most_frequent(df: pd.DataFrame, column: str = None) -> str:
    if not column:
        return "Please mention a column name so I can tell you its most frequent value."
    value, count = analysis.get_most_frequent_value(df, column)
    if value is None:
        return f"**{column}** has no non-missing values to analyze."
    return f"The most frequent value in **{column}** is **{value}**, appearing **{count}** time(s)."


# ----------------------------------------------------------------------
# DATA QUALITY / AI READINESS
# ----------------------------------------------------------------------

def data_quality(df: pd.DataFrame) -> str:
    missing_info = analysis.get_missing_info(df)
    dup_count = analysis.get_duplicate_count(df)
    outlier_summary = analysis.get_outlier_summary(df)

    issues = []
    if missing_info["total_missing"] > 0:
        issues.append(f"{missing_info['missing_pct']}% missing values")
    if dup_count > 0:
        issues.append(f"{_format_number(dup_count)} duplicate row(s)")
    if outlier_summary:
        top_outlier_col, top_outlier_count = outlier_summary[0]
        issues.append(f"outliers detected in **{top_outlier_col}** ({top_outlier_count} rows) and possibly other columns")

    if not issues:
        return "This dataset looks **clean** — no missing values, no duplicates, and no major outliers detected. ✅"

    return (
        "This dataset has a few things worth addressing: " + "; ".join(issues) + ". "
        "Check the **Data Cleaning** tab to fix these in a couple of clicks."
    )


def ai_ready(df: pd.DataFrame) -> str:
    result = analysis.get_ai_readiness_info(df)
    return (
        f"This dataset scores **{result['score']}/100** on AI Readiness, rated **{result['status']}**. "
        f"(Breakdown: -{result['missing_penalty']} for missing values, "
        f"-{result['duplicate_penalty']} for duplicates, -{result['feature_penalty']} for feature-type balance. "
        "See the **AI Workspace** tab for the full picture.)"
    )


def target_column(df: pd.DataFrame) -> str:
    """
    Suggests a plausible prediction target column using a simple,
    transparent heuristic: columns with a small number of unique
    values (2-10) that AREN'T identifier-like columns are common
    classification labels. This is NOT a guarantee - it's a starting
    point based on cardinality, not an understanding of your actual
    business goal.
    """
    total_rows = len(df)
    id_candidates = [
        col for col in df.columns
        if total_rows > 0 and df[col].nunique(dropna=True) == total_rows
    ]
    candidates = [
        col for col in df.columns
        if col not in id_candidates and 2 <= df[col].nunique(dropna=True) <= 10
    ]

    if not candidates:
        return (
            "I couldn't confidently identify a likely target column — most columns either have too "
            "many unique values or look like identifiers. Consider your actual prediction goal to "
            "choose the target manually."
        )

    best = candidates[-1]
    unique_count = df[best].nunique(dropna=True)
    return (
        f"**{best}** looks like a plausible target column — it has only **{unique_count}** unique "
        "value(s), which is typical for a classification label. (This is a heuristic based on "
        "cardinality, not a guarantee — confirm it matches your actual prediction goal.)"
    )


# ----------------------------------------------------------------------
# CORRELATION / OUTLIERS
# ----------------------------------------------------------------------

def correlation(df: pd.DataFrame, column: str = None, column2: str = None) -> str:
    if column and column2:
        value = df[column].corr(df[column2])
        return f"The correlation between **{column}** and **{column2}** is **{value:.3f}**."

    pairs = analysis.get_highly_correlated_pairs(df, threshold=0.5)
    if not pairs:
        matrix = analysis.get_correlation_matrix(df)
        if matrix is None:
            return "This dataset needs at least 2 numeric columns to compute correlation."
        return "No strongly correlated column pairs were found (all correlations are below 0.5)."

    top = pairs[:5]
    lines = [f"- **{c1}** ↔ **{c2}**: {corr:+.3f}" for c1, c2, corr in top]
    return "Here are the most correlated numeric column pairs:\n\n" + "\n".join(lines)


def highly_correlated(df: pd.DataFrame) -> str:
    pairs = analysis.get_highly_correlated_pairs(df, threshold=0.7)
    if not pairs:
        return "No column pairs are highly correlated (using a 0.7 threshold) - your features look fairly independent."
    lines = [f"- **{c1}** ↔ **{c2}**: {corr:+.3f}" for c1, c2, corr in pairs[:8]]
    return f"Found **{len(pairs)}** highly correlated pair(s) (|correlation| ≥ 0.7):\n\n" + "\n".join(lines)


def outliers(df: pd.DataFrame) -> str:
    summary = analysis.get_outlier_summary(df)
    if not summary:
        return "No significant outliers were detected in the numeric columns (using the 1.5×IQR rule). ✅"
    lines = [f"- **{col}**: {count} outlier(s)" for col, count in summary]
    return (
        f"Outliers were detected in **{len(summary)} numeric column(s)**:\n\n" + "\n".join(lines) +
        "\n\nYou can review and remove these in the **Data Cleaning** tab."
    )


# ----------------------------------------------------------------------
# SUMMARY STATISTICS (mean/median/std/min/max)
# ----------------------------------------------------------------------

def summary_statistics(df: pd.DataFrame, column: str = None) -> str:
    if column:
        stats = analysis.get_column_summary_stats(df, column)
        if not stats:
            return f"**{column}** doesn't have numeric data to summarize."
        return (
            f"Summary statistics for **{column}**: "
            f"mean = {stats['mean']:.2f}, median = {stats['median']:.2f}, "
            f"std = {stats['std']:.2f}, min = {stats['min']:.2f}, max = {stats['max']:.2f}."
        )

    described = analysis.get_overall_numeric_summary(df)
    if described.empty:
        return "There are no numeric columns to compute summary statistics for."

    lines = []
    for col in described.columns[:6]:
        lines.append(
            f"- **{col}**: mean {described.loc['mean', col]}, median {described.loc['50%', col]}, "
            f"std {described.loc['std', col]}, min {described.loc['min', col]}, max {described.loc['max', col]}"
        )
    return "Summary statistics for the numeric columns:\n\n" + "\n".join(lines)


def max_value(df: pd.DataFrame, column: str = None) -> str:
    if not column:
        return "Please mention a numeric column name so I can find its maximum."
    return f"The maximum **{column}** is **{df[column].max():.2f}**."


def min_value(df: pd.DataFrame, column: str = None) -> str:
    if not column:
        return "Please mention a numeric column name so I can find its minimum."
    return f"The minimum **{column}** is **{df[column].min():.2f}**."


def average_value(df: pd.DataFrame, column: str = None) -> str:
    if not column:
        return "Please mention a numeric column name so I can calculate the average."
    return f"The average **{column}** is **{df[column].mean():.2f}**."


def sum_value(df: pd.DataFrame, column: str = None) -> str:
    if not column:
        return "Please mention a numeric column name so I can calculate the total."
    return f"The total **{column}** is **{df[column].sum():.2f}**."


# ----------------------------------------------------------------------
# SUGGESTIONS (preprocessing / visualizations / ML models)
# ----------------------------------------------------------------------

def recommend_preprocessing(df: pd.DataFrame) -> str:
    missing_info = analysis.get_missing_info(df)
    dup_count = analysis.get_duplicate_count(df)
    col_types = analysis.get_column_types(df)
    outlier_summary = analysis.get_outlier_summary(df)

    suggestions = []
    if missing_info["total_missing"] > 0:
        suggestions.append("Handle missing values (drop rows, or fill numeric columns with mean/median and categorical columns with the most frequent value) — the **Data Cleaning** tab does this in one click.")
    if dup_count > 0:
        suggestions.append(f"Remove the **{dup_count}** duplicate row(s) to avoid biasing any analysis.")
    if outlier_summary:
        suggestions.append(f"Review outliers in **{outlier_summary[0][0]}** (and possibly other numeric columns) - they can skew averages and model training.")
    if col_types["categorical"]:
        suggestions.append(f"Encode the categorical column(s) ({', '.join(col_types['categorical'][:5])}) before feeding this into most ML models (e.g. one-hot or label encoding).")
    if not suggestions:
        return "This dataset is already in good shape — no missing values, duplicates, or major outliers detected. You're ready to move on to analysis or modeling."

    return "Here's what I'd recommend before further analysis:\n\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(suggestions))


def suggest_visualizations(df: pd.DataFrame) -> str:
    col_types = analysis.get_column_types(df)
    numeric_count = len(col_types["numeric"])
    categorical_count = len(col_types["categorical"])

    suggestions = []
    if numeric_count >= 1:
        suggestions.append("**Histogram** — to see the distribution of a numeric column.")
    if categorical_count >= 1 and numeric_count >= 1:
        suggestions.append("**Bar Chart** — to compare a numeric metric across categories.")
    if numeric_count >= 2:
        suggestions.append("**Scatter Plot** — to explore the relationship between two numeric columns.")
        suggestions.append("**Correlation Heatmap** — to spot which numeric columns move together.")
    if numeric_count >= 1:
        suggestions.append("**Box Plot** — to check for spread and outliers in a numeric column.")

    if not suggestions:
        return "With mostly text columns, a Bar Chart of value counts is your best option here."

    return "Based on your dataset's columns, try these in the **Visualization** tab:\n\n" + "\n".join(f"- {s}" for s in suggestions)


def suggest_ml_models(df: pd.DataFrame) -> str:
    col_types = analysis.get_column_types(df)
    numeric_count = len(col_types["numeric"])
    categorical_count = len(col_types["categorical"])

    return (
        "I can't know your exact target column, but here's general guidance based on this data:\n\n"
        f"- If you're predicting a **numeric value** (regression): Linear Regression, Random Forest Regressor, or Gradient Boosting (XGBoost/LightGBM).\n"
        f"- If you're predicting a **category** (classification): Logistic Regression, Random Forest Classifier, or Gradient Boosting.\n"
        f"- With {numeric_count} numeric and {categorical_count} categorical column(s), tree-based models "
        "(Random Forest, XGBoost) usually work well since they handle mixed data types gracefully with minimal preprocessing."
    )


# ----------------------------------------------------------------------
# VISUALIZATION EDUCATION (static explanations, not data-dependent)
# ----------------------------------------------------------------------

def which_graph(df: pd.DataFrame) -> str:
    return suggest_visualizations(df)


def explain_histogram(df: pd.DataFrame = None) -> str:
    return (
        "A **histogram** shows how values in a single numeric column are distributed — it groups values into "
        "bins and shows how many rows fall into each bin. Use it to spot skew, spread, and common ranges."
    )


def explain_boxplot(df: pd.DataFrame = None) -> str:
    return (
        "A **box plot** summarizes a numeric column's spread using its median, quartiles, and potential outliers "
        "(shown as points beyond the 'whiskers'). It's great for comparing spread and spotting anomalies at a glance."
    )


def explain_scatter(df: pd.DataFrame = None) -> str:
    return (
        "A **scatter plot** places two numeric columns on X and Y axes to reveal relationships between them — "
        "useful for spotting trends, clusters, or correlation between two variables."
    )


# ----------------------------------------------------------------------
# FALLBACK - helpful instead of a blunt rejection
# ----------------------------------------------------------------------

def fallback() -> str:
    return (
        "I couldn't match that to something I can compute from your dataset. Try asking things like:\n\n"
        "- \"Summarize this dataset\"\n"
        "- \"What is the average <column>?\"\n"
        "- \"Which columns have missing values?\"\n"
        "- \"Show me the correlation\"\n"
        "- \"Suggest visualizations\"\n\n"
        "Open the **tips & examples** section above for the full list."
    )
