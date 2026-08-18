"""
data_cleaning.py
-----------------
This module powers the "Data Cleaning" tab: a focused, two-option
cleaning toolkit (Missing Values + Outlier Handling) that can be
combined in a single "Apply Cleaning" pass, plus exporting the
cleaned result back to CSV or Excel.

Design choice: cleaning NEVER mutates the original DataFrame in place.
Every function returns a NEW cleaned copy, so the raw dataset used by
the Dashboard / Visualization Studio / AI Chat tabs is never silently
changed - the app keeps the cleaned result in its own separate state.
"""

import os
import tempfile
import pandas as pd


# ----------------------------------------------------------------------
# MISSING VALUES
# ----------------------------------------------------------------------

def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a simple per-column missing-value table:
    Column | Missing Count | Missing %
    """
    columns_order = ["Column", "Missing Count", "Missing %"]

    if df is None or df.empty:
        return pd.DataFrame(columns=columns_order)

    total_rows = len(df)
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / total_rows * 100).round(2) if total_rows > 0 else missing_counts * 0

    summary = pd.DataFrame({
        "Column": missing_counts.index,
        "Missing Count": missing_counts.values,
        "Missing %": missing_pct.values,
    })
    # Show the worst offenders first.
    summary = summary.sort_values(by="Missing Count", ascending=False).reset_index(drop=True)
    return summary


def get_missing_value_stats(df: pd.DataFrame) -> dict:
    """
    A compact, at-a-glance version of get_missing_summary() - three
    numbers instead of a full table, for the small stat row shown
    above the searchable table in the Cleaning tab. Purely a different
    view of the same data; doesn't change any cleaning behavior.
    """
    if df is None or df.empty:
        return {"has_dataset": False}

    total_rows = len(df)
    missing_counts = df.isnull().sum()
    columns_with_missing = int((missing_counts > 0).sum())
    total_missing_cells = int(missing_counts.sum())

    if total_rows > 0 and columns_with_missing > 0:
        missing_pct = missing_counts / total_rows * 100
        highest_missing_pct = round(float(missing_pct.max()), 1)
        highest_missing_column = str(missing_pct.idxmax())
    else:
        highest_missing_pct = 0.0
        highest_missing_column = None

    return {
        "has_dataset": True,
        "columns_with_missing": columns_with_missing,
        "total_missing_cells": total_missing_cells,
        "highest_missing_pct": highest_missing_pct,
        "highest_missing_column": highest_missing_column,
    }


def search_missing_summary(df: pd.DataFrame, search_term: str) -> pd.DataFrame:
    """
    Same table as get_missing_summary(), optionally filtered to columns
    whose name contains `search_term` (case-insensitive) - powers the
    search box above the table. Purely a display filter; doesn't touch
    the underlying data or any cleaning logic.
    """
    summary = get_missing_summary(df)
    if not search_term or not search_term.strip() or summary.empty:
        return summary

    mask = summary["Column"].astype(str).str.contains(search_term.strip(), case=False, na=False, regex=False)
    return summary[mask].reset_index(drop=True)


def format_missing_value_stats_html(stats: dict) -> str:
    """
    Renders get_missing_value_stats() as a compact 3-stat row (or a
    single "dataset is clean" banner when there's nothing missing) -
    the small at-a-glance summary shown above the searchable missing-
    values table, so the table itself doesn't have to carry that load.
    """
    if not stats or not stats.get("has_dataset"):
        return "<div class='mini-stat-empty'>Upload a dataset to see missing-value stats.</div>"

    if stats["total_missing_cells"] == 0:
        return (
            "<div class='mini-stat-clean'>"
            "<span class='mini-stat-clean-icon'>✓</span>"
            "<span>No missing values detected — this dataset is clean.</span>"
            "</div>"
        )

    top_col = stats["highest_missing_column"]
    return (
        "<div class='mini-stat-row'>"
        f"<div class='mini-stat'><div class='mini-stat-value'>{stats['columns_with_missing']}</div>"
        "<div class='mini-stat-label'>Columns Affected</div></div>"
        f"<div class='mini-stat'><div class='mini-stat-value'>{stats['total_missing_cells']:,}</div>"
        "<div class='mini-stat-label'>Missing Cells</div></div>"
        f"<div class='mini-stat'><div class='mini-stat-value'>{stats['highest_missing_pct']}%</div>"
        f"<div class='mini-stat-label'>Highest ({top_col})</div></div>"
        "</div>"
    )


MISSING_VALUE_STRATEGIES = [
    "Do Nothing",
    "Drop rows with any missing values",
    "Fill numeric columns with Mean",
    "Fill numeric columns with Median",
    "Fill categorical columns with Mode",
]


def _apply_missing_value_strategy(df: pd.DataFrame, strategy: str) -> tuple:
    """Returns (cleaned_df, note_string) for the chosen strategy."""
    cleaned = df.copy()

    if strategy == "Drop rows with any missing values":
        before_rows = len(cleaned)
        cleaned = cleaned.dropna()
        removed = before_rows - len(cleaned)
        return cleaned, f"Dropped **{removed}** row(s) containing missing values."

    if strategy == "Fill numeric columns with Mean":
        numeric_cols = cleaned.select_dtypes(include="number").columns
        filled_cols = [c for c in numeric_cols if cleaned[c].isnull().any()]
        for col in filled_cols:
            cleaned[col] = cleaned[col].fillna(cleaned[col].mean())
        return cleaned, (
            f"Filled missing values in **{len(filled_cols)}** numeric column(s) with the column mean."
            if filled_cols else "No missing numeric values found to fill."
        )

    if strategy == "Fill numeric columns with Median":
        numeric_cols = cleaned.select_dtypes(include="number").columns
        filled_cols = [c for c in numeric_cols if cleaned[c].isnull().any()]
        for col in filled_cols:
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())
        return cleaned, (
            f"Filled missing values in **{len(filled_cols)}** numeric column(s) with the column median."
            if filled_cols else "No missing numeric values found to fill."
        )

    if strategy == "Fill categorical columns with Mode":
        cat_cols = cleaned.select_dtypes(include=["object", "category", "bool"]).columns
        filled_cols = []
        for col in cat_cols:
            if cleaned[col].isnull().any() and not cleaned[col].dropna().empty:
                mode_value = cleaned[col].mode().iloc[0]
                cleaned[col] = cleaned[col].fillna(mode_value)
                filled_cols.append(col)
        return cleaned, (
            f"Filled missing values in **{len(filled_cols)}** categorical column(s) with the most frequent value."
            if filled_cols else "No missing categorical values found to fill."
        )

    # "Do Nothing" or anything unrecognized.
    return cleaned, "No missing-value handling applied."


# ----------------------------------------------------------------------
# OUTLIER DETECTION (IQR method)
# ----------------------------------------------------------------------

def detect_outliers_iqr(df: pd.DataFrame, column: str):
    """
    Detects outliers in a numeric column using the standard IQR
    (Interquartile Range) rule: anything below Q1 - 1.5*IQR or above
    Q3 + 1.5*IQR is flagged.

    Returns
    -------
    (outlier_count, lower_bound, upper_bound, message) : tuple
        message is a ready-to-display markdown string; outlier_count/
        bounds are None if the column/data isn't usable.
    """
    if df is None:
        return None, None, None, "⚠ Please load a dataset first."
    if not column or column not in df.columns:
        return None, None, None, "⚠ Please select a numeric column."
    if not pd.api.types.is_numeric_dtype(df[column]):
        return None, None, None, f"⚠ **{column}** is not a numeric column."

    series = df[column].dropna()
    if series.empty:
        return None, None, None, f"⚠ **{column}** has no non-missing values to analyze."

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    outlier_count = int(outlier_mask.sum())

    message = (
        f"**{column}**: found **{outlier_count}** outlier(s) "
        f"outside the range [{lower_bound:.2f}, {upper_bound:.2f}] (using the 1.5×IQR rule)."
    )
    return outlier_count, lower_bound, upper_bound, message


def _remove_outliers(df: pd.DataFrame, column: str) -> tuple:
    """Removes rows outside the IQR bounds for the given column."""
    outlier_count, lower_bound, upper_bound, message = detect_outliers_iqr(df, column)

    if outlier_count is None:
        # Detection failed (bad column, no data, etc.) - leave df untouched.
        return df.copy(), f"Outlier removal skipped: {message}"

    if outlier_count == 0:
        return df.copy(), f"No outliers detected in **{column}** — nothing removed."

    cleaned = df[(df[column] >= lower_bound) & (df[column] <= upper_bound) | df[column].isnull()].copy()
    return cleaned, f"Removed **{outlier_count}** outlier row(s) based on **{column}**."


def _cap_outliers_with_max(df: pd.DataFrame, column: str) -> tuple:
    """
    Replaces every detected outlier in `column` with the maximum VALID
    (non-outlier) value found in that column - i.e. every value outside
    the IQR bounds gets clipped up to the largest value that wasn't
    flagged, instead of being dropped as a row. This keeps the row (and
    its other columns) in the dataset while neutralizing the extreme
    value.
    """
    outlier_count, lower_bound, upper_bound, message = detect_outliers_iqr(df, column)

    if outlier_count is None:
        # Detection failed (bad column, no data, etc.) - leave df untouched.
        return df.copy(), f"Outlier replacement skipped: {message}"

    if outlier_count == 0:
        return df.copy(), f"No outliers detected in **{column}** — nothing replaced."

    cleaned = df.copy()
    outlier_mask = (cleaned[column] < lower_bound) | (cleaned[column] > upper_bound)
    valid_values = cleaned.loc[~outlier_mask, column].dropna()

    if valid_values.empty:
        # Every non-missing value in this column was flagged as an
        # outlier - there's no "valid" value left to replace with.
        return df.copy(), (
            f"Outlier replacement skipped: every non-missing value in **{column}** was flagged as an "
            "outlier, so there's no valid value left to replace them with."
        )

    max_valid_value = valid_values.max()
    cleaned.loc[outlier_mask, column] = max_valid_value

    return cleaned, (
        f"Replaced **{outlier_count}** outlier value(s) in **{column}** with the maximum valid value "
        f"(**{max_valid_value:.2f}**)."
    )


# The two supported outlier strategies, plus "Do Nothing" for when the
# user just wants to preview outlier counts without touching the data.
OUTLIER_STRATEGIES = ["Do Nothing", "Remove Outliers", "Replace with Maximum Value"]


def _apply_outlier_strategy(df: pd.DataFrame, strategy: str, column: str) -> tuple:
    """Dispatches to the chosen outlier-handling function."""
    if strategy == "Remove Outliers" and column:
        return _remove_outliers(df, column)
    if strategy == "Replace with Maximum Value" and column:
        return _cap_outliers_with_max(df, column)
    # "Do Nothing", or a strategy was picked without a column selected.
    return df.copy(), "No outlier handling applied."


# ----------------------------------------------------------------------
# COMBINED CLEANING PASS
# ----------------------------------------------------------------------

def clean_dataset(
    df: pd.DataFrame,
    missing_strategy: str,
    outlier_strategy: str,
    outlier_column: str,
):
    """
    Runs the full cleaning pipeline in a fixed, predictable order:
        1. Missing value handling
        2. Outlier handling (Remove Outliers, or Replace with Maximum Value)

    Returns
    -------
    (cleaned_df, summary_markdown, missing_summary_df) : tuple
        cleaned_df is None if there was nothing to clean (no dataset loaded).
    """
    if df is None:
        return None, "⚠ Please load a dataset first.", get_missing_summary(None)

    original_rows, original_columns = df.shape
    original_missing = int(df.isnull().sum().sum())

    notes = []

    # ---- Step 1: Missing values ----
    cleaned, missing_note = _apply_missing_value_strategy(df, missing_strategy)
    notes.append(missing_note)

    # ---- Step 2: Outliers ----
    if outlier_strategy and outlier_strategy != "Do Nothing":
        cleaned, outlier_note = _apply_outlier_strategy(cleaned, outlier_strategy, outlier_column)
        notes.append(outlier_note)

    final_rows = len(cleaned)
    final_missing = int(cleaned.isnull().sum().sum())

    summary_md = "### 🧹 Cleaning Summary\n\n"
    summary_md += f"- **Rows:** {original_rows} → {final_rows} ({original_rows - final_rows} removed)\n"
    summary_md += f"- **Columns:** {original_columns} (unchanged)\n"
    summary_md += f"- **Missing values:** {original_missing} → {final_missing}\n\n"
    summary_md += "**Steps applied:**\n"
    for note in notes:
        summary_md += f"- {note}\n"

    return cleaned, summary_md, get_missing_summary(cleaned)


# ----------------------------------------------------------------------
# EXPORT
# ----------------------------------------------------------------------

EXPORT_FORMATS = ["CSV (.csv)", "Excel (.xlsx)"]


def export_dataset(df: pd.DataFrame, file_format: str):
    """
    Writes the cleaned dataset to a temporary file and returns its path,
    ready to be handed to a Gradio gr.File() output for download.

    Returns
    -------
    file_path : str or None   None if there's no cleaned dataset yet.
    """
    if df is None or df.empty:
        return None

    tmp_dir = tempfile.mkdtemp()

    if file_format == "Excel (.xlsx)":
        file_path = os.path.join(tmp_dir, "cleaned_dataset.xlsx")
        df.to_excel(file_path, index=False, engine="openpyxl")
    else:
        file_path = os.path.join(tmp_dir, "cleaned_dataset.csv")
        df.to_csv(file_path, index=False)

    return file_path


# ----------------------------------------------------------------------
# CLEANING SUMMARY PREVIEW - describes what WOULD happen, without
# actually cleaning anything. Purely read-only / additive: doesn't
# touch clean_dataset() or any existing function above.
# ----------------------------------------------------------------------

def preview_cleaning_plan(
    df: pd.DataFrame,
    missing_strategy: str,
    outlier_strategy: str,
    outlier_column: str,
) -> str:
    """
    Builds a live "here's what will happen if you click Apply Cleaning"
    summary based on the CURRENT control values - this only reads the
    data to describe the plan, it never modifies or cleans anything.
    Meant to update instantly as the user toggles the cleaning options,
    before they've committed to anything.
    """
    if df is None or df.empty:
        return "Upload a dataset to see the cleaning plan."

    steps = []
    total_missing = int(df.isnull().sum().sum())

    if missing_strategy and missing_strategy != "Do Nothing":
        if total_missing > 0:
            steps.append(f"Missing values will be handled using: **{missing_strategy}**.")
        else:
            steps.append(f"'{missing_strategy}' is selected, but no missing values were found — nothing to change.")

    if outlier_strategy and outlier_strategy != "Do Nothing" and outlier_column:
        outlier_count, _, _, _ = detect_outliers_iqr(df, outlier_column)
        if outlier_count:
            action = "removed" if outlier_strategy == "Remove Outliers" else "replaced with the maximum valid value"
            steps.append(
                f"Outliers in **{outlier_column}** will be {action} (**{outlier_count}** row(s) detected)."
            )
        else:
            steps.append(f"**{outlier_strategy}** is selected for **{outlier_column}**, but no outliers were detected.")

    if not steps:
        return "No cleaning actions selected yet — adjust the options above, then click **Apply Cleaning**."

    return "**Planned changes:**\n\n" + "\n".join(f"- {step}" for step in steps)
