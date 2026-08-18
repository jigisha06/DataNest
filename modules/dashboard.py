"""
dashboard.py
------------
This module calculates all the statistics shown on the "Dashboard" tab:

    - Total Rows
    - Total Columns
    - Missing Values
    - Duplicate Records
    - Memory Usage
    - Missing Percentage
    - Numeric Features (count)
    - Categorical Features (count)
    - Dataset Preview (head)

Separating this from app.py means that later, other tabs (like AI
Readiness) can re-use the same numbers without recalculating them.
"""

import pandas as pd


def get_dashboard_stats(df: pd.DataFrame) -> dict:
    """
    Calculates key summary statistics for a given DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset currently loaded in the app.

    Returns
    -------
    stats : dict
        A dictionary containing every metric needed for the Dashboard tab.
    """

    if df is None:
        return {}

    # --- Basic shape ---
    total_rows = df.shape[0]
    total_columns = df.shape[1]

    # --- Missing values ---
    total_missing = int(df.isnull().sum().sum())
    total_cells = total_rows * total_columns
    missing_percentage = round((total_missing / total_cells) * 100, 2) if total_cells > 0 else 0.0

    # --- Duplicate rows ---
    duplicate_records = int(df.duplicated().sum())

    # --- Memory usage (convert bytes -> KB or MB for readability) ---
    memory_bytes = df.memory_usage(deep=True).sum()
    if memory_bytes >= 1_048_576:  # >= 1 MB
        memory_usage = f"{memory_bytes / 1_048_576:.2f} MB"
    else:
        memory_usage = f"{memory_bytes / 1024:.2f} KB"

    # --- Feature types ---
    numeric_features = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    stats = {
        "Total Rows": total_rows,
        "Total Columns": total_columns,
        "Missing Values": total_missing,
        "Missing Percentage": f"{missing_percentage}%",
        "Duplicate Records": duplicate_records,
        "Numeric Features": len(numeric_features),
        "Categorical Features": len(categorical_features),
    }

    return stats


def format_stats_as_markdown(stats: dict) -> str:
    """
    Converts the stats dictionary into a clean Markdown summary
    so it displays nicely inside a Gradio Markdown component.

    (Kept for reference/backward compatibility - the Dashboard tab
    itself now uses format_stats_as_kpi_cards() below instead.)
    """
    if not stats:
        return "### ⚠ No dataset loaded yet.\nPlease upload a CSV or Excel file and click **Load Dataset**."

    md = "### 📊 Dataset Summary\n\n"
    md += "| Metric | Value |\n"
    md += "|---|---|\n"
    for key, value in stats.items():
        md += f"| **{key}** | {value} |\n"

    return md


def format_stats_as_kpi_cards(stats: dict) -> str:
    """
    Converts the stats dictionary into an HTML grid of KPI cards -
    a modern dashboard look instead of a plain table.

    Numeric/Categorical feature counts are shown as small chips INSIDE
    the "Total Columns" card (rather than as their own separate KPI
    cards) to keep the dashboard cleaner with fewer cards overall.

    Returned as raw HTML so it can be dropped straight into a
    Gradio gr.HTML() component. Styling (the .kpi-* classes) lives
    in app.py's CUSTOM_CSS so all dashboard-style visuals share one
    place for their look and feel.
    """
    if not stats:
        return (
            "<div class='kpi-empty'>"
            "No dataset loaded yet. Please upload a CSV or Excel file and click "
            "<b>Load Dataset</b>."
            "</div>"
        )

    numeric_count = stats.get("Numeric Features")
    categorical_count = stats.get("Categorical Features")

    # ---- The special "Total Columns" card, with Numeric/Categorical
    #      shown as small chips underneath the main value instead of
    #      as their own separate cards. ----
    total_columns_card = f"""
    <div class="kpi-card">
        <div class="kpi-icon">📊</div>
        <div class="kpi-value">{stats.get("Total Columns")}</div>
        <div class="kpi-label">Total Number of Columns</div>
        <div class="kpi-subchips">
            <span class="kpi-subchip"> Numeric :{numeric_count} </span>
            <span class="kpi-subchip"> Categorical : {categorical_count} </span>
        </div>
    </div>
    """

    # (icon, label, value) - every OTHER card, in the order they'll appear.
    kpi_definitions = [
        ("📄", "Total Number of Rows", stats.get("Total Rows")),
        ("📉", "Percentage of Missing Values", stats.get("Missing Percentage")),
        ("🔁", "Total Duplicate Records", stats.get("Duplicate Records")),
    ]

    cards_html = total_columns_card
    for icon, label, value in kpi_definitions:
        cards_html += f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{label}</div>
        </div>
        """

    return f'<div class="kpi-grid">{cards_html}</div>'


def get_preview(df: pd.DataFrame, n_rows: int = 10) -> pd.DataFrame:
    """
    Returns the first `n_rows` of the dataset for a quick preview.
    Returns an empty DataFrame if no data is loaded.
    """
    if df is None:
        return pd.DataFrame()
    return df.head(n_rows)
