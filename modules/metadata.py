"""
metadata.py
-----------
Powers the "Metadata" tab - a full analytics report generated
automatically for whatever CSV/Excel file is uploaded, inspired by
Power BI / Tableau / Looker Studio dashboards rather than a plain
dataframe viewer.

Organized into sections, each with its own (data-building) + (HTML-
rendering) function pair so the math and the presentation stay
cleanly separated - the same principle used throughout this project:

    1. Statistical Summary          - describe()-style stats per column type
    2. Business Intelligence KPIs   - Average/Total/Max/Min/Median/Std per numeric column
    3. Automatic Visual Analytics   - dataset-aware charts (only what makes sense)
    4. Smart AI Insights            - pattern-based observations (heuristics
                                       computed directly from the data - NOT a
                                       machine-learning model, see docstrings)

NOTE: get_dataset_profile() / format_dataset_profile_html() (the old
per-column "report card" section) are kept below for reuse elsewhere,
but are no longer included in the live Data Insights report.

Where equivalent logic already exists elsewhere in the app (missing
value stats, correlation pairs, outlier detection), we import and
reuse it from modules.ai_chat.analysis instead of duplicating the
math - the same reuse pattern already used by the AI Chat assistant.
"""

import io
import re
import base64

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend - no display in this environment
import matplotlib.pyplot as plt
import seaborn as sns

from modules.ai_chat.analysis import (
    get_column_types,
    get_missing_info,
    get_highly_correlated_pairs,
    get_outlier_summary,
    get_overall_numeric_summary,
)
from modules.data_cleaning import detect_outliers_iqr


# ----------------------------------------------------------------------
# LEGACY FUNCTION (kept for backward compatibility)
# ----------------------------------------------------------------------

def get_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    The ORIGINAL simple column-level metadata table:
    Column Name | Data Type | Missing Values | Unique Values

    No longer used by the Metadata tab itself (which now shows the
    full BI dashboard below), but kept here in case anything else in
    the project still references it.
    """
    columns_order = ["Column Name", "Data Type", "Missing Values", "Unique Values"]

    if df is None:
        return pd.DataFrame(columns=columns_order)

    rows = []
    for column in df.columns:
        rows.append({
            "Column Name": column,
            "Data Type": str(df[column].dtype),
            "Missing Values": int(df[column].isnull().sum()),
            "Unique Values": int(df[column].nunique(dropna=True)),
        })

    return pd.DataFrame(rows, columns=columns_order)


# ----------------------------------------------------------------------
# SMALL SHARED HELPERS
# ----------------------------------------------------------------------

def _dataframe_to_html_table(df: pd.DataFrame) -> str:
    """Renders a small pandas DataFrame as a plain HTML table using our
    own CSS classes (.meta-table), instead of pandas' default styling."""
    if df is None or df.empty:
        return ""
    headers = "".join(f"<th>{col}</th>" for col in df.columns)
    body_rows = ""
    for _, row in df.iterrows():
        cells = "".join(f"<td>{row[col]}</td>" for col in df.columns)
        body_rows += f"<tr>{cells}</tr>"
    return f"<table class='meta-table'><thead><tr>{headers}</tr></thead><tbody>{body_rows}</tbody></table>"


def _fig_to_data_uri(fig) -> str:
    """Converts a Matplotlib figure to a base64 PNG data URI, so several
    dynamically-chosen charts can be embedded directly in one HTML
    string (a single gr.HTML component) instead of needing a fixed
    number of separate gr.Plot components."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _chart_card(title: str, image_data_uri: str) -> str:
    return (
        f"<div class='chart-card'>"
        f"<div class='chart-card-title'>{title}</div>"
        f"<img src='{image_data_uri}' class='chart-card-img' alt='{title}' />"
        f"</div>"
    )


# ----------------------------------------------------------------------
# SECTION 1 - STATISTICAL SUMMARY
# ----------------------------------------------------------------------

def get_categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every categorical column: Count, Unique, Most Frequent value,
    and how often that most-frequent value occurs.
    """
    categorical_cols = get_column_types(df)["categorical"]
    if not categorical_cols:
        return pd.DataFrame()

    rows = []
    for col in categorical_cols:
        series = df[col].dropna()
        count = int(series.shape[0])
        unique = int(series.nunique())
        if not series.empty:
            mode_value = series.mode().iloc[0]
            frequency = int((series == mode_value).sum())
        else:
            mode_value, frequency = "N/A", 0
        rows.append({
            "Column": col, "Count": count, "Unique": unique,
            "Most Frequent": mode_value, "Frequency": frequency,
        })
    return pd.DataFrame(rows)


def format_statistical_summary_html(df: pd.DataFrame) -> str:
    """Section 1: numeric describe()-style stats + categorical summary,
    presented as clean tables inside a card instead of a raw dataframe."""
    if df is None or df.empty:
        return "<div class='kpi-empty'>No dataset loaded yet.</div>"

    numeric_summary = get_overall_numeric_summary(df)  # count/mean/std/min/25%/50%/75%/max
    categorical_summary = get_categorical_summary(df)

    html = ""
    if not numeric_summary.empty:
        numeric_table = numeric_summary.reset_index().rename(columns={"index": "Statistic"})
        # "50%" is technically the median - relabel it so non-technical
        # users immediately recognize it, per the requested statistical summary.
        numeric_table["Statistic"] = numeric_table["Statistic"].replace({"50%": "Median"})
        html += "<div class='meta-subsection-title'>Numeric Columns</div>"
        html += _dataframe_to_html_table(numeric_table)
    if not categorical_summary.empty:
        html += "<div class='meta-subsection-title'>Categorical Columns</div>"
        html += _dataframe_to_html_table(categorical_summary)

    if not html:
        return "<div class='kpi-empty'>No columns available to summarize.</div>"

    return f"<div class='meta-card'>{html}</div>"


# ----------------------------------------------------------------------
# SECTION 2 - DATASET PROFILE
# ----------------------------------------------------------------------

def get_dataset_profile(df: pd.DataFrame) -> list:
    """
    Builds a per-column analytical profile: name, dtype, missing count/%,
    unique count, a sample value, and a heuristic "recommended usage"
    note (e.g. "likely an identifier", "consider encoding").

    These recommendations are simple, transparent heuristics based on
    cardinality and dtype - not a trained model - so they're always
    explainable and reproducible for any dataset.
    """
    if df is None or df.empty:
        return []

    total_rows = len(df)
    numeric_cols = set(get_column_types(df)["numeric"])
    profile = []

    for col in df.columns:
        series = df[col]
        missing = int(series.isnull().sum())
        missing_pct = round((missing / total_rows) * 100, 1) if total_rows else 0.0
        unique = int(series.nunique(dropna=True))
        non_null = series.dropna()
        sample_value = non_null.iloc[0] if not non_null.empty else "N/A"

        # ---- Recommended usage heuristic ----
        if total_rows > 0 and unique == total_rows:
            usage = "Likely an identifier — exclude from modeling"
        elif col in numeric_cols:
            usage = "Numeric — low cardinality, could be treated as categorical" if unique <= 10 \
                else "Numeric feature — suitable for analysis and modeling"
        else:
            usage = "Categorical — consider one-hot or label encoding" if unique <= 20 \
                else "High-cardinality text — consider grouping or NLP techniques"

        profile.append({
            "Column": col,
            "Data Type": str(series.dtype),
            "Missing": missing,
            "Missing %": missing_pct,
            "Unique": unique,
            "Sample Value": sample_value,
            "Recommended Usage": usage,
        })

    return profile


def format_dataset_profile_html(profile: list) -> str:
    """Section 2: one report-style card per column instead of a plain dataframe."""
    if not profile:
        return "<div class='kpi-empty'>No dataset loaded yet.</div>"

    cards = ""
    for entry in profile:
        if entry["Missing %"] > 20:
            missing_color = "#f43f5e"
        elif entry["Missing %"] > 0:
            missing_color = "#f59e0b"
        else:
            missing_color = "#22c55e"

        cards += f"""
        <div class="profile-card">
            <div class="profile-card-header">
                <span class="profile-col-name">{entry['Column']}</span>
                <span class="profile-dtype-badge">{entry['Data Type']}</span>
            </div>
            <div class="profile-card-body">
                <div class="profile-stat"><span>Missing</span><b style="color:{missing_color}">{entry['Missing']} ({entry['Missing %']}%)</b></div>
                <div class="profile-stat"><span>Unique</span><b>{entry['Unique']}</b></div>
                <div class="profile-stat"><span>Sample</span><b>{entry['Sample Value']}</b></div>
            </div>
            <div class="profile-usage">{entry['Recommended Usage']}</div>
        </div>
        """

    return f"<div class='profile-grid'>{cards}</div>"


# ----------------------------------------------------------------------
# SECTION 3 - BUSINESS INTELLIGENCE KPIs
# ----------------------------------------------------------------------

# Column-name tokens that strongly suggest "this is an identifier, not
# a business metric" - Year, ID, Index, Serial, Code, etc. Checked as
# WHOLE tokens (not substrings) so "width" or "grid" never false-match
# just because they happen to contain "id".
_IDENTIFIER_NAME_TOKENS = {"id", "index", "idx", "serial", "code", "year", "no", "num", "number"}


def _tokenize_column_name(column_name: str) -> list:
    """
    Splits a column name into lowercase tokens, handling both explicit
    separators ("order_id", "Order ID") AND camelCase/concatenated
    forms ("CustomerID", "OrderId") by inserting a boundary before an
    uppercase letter that follows a lowercase one - without this,
    "CustomerID" would stay as one token and never match "id".
    """
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", column_name)
    return [t for t in re.split(r"[^a-zA-Z0-9]+", spaced.lower()) if t]


def _is_identifier_like_column(df: pd.DataFrame, column: str) -> bool:
    """
    True if a numeric column looks like an identifier rather than a
    real business metric - either by name (Year, Order ID, Customer ID,
    Serial Number, Index, ...) or by behavior.

    The "every value is unique" signal is only applied to INTEGER
    columns - continuous float metrics (Sales, Profit, Discount, ...)
    are naturally all-unique too just from floating-point precision,
    so treating that as an identifier signal for floats would wrongly
    exclude genuine business metrics.
    """
    tokens = _tokenize_column_name(column)
    if any(token in _IDENTIFIER_NAME_TOKENS for token in tokens):
        return True

    series = df[column]
    total_rows = len(df)
    if pd.api.types.is_integer_dtype(series) and total_rows > 0 and series.nunique(dropna=True) == total_rows:
        return True

    return False


def get_business_metric_columns(df: pd.DataFrame) -> list:
    """
    Numeric columns that look like genuine business metrics - i.e.
    NOT identifiers, years, index/serial columns, or any other
    numeric column that merely labels a row rather than measuring
    something. This is what powers the "Select Metric to Analyze"
    dropdown and the default KPI cards.
    """
    if df is None or df.empty:
        return []
    numeric_cols = get_column_types(df)["numeric"]
    return [col for col in numeric_cols if not _is_identifier_like_column(df, col)]


def get_default_business_metrics(df: pd.DataFrame, top_n: int = 1) -> list:
    """
    Auto-picks the most relevant business metric(s) to show by default,
    ranked by coefficient of variation (relative spread) - the same
    "which feature has the most going on" heuristic already used for
    Smart Insights, just applied here to pick a sensible default
    instead of showing every numeric column's KPIs at once.
    """
    business_cols = get_business_metric_columns(df)
    if not business_cols:
        return []

    scores = {}
    for col in business_cols:
        series = df[col].dropna()
        if len(series) > 1 and series.mean() != 0:
            scores[col] = abs(series.std() / series.mean())
        else:
            scores[col] = 0.0

    ranked = sorted(business_cols, key=lambda c: scores.get(c, 0.0), reverse=True)
    return ranked[:top_n]


def get_bi_kpis(df: pd.DataFrame, columns=None, max_columns: int = 6) -> dict:
    """
    Average / Total / Max / Min / Median / Std for the given column(s).

    Parameters
    ----------
    df : pandas.DataFrame
    columns : str, list, or None
        - None (default): auto-picks the top business metric via
          get_default_business_metrics() - identifier/year/index-like
          columns are EXCLUDED automatically.
        - a single column name (str): shows KPIs for just that column
          (used by the "Select Metric to Analyze" dropdown).
        - a list of column names: shows KPIs for each (capped at
          `max_columns` so a very wide dataset can't overwhelm the page).
    """
    if df is None or df.empty:
        return {}

    if columns is None:
        columns = get_default_business_metrics(df, top_n=1)
    elif isinstance(columns, str):
        columns = [columns]

    columns = [c for c in columns if c in df.columns][:max_columns]

    kpis = {}
    for col in columns:
        series = df[col].dropna()
        if series.empty:
            continue
        kpis[col] = {
            "avg": round(series.mean(), 2),
            "total": round(series.sum(), 2),
            "max": round(series.max(), 2),
            "min": round(series.min(), 2),
            "median": round(series.median(), 2),
            "std": round(series.std(), 2) if len(series) > 1 else 0.0,
        }
    return kpis


def format_bi_kpis_html(kpis: dict) -> str:
    """Section 3: a KPI card row per numeric column."""
    if not kpis:
        return "<div class='kpi-empty'>No numeric columns available for BI KPIs.</div>"

    blocks = ""
    for column, stats in kpis.items():
        blocks += f"""
        <div class="bi-kpi-block">
            <div class="bi-kpi-column-title">{column}</div>
            <div class="kpi-grid bi-kpi-grid">
                <div class="kpi-card"><div class="kpi-value">{stats['avg']}</div><div class="kpi-label">Average</div></div>
                <div class="kpi-card"><div class="kpi-value">{stats['total']}</div><div class="kpi-label">Total</div></div>
                <div class="kpi-card"><div class="kpi-value">{stats['max']}</div><div class="kpi-label">Max</div></div>
                <div class="kpi-card"><div class="kpi-value">{stats['min']}</div><div class="kpi-label">Min</div></div>
                <div class="kpi-card"><div class="kpi-value">{stats['median']}</div><div class="kpi-label">Median</div></div>
                <div class="kpi-card"><div class="kpi-value">{stats['std']}</div><div class="kpi-label">Std Dev</div></div>
            </div>
        </div>
        """
    return blocks


# ----------------------------------------------------------------------
# SECTION 4 - AUTOMATIC VISUAL ANALYTICS
# ----------------------------------------------------------------------

def generate_visual_analytics_html(df: pd.DataFrame) -> str:
    """
    Auto-generates only the charts that make sense for THIS dataset:
        - Data Type Distribution   (always, if there's at least 1 column)
        - Missing Value Distribution (only if there ARE missing values)
        - Histogram                (only if there's a numeric column)
        - Box Plot                 (only if there's a numeric column)
        - Correlation Heatmap      (only if there are 2+ numeric columns)
        - Top Categories           (only if there's a categorical column)

    Every chart is wrapped in its own try/except so one bad chart
    (e.g. unusual data) can never take down the whole report - it's
    just silently skipped.
    """
    if df is None or df.empty:
        return "<div class='kpi-empty'>No dataset loaded yet.</div>"

    col_types = get_column_types(df)
    numeric_cols = col_types["numeric"]
    categorical_cols = col_types["categorical"]
    charts_html = ""

    # ---- Data Type Distribution ----
    try:
        type_counts = {
            "Numeric": len(numeric_cols),
            "Categorical": len(categorical_cols),
            "Datetime": len(col_types["datetime"]),
        }
        type_counts = {k: v for k, v in type_counts.items() if v > 0}
        if type_counts:
            fig, ax = plt.subplots(figsize=(4, 3))
            colors = ["#6366f1", "#8b5cf6", "#22d3ee"][:len(type_counts)]
            ax.bar(type_counts.keys(), type_counts.values(), color=colors)
            ax.set_title("Data Type Distribution")
            ax.set_ylabel("Number of Columns")
            fig.tight_layout()
            charts_html += _chart_card("Data Type Distribution", _fig_to_data_uri(fig))
    except Exception:
        pass

    # ---- Missing Value Distribution ----
    try:
        missing_info = get_missing_info(df)
        if missing_info["total_missing"] > 0:
            top_missing = missing_info["columns_with_missing"].head(10)
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.barh(top_missing.index.astype(str), top_missing.values, color="#f59e0b")
            ax.set_title("Missing Values by Column")
            ax.invert_yaxis()
            fig.tight_layout()
            charts_html += _chart_card("Missing Value Distribution", _fig_to_data_uri(fig))
    except Exception:
        pass

    # ---- Histogram (first numeric column) ----
    try:
        if numeric_cols:
            col = numeric_cols[0]
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(df[col].dropna(), bins=25, color="#6366f1", edgecolor="white")
            ax.set_title(f"Histogram — {col}")
            fig.tight_layout()
            charts_html += _chart_card(f"Histogram — {col}", _fig_to_data_uri(fig))
    except Exception:
        pass

    # ---- Box Plot (up to 4 numeric columns) ----
    try:
        if numeric_cols:
            cols_to_plot = numeric_cols[:4]
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.boxplot(
                [df[c].dropna() for c in cols_to_plot],
                tick_labels=cols_to_plot, patch_artist=True,
                boxprops=dict(facecolor="#8b5cf6", alpha=0.5),
            )
            ax.set_title("Box Plot — Numeric Columns")
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            fig.tight_layout()
            charts_html += _chart_card("Box Plot", _fig_to_data_uri(fig))
    except Exception:
        pass

    # ---- Correlation Heatmap (2+ numeric columns) ----
    try:
        if len(numeric_cols) >= 2:
            fig, ax = plt.subplots(figsize=(5, 4))
            correlation = df[numeric_cols].corr()
            sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, cbar=False)
            ax.set_title("Correlation Heatmap")
            fig.tight_layout()
            charts_html += _chart_card("Correlation Heatmap", _fig_to_data_uri(fig))
    except Exception:
        pass

    # ---- Top Categories (first categorical column) ----
    try:
        if categorical_cols:
            col = categorical_cols[0]
            value_counts = df[col].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.barh(value_counts.index.astype(str), value_counts.values, color="#22d3ee")
            ax.set_title(f"Top Categories — {col}")
            ax.invert_yaxis()
            fig.tight_layout()
            charts_html += _chart_card(f"Top Categories — {col}", _fig_to_data_uri(fig))
    except Exception:
        pass

    if not charts_html:
        return "<div class='kpi-empty'>No charts could be generated for this dataset.</div>"

    return f"<div class='chart-grid'>{charts_html}</div>"


# ----------------------------------------------------------------------
# SECTION 5 - SMART AI INSIGHTS
# ----------------------------------------------------------------------

def get_smart_insights(df: pd.DataFrame) -> list:
    """
    Generates a handful of pattern-based observations directly from the
    data - highest-variance feature, most/least complete columns,
    likely identifier/target columns, correlated pairs, and possible
    outlier columns.

    IMPORTANT: these are deterministic statistical heuristics (variance,
    cardinality, correlation, IQR) - NOT predictions from a trained
    machine learning model. They're transparent and reproducible for
    any dataset, which is why they can run instantly with zero setup.
    """
    if df is None or df.empty:
        return []

    insights = []
    total_rows = len(df)
    numeric_cols = get_column_types(df)["numeric"]

    # ---- Highest variance feature (coefficient of variation, so
    #      columns on different scales can be compared fairly) ----
    cv_scores = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) > 1 and series.mean() != 0:
            cv_scores[col] = abs(series.std() / series.mean())
    if cv_scores:
        top_variance_col = max(cv_scores, key=cv_scores.get)
        insights.append(("📈", "Highest Variance Feature",
                          f"<b>{top_variance_col}</b> shows the most relative spread in its values.",
                          "Distribution"))

    # ---- Most complete column(s) ----
    missing_info = get_missing_info(df)
    complete_columns = missing_info["per_column"][missing_info["per_column"] == 0]
    if not complete_columns.empty:
        insights.append(("✅", "Most Complete Column(s)",
                          f"<b>{len(complete_columns)}</b> column(s) have zero missing values, "
                          f"e.g. <b>{complete_columns.index[0]}</b>.",
                          "Quality"))

    # ---- Column with the most missing values ----
    if not missing_info["columns_with_missing"].empty:
        top_missing_col = missing_info["columns_with_missing"].index[0]
        top_missing_count = int(missing_info["columns_with_missing"].iloc[0])
        insights.append(("⚠️", "Most Missing Values",
                          f"<b>{top_missing_col}</b> has the most missing values ({top_missing_count}).",
                          "Quality"))

    # ---- Possible identifier column (every value unique) ----
    id_candidates = [
        col for col in df.columns
        if total_rows > 0 and df[col].nunique(dropna=True) == total_rows
    ]
    if id_candidates:
        insights.append(("🔑", "Possible Identifier Column",
                          f"<b>{id_candidates[0]}</b> has a unique value for every row — likely an ID column.",
                          "Structure"))

    # ---- Potential target column (few unique values, not an ID) ----
    target_candidates = [
        col for col in df.columns
        if col not in id_candidates and 2 <= df[col].nunique(dropna=True) <= 10
    ]
    if target_candidates:
        insights.append(("🎯", "Potential Target Column",
                          f"<b>{target_candidates[-1]}</b> has few unique values, making it a plausible prediction target.",
                          "Structure"))

    # ---- Highly correlated columns ----
    high_corr_pairs = get_highly_correlated_pairs(df, threshold=0.7)
    if high_corr_pairs:
        col1, col2, corr_value = high_corr_pairs[0]
        insights.append(("🔗", "Highly Correlated Columns",
                          f"<b>{col1}</b> and <b>{col2}</b> are strongly correlated ({corr_value:+.2f}).",
                          "Correlation"))

    # ---- Possible outlier columns ----
    outlier_summary = get_outlier_summary(df)
    if outlier_summary:
        top_outlier_col, top_outlier_count = outlier_summary[0]
        insights.append(("🚨", "Possible Outlier Column",
                          f"<b>{top_outlier_col}</b> has {top_outlier_count} potential outlier value(s).",
                          "Outlier"))

    return insights


def format_smart_insights_html(insights: list) -> str:
    """Section 5: one insight card per observation, each with a small
    category badge, instead of a paragraph."""
    if not insights:
        return "<div class='kpi-empty'>No notable insights detected for this dataset.</div>"

    cards = ""
    for icon, title, description, badge in insights:
        cards += f"""
        <div class="insight-card">
            <div class="insight-icon">{icon}</div>
            <div class="insight-content">
                <div class="insight-card-header">
                    <span class="insight-title">{title}</span>
                    <span class="insight-badge">{badge}</span>
                </div>
                <div class="insight-description">{description}</div>
            </div>
        </div>
        """
    return f"<div class='insight-grid'>{cards}</div>"


# ----------------------------------------------------------------------
# MASTER FUNCTION - combines all 5 sections into one HTML report
# ----------------------------------------------------------------------

def render_metadata_dashboard(df: pd.DataFrame) -> str:
    """
    Builds the complete Metadata tab as a single HTML string (all 5
    sections, with section headings), so the app only needs ONE
    gr.HTML() component for this whole tab.

    NOTE: kept for backward compatibility. The Metadata tab itself now
    uses render_metadata_top_html() / the BI KPI section / and
    render_metadata_bottom_html() as three separate pieces instead,
    so the "Select Metric to Analyze" dropdown can refresh just the
    BI KPI cards without re-rendering everything else on the page.
    """
    if df is None or df.empty:
        return (
            "<div class='kpi-empty'>"
            "No dataset loaded yet. Please upload a CSV or Excel file and click "
            "<b>Load Dataset</b>."
            "</div>"
        )

    html = ""
    html += "<h3 class='meta-section-heading'>Statistical Summary</h3>"
    html += format_statistical_summary_html(df)

    html += "<h3 class='meta-section-heading'>Dataset Profile</h3>"
    html += format_dataset_profile_html(get_dataset_profile(df))

    html += "<h3 class='meta-section-heading'>Business Intelligence KPIs</h3>"
    html += format_bi_kpis_html(get_bi_kpis(df))

    html += "<h3 class='meta-section-heading'>Automatic Visual Analytics</h3>"
    html += generate_visual_analytics_html(df)

    html += "<h3 class='meta-section-heading'>Smart AI Insights</h3>"
    html += format_smart_insights_html(get_smart_insights(df))

    return html


def render_metadata_top_html(df: pd.DataFrame) -> str:
    """
    Everything ABOVE the interactive Business Intelligence KPIs
    section: Statistical Summary, then Automatic Visual Analytics
    (moved directly below the statistics, per the requested layout).
    """
    if df is None or df.empty:
        return (
            "<div class='kpi-empty'>"
            "No dataset loaded yet. Please upload a CSV or Excel file and click "
            "<b>Load Dataset</b>."
            "</div>"
        )

    html = ""
    html += "<h3 class='meta-section-heading'>Statistical Summary</h3>"
    html += format_statistical_summary_html(df)

    html += "<h3 class='meta-section-heading'>Automatic Visual Analytics</h3>"
    html += generate_visual_analytics_html(df)

    return html


def render_metadata_bottom_html(df: pd.DataFrame) -> str:
    """Everything BELOW the interactive Business Intelligence KPIs section: Smart AI Insights."""
    if df is None or df.empty:
        return ""

    html = ""
    html += "<h3 class='meta-section-heading'>Smart AI Insights</h3>"
    html += format_smart_insights_html(get_smart_insights(df))

    return html


# ----------------------------------------------------------------------
# QUICK VISUAL SUMMARY - interactive, per-column, on-demand charts
# (separate from Section 4's automatic dataset-wide charts above -
# this lets the user explore any ONE column without leaving the tab).
# ----------------------------------------------------------------------

def _describe_numeric_skew(skew_value: float) -> str:
    """Translates a skewness number into a plain-English shape description."""
    if pd.isna(skew_value):
        return "roughly symmetric"
    if skew_value > 1:
        return "strongly right-skewed, with a long tail of unusually high values"
    if skew_value > 0.5:
        return "moderately right-skewed, leaning toward higher values"
    if skew_value < -1:
        return "strongly left-skewed, with a long tail of unusually low values"
    if skew_value < -0.5:
        return "moderately left-skewed, leaning toward lower values"
    return "roughly symmetric, close to a normal distribution"


def _generate_numeric_column_insight(df: pd.DataFrame, column: str) -> str:
    """
    Builds a 2-3 sentence, plain-English read of a numeric column's
    histogram/box plot: range + central tendency, distribution shape,
    and outliers - the same IQR rule used everywhere else in the app,
    reused here instead of re-implemented.
    """
    series = df[column].dropna()
    if series.empty:
        return "No non-missing values are available for this column, so no insight could be generated."

    total_rows = len(df)
    mean_val = series.mean()
    median_val = series.median()
    min_val = series.min()
    max_val = series.max()
    skew_val = series.skew() if len(series) > 2 else 0.0
    shape_desc = _describe_numeric_skew(skew_val)

    outlier_count, lower_bound, upper_bound, _ = detect_outliers_iqr(df, column)
    outlier_count = outlier_count or 0

    sentences = [
        f"<b>{column}</b> ranges from <b>{min_val:,.2f}</b> to <b>{max_val:,.2f}</b>, "
        f"with a mean of <b>{mean_val:,.2f}</b> and a median of <b>{median_val:,.2f}</b>."
    ]

    # A mean noticeably above/below the median is itself a quick tell
    # of skew direction, so fold that observation into the shape line.
    sentences.append(f"The distribution looks {shape_desc}.")

    if outlier_count > 0:
        outlier_pct = round((outlier_count / total_rows) * 100, 1) if total_rows else 0
        # Spell out the actual rule that flagged these points (not just
        # the count) - this is what the box plot's fliers/dots are:
        # anything outside [lower_bound, upper_bound] gets drawn as an
        # individual point instead of inside the whiskers. Naming the
        # bounds explains *why* the box plot can look "dotty" even
        # though nothing is actually wrong with the chart.
        sentences.append(
            f"⚠ <b>{outlier_count}</b> potential outlier(s) ({outlier_pct}% of rows) fall outside the "
            f"typical range of <b>{lower_bound:,.2f}</b> to <b>{upper_bound:,.2f}</b> (the 1.5×IQR rule "
            "used for the box plot's whiskers) — these show up as the individual dots above/below the "
            "whiskers, and are worth a closer look if this column has a long tail (e.g. profit/loss data), "
            "that's expected rather than a data error."
        )
    else:
        sentences.append("No significant outliers were detected using the 1.5×IQR rule.")

    return " ".join(sentences)


def _generate_categorical_column_insight(df: pd.DataFrame, column: str) -> str:
    """
    Builds a 2-3 sentence, plain-English read of a categorical column's
    top-categories bar chart: how many distinct values exist, which
    one dominates, and whether the split looks balanced or skewed
    toward a single category.
    """
    series = df[column].dropna()
    if series.empty:
        return "No non-missing values are available for this column, so no insight could be generated."

    total_non_null = len(series)
    value_counts = series.value_counts()
    unique_count = int(series.nunique(dropna=True))
    top_value = value_counts.index[0]
    top_count = int(value_counts.iloc[0])
    top_pct = round((top_count / total_non_null) * 100, 1) if total_non_null else 0.0

    sentences = [
        f"<b>{column}</b> has <b>{unique_count}</b> distinct value(s), and <b>{top_value}</b> is the "
        f"most common — appearing in <b>{top_count}</b> row(s) ({top_pct}% of non-missing entries)."
    ]

    if len(value_counts) >= 2:
        second_value = value_counts.index[1]
        second_count = int(value_counts.iloc[1])
        sentences.append(f"The next most common value is <b>{second_value}</b> ({second_count} row(s)).")

    if top_pct >= 70:
        sentences.append(
            f"⚠ The column is heavily imbalanced toward <b>{top_value}</b>, which may skew any "
            "analysis or model trained using it."
        )
    elif unique_count > 20:
        sentences.append(
            "This is a high-cardinality column, which can make it less useful for grouping or as a "
            "categorical feature without further processing (e.g. grouping rare values into \"Other\")."
        )
    else:
        sentences.append("Values appear reasonably distributed across categories.")

    return " ".join(sentences)


def generate_column_insight_text(df: pd.DataFrame, column: str) -> str:
    """
    The "AI-generated insight" shown under the Quick Visual Summary
    chart(s) - a short, deterministic read of whatever chart(s) are
    currently on screen for the selected column. Numeric columns get a
    histogram/box-plot-style read (range, shape, outliers); categorical
    columns get a bar-chart-style read (dominant category, balance).

    NOTE: like the rest of this file's "Smart Insights", this is
    computed directly from the data with pandas - not a call to an
    external LLM - so it's instant, free, and updates the moment the
    selected column changes.
    """
    if df is None or df.empty or not column or column not in df.columns:
        return ""

    try:
        if pd.api.types.is_numeric_dtype(df[column]):
            return _generate_numeric_column_insight(df, column)
        return _generate_categorical_column_insight(df, column)
    except Exception:
        return "An insight couldn't be generated for this column."


def generate_column_visual_summary_html(df: pd.DataFrame, column: str) -> str:
    """
    Generates a quick visual summary for a single selected column:
    histogram + box plot for numeric columns, or a top-categories bar
    chart for categorical columns - plus a one-line missing/unique
    stat header and a short AI-generated insight underneath the
    chart(s), summarizing distribution, highest/lowest values, and any
    possible anomalies. Chosen chart types depend on the column's
    actual dtype, same "only show what makes sense" principle as
    Section 4. The insight text regenerates every time a different
    column is selected, since it's a live description of whatever's
    currently on screen.
    """
    if df is None or df.empty or not column or column not in df.columns:
        return "<div class='kpi-empty'>Select a column above to see its visual summary.</div>"

    series = df[column]
    total_rows = len(df)
    missing_count = int(series.isnull().sum())
    missing_pct = round((missing_count / total_rows) * 100, 1) if total_rows > 0 else 0.0
    unique_count = int(series.nunique(dropna=True))

    summary_line = (
        f"<div class='meta-subsection-title'>{column} — {missing_count} missing "
        f"({missing_pct}%), {unique_count} unique value(s)</div>"
    )

    charts_html = ""
    try:
        if pd.api.types.is_numeric_dtype(series):
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(series.dropna(), bins=25, color="#6366f1", edgecolor="white")
            ax.set_title(f"Histogram — {column}")
            fig.tight_layout()
            charts_html += _chart_card(f"Histogram — {column}", _fig_to_data_uri(fig))

            fig2, ax2 = plt.subplots(figsize=(5, 3))
            ax2.boxplot(
                series.dropna(), patch_artist=True,
                boxprops=dict(facecolor="#8b5cf6", alpha=0.5),
                tick_labels=[column],
            )
            ax2.set_title(f"Box Plot — {column}")
            fig2.tight_layout()
            charts_html += _chart_card(f"Box Plot — {column}", _fig_to_data_uri(fig2))
        else:
            value_counts = series.value_counts().head(10)
            if not value_counts.empty:
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.barh(value_counts.index.astype(str), value_counts.values, color="#22d3ee")
                ax.set_title(f"Top Categories — {column}")
                ax.invert_yaxis()
                fig.tight_layout()
                charts_html += _chart_card(f"Top Categories — {column}", _fig_to_data_uri(fig))
    except Exception:
        pass

    if not charts_html:
        return summary_line + "<div class='kpi-empty'>No chart could be generated for this column.</div>"

    insight_text = generate_column_insight_text(df, column)
    insight_html = ""
    if insight_text:
        insight_html = (
            "<div class='chart-insight-box'>"
            "<div class='chart-insight-icon'>💡</div>"
            f"<div class='chart-insight-text'>{insight_text}</div>"
            "</div>"
        )

    return summary_line + f"<div class='chart-grid'>{charts_html}</div>" + insight_html
