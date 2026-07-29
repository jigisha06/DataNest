"""
visualization.py
-----------------
This module powers the "Visualization Studio" tab.

It is responsible for THREE things:
    1. Building the list of selectable features for each chart type,
       tagging numeric columns as:
           ⭐ Recommended         -> normal numeric columns
           ⚠ Less Recommended    -> numeric columns that look like an
                                     ID / Year / Month / Date / Day
                                     (these rarely make useful chart axes)
    2. Deciding which controls (Feature 1 / Feature 2) should be
       visible for the currently selected chart type.
    3. Actually drawing the chart with Matplotlib/Seaborn, ONLY when
       the user clicks "Generate Visualization" (never automatically).

Supported chart types:
    - Histogram             (needs: 1 numeric feature)
    - Bar Chart             (needs: 1 categorical feature (X-axis) + 1 numeric feature (Y-axis))
    - Scatter Plot          (needs: 2 numeric features)
    - Line Chart            (needs: 2 numeric features)
    - Box Plot              (needs: 1 numeric feature)
    - Correlation Heatmap   (needs: no feature selection - uses all numeric columns)
"""

import matplotlib
matplotlib.use("Agg")  # Headless backend - required since there's no display in this environment
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import gradio as gr


# Keywords that make a numeric column "less useful" as a chart axis
# (e.g. "customer_id", "order_year", "birth_month" ...)
LESS_RECOMMENDED_KEYWORDS = ["id", "year", "month", "date", "day"]

# The full list of chart types shown in the dropdown.
CHART_TYPES = [
    "Histogram",
    "Bar Chart",
    "Scatter Plot",
    "Line Chart",
    "Box Plot",
    "Correlation Heatmap",
]


# ----------------------------------------------------------------------
# FEATURE CHOICE HELPERS
# ----------------------------------------------------------------------

def _is_less_recommended(column_name: str) -> bool:
    """Checks if a column name suggests it's an ID/date-like field."""
    name_lower = column_name.lower()
    return any(keyword in name_lower for keyword in LESS_RECOMMENDED_KEYWORDS)


def get_numeric_feature_choices(df: pd.DataFrame):
    """
    Builds a list of (label, value) tuples for every numeric column,
    tagging ID/Year/Month/Date/Day-like columns as "Less Recommended".

    Gradio Dropdowns accept (display_label, actual_value) tuples, so the
    user sees the friendly label but we still get back the real column
    name when they pick one.
    """
    if df is None:
        return []

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    choices = []
    for col in numeric_columns:
        if _is_less_recommended(col):
            label = f"⚠ {col} (Less Recommended)"
        else:
            label = f"⭐ {col} (Recommended)"
        choices.append((label, col))

    return choices


def get_categorical_feature_choices(df: pd.DataFrame):
    """
    Builds a list of (label, value) tuples for every categorical column.
    No recommendation tagging is needed here — only numeric columns
    have the ID/Year/Month/Date/Day concern.
    """
    if df is None:
        return []

    categorical_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return [(col, col) for col in categorical_columns]


def get_feature_availability_warning(chart_type: str, df: pd.DataFrame) -> str:
    """
    Proactively explains why Feature 1/2 might be empty for the chosen
    chart type - BEFORE the user clicks Generate. Some datasets are
    text-only (e.g. a list of device names and serial numbers) and have
    no numeric columns at all, which makes several chart types
    impossible no matter what the user selects. Rather than leaving
    them to guess why a dropdown has nothing in it, we say so directly.

    Returns an empty string when everything needed is available (or no
    dataset is loaded yet, since there's nothing to warn about before
    upload).
    """
    if df is None:
        return ""

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    if chart_type == "Correlation Heatmap":
        if len(numeric_cols) < 2:
            return (
                f"⚠ This dataset has only **{len(numeric_cols)}** numeric column(s), but a "
                "Correlation Heatmap needs at least 2. Try a different chart type."
            )
        return ""

    if chart_type == "Bar Chart":
        if not categorical_cols and not numeric_cols:
            return "⚠ This dataset has no categorical or numeric columns, so no chart type will work here."
        if not categorical_cols:
            return "⚠ This dataset has no categorical (text) columns to use as Feature 1 (X-axis)."
        if not numeric_cols:
            return (
                "⚠ This dataset has **no numeric columns**, so Feature 2 (Y-axis) has nothing to offer. "
                "Bar Chart needs a number column (e.g. price, quantity, age) to average per category. "
                "Try uploading a dataset that includes at least one numeric column."
            )
        return ""

    # Histogram, Scatter Plot, Line Chart, Box Plot all need numeric columns.
    if not numeric_cols:
        return (
            "⚠ This dataset has **no numeric columns**, so Histogram, Scatter Plot, Line Chart, "
            "and Box Plot can't be used here. If your dataset has only text/ID columns "
            "(like this one), try **Bar Chart** instead if it has categorical columns — "
            "otherwise upload a dataset with at least one numeric column."
        )

    if chart_type in ["Scatter Plot", "Line Chart"] and len(numeric_cols) < 2:
        return (
            f"⚠ {chart_type} needs 2 numeric columns, but this dataset only has "
            f"**{len(numeric_cols)}**. Try Histogram or Box Plot instead."
        )

    return ""


def get_feature_dropdown_updates(chart_type: str, df: pd.DataFrame):
    """
    Decides what Feature 1 and Feature 2 dropdowns should look like
    for the currently selected chart type.

    IMPORTANT: both dropdowns stay `visible=True` at all times. We only
    toggle `interactive` (enabled/disabled) and swap their `choices`.
    Toggling `visible` on a component that lives inside a `gr.Row`
    can, in some Gradio versions, leave it stuck hidden even after a
    later update sets visible=True again - this keeps the layout
    stable (always 3 columns) and avoids that class of bug entirely.

    Returns
    -------
    (feature1_update, feature2_update) : tuple of gr.update(...) objects
    """

    if chart_type == "Correlation Heatmap":
        # No manual feature selection needed - it uses ALL numeric columns.
        # Both dropdowns stay visible but disabled, with a label explaining why.
        feature1_update = gr.update(
            choices=[], value=None, visible=True, interactive=False,
            label="Feature 1 (not needed - uses all numeric columns)",
        )
        feature2_update = gr.update(
            choices=[], value=None, visible=True, interactive=False,
            label="Feature 2 (not needed - uses all numeric columns)",
        )
        return feature1_update, feature2_update

    if chart_type == "Bar Chart":
        # Bar Chart works like Scatter Plot: Feature 1 = X-axis
        # (categorical), Feature 2 = Y-axis (numeric). Both are shown,
        # enabled, and required - the average of Feature 2 is plotted
        # per category of Feature 1.
        categorical_choices = get_categorical_feature_choices(df)
        numeric_choices = get_numeric_feature_choices(df)
        feature1_update = gr.update(
            choices=categorical_choices, value=None, visible=True, interactive=True,
            label="Feature 1 (X-axis - Categorical)",
        )
        feature2_update = gr.update(
            choices=numeric_choices, value=None, visible=True, interactive=True,
            label="Feature 2 (Y-axis - Numeric)",
        )
        return feature1_update, feature2_update

    # Histogram, Scatter Plot, Line Chart, Box Plot all use numeric columns.
    numeric_choices = get_numeric_feature_choices(df)
    feature1_update = gr.update(
        choices=numeric_choices, value=None, visible=True, interactive=True,
        label="Feature 1 (Numeric Column)",
    )

    if chart_type in ["Scatter Plot", "Line Chart"]:
        # These two chart types need a SECOND numeric feature.
        feature2_update = gr.update(
            choices=numeric_choices, value=None, visible=True, interactive=True,
            label="Feature 2 (Numeric Column)",
        )
    else:
        # Histogram / Box Plot only need one feature - Feature 2 stays
        # visible but disabled, so the layout never jumps around.
        feature2_update = gr.update(
            choices=[], value=None, visible=True, interactive=False,
            label="Feature 2 (not needed for this chart)",
        )

    return feature1_update, feature2_update


# ----------------------------------------------------------------------
# CHART GENERATION
# ----------------------------------------------------------------------

def generate_visualization(df: pd.DataFrame, chart_type: str, feature1: str, feature2: str):
    """
    Builds the requested Matplotlib chart. This function is ONLY called
    when the user clicks "Generate Visualization" - never automatically.

    Returns
    -------
    (figure, status_message) : tuple
        figure is a Matplotlib Figure (or None if we couldn't build one)
        status_message explains success or why it failed.
    """

    # ---- Guard clauses (validate inputs before touching Matplotlib) ----
    if df is None:
        return None, "⚠ Please upload and load a dataset first."

    if chart_type is None:
        return None, "⚠ Please select a chart type."

    if chart_type != "Correlation Heatmap" and not feature1:
        return None, "⚠ Please select Feature 1 to generate this chart."

    if chart_type in ["Scatter Plot", "Line Chart", "Bar Chart"] and not feature2:
        return None, "⚠ Please select Feature 2 as well - this chart needs two features."

    try:
        fig, ax = plt.subplots(figsize=(8, 5))

        if chart_type == "Histogram":
            ax.hist(df[feature1].dropna(), bins=30, color="#4C72B0", edgecolor="white")
            ax.set_title(f"Histogram of {feature1}")
            ax.set_xlabel(feature1)
            ax.set_ylabel("Frequency")

        elif chart_type == "Bar Chart":
            # X-axis = categorical Feature 1, Y-axis = numeric Feature 2.
            # We plot the AVERAGE of Feature 2 for each category of
            # Feature 1 - the standard way to compare a numeric metric
            # across categories. Limited to the top 20 categories (by
            # average value) so the chart stays readable.
            grouped = (
                df.groupby(feature1)[feature2]
                .mean()
                .sort_values(ascending=False)
                .head(20)
            )
            ax.bar(grouped.index.astype(str), grouped.values, color="#55A868")
            ax.set_title(f"Bar Chart: Average {feature2} by {feature1}")
            ax.set_xlabel(feature1)
            ax.set_ylabel(f"Average {feature2}")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        elif chart_type == "Scatter Plot":
            ax.scatter(df[feature1], df[feature2], alpha=0.6, color="#C44E52")
            ax.set_title(f"Scatter Plot: {feature1} vs {feature2}")
            ax.set_xlabel(feature1)
            ax.set_ylabel(feature2)

        elif chart_type == "Line Chart":
            # Sort by Feature 1 so the line reads left-to-right sensibly.
            sorted_df = df[[feature1, feature2]].dropna().sort_values(by=feature1)
            ax.plot(sorted_df[feature1], sorted_df[feature2], color="#8172B2", marker="o", markersize=3)
            ax.set_title(f"Line Chart: {feature1} vs {feature2}")
            ax.set_xlabel(feature1)
            ax.set_ylabel(feature2)

        elif chart_type == "Box Plot":
            ax.boxplot(df[feature1].dropna(), vert=True, patch_artist=True,
                       boxprops=dict(facecolor="#64B5CD"))
            ax.set_title(f"Box Plot of {feature1}")
            ax.set_ylabel(feature1)
            ax.set_xticklabels([feature1])

        elif chart_type == "Correlation Heatmap":
            numeric_df = df.select_dtypes(include=["number"])
            if numeric_df.shape[1] < 2:
                plt.close(fig)
                return None, "⚠ Need at least 2 numeric columns to build a correlation heatmap."
            correlation_matrix = numeric_df.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            ax.set_title("Correlation Heatmap")

        else:
            plt.close(fig)
            return None, f"❌ Unknown chart type: {chart_type}"

        fig.tight_layout()
        return fig, "✅ Visualization generated successfully."

    except Exception as e:
        plt.close("all")
        return None, f"❌ Error while generating chart: {str(e)}"


# ----------------------------------------------------------------------
# PIVOT TABLE BUILDER (Excel-style)
# ----------------------------------------------------------------------
# Lets the user drag-and-drop-style configure Rows / Columns / Values /
# Aggregation / an optional Filter, exactly like a classic Excel pivot
# table - built dynamically off whatever fields are selected, and only
# recomputed when the user clicks "Generate Pivot Table".

PIVOT_AGGREGATIONS = ["Sum", "Count", "Average", "Min", "Max", "Median", "Std Dev", "Count Distinct"]

_PIVOT_AGG_FUNC_MAP = {
    "Sum": "sum",
    "Count": "count",
    "Average": "mean",
    "Min": "min",
    "Max": "max",
    "Median": "median",
    "Std Dev": "std",
    "Count Distinct": "nunique",
}

# Aggregations that only make sense on real numbers - Count and Count
# Distinct work fine on text/categorical columns too.
_PIVOT_NUMERIC_ONLY_AGGS = {"Sum", "Average", "Min", "Max", "Median", "Std Dev"}


# A Columns field with more distinct values than this explodes the
# pivot table into one column per value (e.g. a raw numeric reading
# with 199 unique values -> a 199-column table nobody can read).
_PIVOT_COLUMNS_CARDINALITY_LIMIT = 15


def get_pivot_field_choices(df: pd.DataFrame):
    """Every column is a valid Row / Value / Filter field candidate -
    unlike the chart builder above, a pivot table can group or count
    by text columns just as easily as numeric ones."""
    if df is None:
        return []
    return df.columns.tolist()


def get_pivot_column_field_choices(df: pd.DataFrame):
    """
    Same idea as get_pivot_field_choices(), but tags each field for
    the Columns slot specifically: fields with a lot of distinct
    values (e.g. a continuous numeric reading) get an ⚠ warning label,
    since spreading a pivot table across that many columns produces an
    unreadable result. Low-cardinality fields (categories, years,
    flags, ...) get a ⭐ label instead - the same ⭐/⚠ convention
    already used for chart axes above.
    """
    if df is None:
        return []

    choices = []
    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        if nunique > _PIVOT_COLUMNS_CARDINALITY_LIMIT:
            label = f"⚠ {col} ({nunique} unique values)"
        else:
            label = f"⭐ {col}"
        choices.append((label, col))
    return choices


def get_pivot_filter_value_choices(df: pd.DataFrame, filter_column: str):
    """
    Populates the Filter Values picker with the actual distinct values
    found in the chosen Filter column - the same experience as opening
    the checklist under Excel's filter dropdown.
    """
    if df is None or not filter_column or filter_column not in df.columns:
        return []

    unique_values = df[filter_column].dropna().unique().tolist()
    try:
        unique_values = sorted(unique_values, key=lambda v: str(v))
    except TypeError:
        pass

    # Cap the list so a high-cardinality column (e.g. a free-text or ID
    # field) doesn't render an unusable, thousands-of-entries dropdown.
    return [str(v) for v in unique_values[:500]]


def get_pivot_dropdown_updates(df: pd.DataFrame):
    """
    Refreshes every Pivot Table Builder control for a newly loaded (or
    cleared) dataset - Rows / Columns / Values reset to the new column
    list with nothing pre-selected, and the Filter column/value pickers
    reset too, so no stale selections from a previous dataset linger.
    The Columns field specifically gets the ⭐/⚠ cardinality-tagged
    choice list so users see the warning before they ever pick a field.

    Returns
    -------
    (rows_update, columns_update, values_update, filter_column_update,
     filter_value_update) : tuple of gr.update(...) objects
    """
    field_choices = get_pivot_field_choices(df)
    column_field_choices = get_pivot_column_field_choices(df)
    rows_update = gr.update(choices=field_choices, value=[])
    columns_update = gr.update(choices=column_field_choices, value=[])
    values_update = gr.update(choices=field_choices, value=[])
    filter_column_update = gr.update(choices=field_choices, value=None)
    filter_value_update = gr.update(choices=[], value=[])
    return rows_update, columns_update, values_update, filter_column_update, filter_value_update


def generate_pivot_table(
    df: pd.DataFrame,
    rows: list,
    columns: list,
    values: list,
    aggregation: str,
    filter_column: str,
    filter_values: list,
):
    """
    Builds an Excel-style pivot table: Rows group the data down the
    left side, Columns (optional) spread values across the top, Values
    are the number(s) being summarized, and Aggregation decides how
    (Sum / Count / Average / Min / Max / Median / Std Dev / Count
    Distinct). An optional Filter narrows the source data first -
    exactly like dragging a field into Excel's "Filters" box and
    checking only the values you want.

    This function is ONLY called when the user clicks "Generate Pivot
    Table" - never automatically, matching the chart builder above it.

    Returns
    -------
    (pivot_df, status_message) : tuple
        pivot_df is a flat pandas.DataFrame ready for gr.Dataframe (or
        None if the pivot couldn't be built); status_message explains
        success or why it failed.
    """
    if df is None:
        return None, "⚠ Please upload and load a dataset first."

    rows = [r for r in (rows or []) if r]
    columns = [c for c in (columns or []) if c]
    values = [v for v in (values or []) if v]

    if not rows:
        return None, "⚠ Please select at least one field for **Rows**."
    if not values:
        return None, "⚠ Please select at least one field for **Values**."
    if not aggregation:
        return None, "⚠ Please choose an aggregation (Sum, Count, Average, ...)."

    agg_func = _PIVOT_AGG_FUNC_MAP.get(aggregation)
    if agg_func is None:
        return None, f"⚠ Unknown aggregation: {aggregation}."

    working_df = df

    # ---- Optional filter: keep only rows whose filter column matches
    #      one of the checked filter values ----
    if filter_column and filter_values:
        if filter_column not in working_df.columns:
            return None, f"⚠ Filter column **{filter_column}** was not found in this dataset."
        working_df = working_df[working_df[filter_column].astype(str).isin([str(v) for v in filter_values])]
        if working_df.empty:
            return None, "⚠ No rows match the selected filter - try different filter values."

    # ---- Aggregations that need real numbers require numeric Value field(s) ----
    if aggregation in _PIVOT_NUMERIC_ONLY_AGGS:
        non_numeric = [v for v in values if not pd.api.types.is_numeric_dtype(working_df[v])]
        if non_numeric:
            return None, (
                f"⚠ **{aggregation}** needs numeric column(s), but "
                f"{', '.join(non_numeric)} is not numeric. Try **Count** or **Count Distinct** "
                "instead, or pick a numeric Value field."
            )

    try:
        pivot_df = pd.pivot_table(
            working_df,
            index=rows,
            columns=columns if columns else None,
            values=values,
            aggfunc=agg_func,
            fill_value=0 if aggregation in ("Sum", "Count", "Count Distinct") else None,
            dropna=False,
        )
    except Exception as e:
        return None, f"❌ Couldn't build the pivot table: {str(e)}"

    if pivot_df.empty:
        return None, "⚠ The pivot table came back empty - try different fields or filters."

    # ---- Flatten MultiIndex columns (happens whenever a Column field
    #      is used, or multiple Value fields are selected) into single,
    #      readable headers instead of nested tuples. ----
    if isinstance(pivot_df.columns, pd.MultiIndex):
        pivot_df.columns = [
            " | ".join(str(part) for part in col if str(part) != "")
            for col in pivot_df.columns.to_flat_index()
        ]
    else:
        pivot_df.columns = [str(c) for c in pivot_df.columns]

    pivot_df = pivot_df.reset_index()

    # Round floats for a cleaner, Excel-style display.
    float_cols = pivot_df.select_dtypes(include=["float"]).columns
    if len(float_cols):
        pivot_df[float_cols] = pivot_df[float_cols].round(2)

    result_rows, result_cols = pivot_df.shape
    status = f"✅ Pivot table generated — **{result_rows}** row(s) × **{result_cols}** column(s)."
    return pivot_df, status
