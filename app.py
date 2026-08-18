"""
app.py
------
Main entry point for DataNest (formerly "AI Data Analyst Agent" / "Prism Analytics").

PART 1: Project setup + File upload (CSV / Excel) + Dashboard tab
PART 2: Dataset Metadata tab (column name, dtype, missing, unique values)
PART 3: AI Readiness tab (0-100 quality score + status label)
PART 4: Visualization Studio (interactive chart builder)
PART 5: Feedback tab + About tab (feedback.csv storage + aggregated stats)
PART 6: Feedback tab UI redesign (card layout) + Persistent Upload History
PART 7: Dashboard KPI cards + Bar Chart X/Y fix + AI Dataset Chat tab
PART 8: Data Cleaning tab (missing values, duplicates, outliers) + Export
PART 9: Full visual redesign - premium SaaS look, rebrand, no logic changes

Future modules (Feature Engineering, Advanced Visualizations) will be
added on top of this foundation — the app stays 100% runnable after
every step.

Run with:
    python app.py
"""

import os
import textwrap
import functools
import gradio as gr

# python-dotenv lets you keep OPENAI_API_KEY (and any other secrets) in
# a local .env file instead of exporting it manually every session.
# This is entirely optional - if python-dotenv isn't installed, or
# there's no .env file, the app just falls back to whatever is already
# in the real environment variables (or runs in rule-based mode).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Our own modules (this is what keeps the project modular)
from modules.data_handler import load_dataset
from modules.dashboard import get_dashboard_stats, format_stats_as_markdown, format_stats_as_kpi_cards, get_preview
from modules.metadata import (
    get_metadata,
    render_metadata_dashboard,
    render_metadata_top_html,
    render_metadata_bottom_html,
    get_business_metric_columns,
    get_default_business_metrics,
    get_bi_kpis,
    format_bi_kpis_html,
    generate_column_visual_summary_html,
)

from modules.visualization import (
    CHART_TYPES,
    get_feature_dropdown_updates,
    get_feature_availability_warning,
    get_numeric_feature_choices,
    generate_visualization,
    PIVOT_AGGREGATIONS,
    get_pivot_dropdown_updates,
    get_pivot_filter_value_choices,
    generate_pivot_table,
)
from modules.feedback import (
    submit_feedback,
    get_feedback_stats,
    format_about_markdown,
    format_about_intro_markdown,
    format_about_kpi_cards,
)
from modules.upload_history import log_upload, get_upload_history
from modules.ai_chat import answer_question, is_openai_available, get_example_questions_markdown
from modules.data_cleaning import (
    MISSING_VALUE_STRATEGIES,
    OUTLIER_STRATEGIES,
    get_missing_summary,
    get_missing_value_stats,
    format_missing_value_stats_html,
    search_missing_summary,
    detect_outliers_iqr,
    clean_dataset,
    export_dataset,
    preview_cleaning_plan,
)


# ----------------------------------------------------------------------
# CALLBACK FUNCTIONS
# These functions connect the UI (buttons/inputs) to our modules.
# ----------------------------------------------------------------------

def handle_file_upload(file):
    """
    Triggered when the user clicks "Load Dataset".

    Returns (in this order, matching the `outputs` list in the button click event):
        1. status_message     -> text shown to the user (success/error)
        2. dashboard_state     -> the DataFrame stored in memory for reuse across tabs
        3. dashboard_md        -> formatted markdown summary for the Dashboard tab
        4. preview_table       -> first 10 rows of the dataset
        5. metadata_top_html   -> Metadata tab: Statistical Summary + Visual Analytics
        6. metric_dropdown     -> repopulates the "Select Metric to Analyze" dropdown
        7. bi_kpis_html        -> Metadata tab: initial Business Intelligence KPI cards
        8. metadata_bottom_html-> Metadata tab: Smart AI Insights
        9. chart_type_reset    -> resets the chart type dropdown to "Histogram"
        10. feature1_reset      -> resets/repopulates the Feature 1 dropdown
        11. feature2_reset      -> resets/repopulates the Feature 2 dropdown (hidden by default)
        12. viz_plot_clear     -> clears any previously generated chart
        13. viz_status_clear   -> clears any previous visualization status message
        14. upload_history_df  -> refreshed upload history table (latest upload on top)
        15. chatbot_reset      -> clears the AI Dataset Chat display (new dataset = fresh chat)
        16. chat_history_reset -> clears the AI Dataset Chat session memory
        17. missing_summary_df -> per-column missing-value table for the Data Cleaning tab
        18. outlier_col_reset  -> repopulates the outlier-detection column dropdown
        19. cleaned_state_reset-> clears any previously cleaned dataset (new upload = start fresh)
        20. clean_summary_clear-> clears the previous cleaning summary text
        21. clean_preview_clear-> clears the previous cleaned-data preview table
        22. export_file_clear  -> hides any previous export download chip (new upload = start fresh)
        23. viz_availability_warning -> proactive warning if this dataset lacks what
                                          the default "Histogram" chart type needs
        24. chips_reset       -> brings the VizBot suggestion chips back for the new dataset
        25. visual_summary_dropdown -> repopulates the Quick Visual Summary column selector
        26. visual_summary_html -> Metadata tab: initial Quick Visual Summary chart(s)
        27. cleaning_plan_html -> Cleaning tab: initial "planned changes" summary
        28. pivot_rows_reset   -> resets the Pivot Table Builder's Rows field
        29. pivot_columns_reset -> resets the Pivot Table Builder's Columns field
        30. pivot_values_reset -> resets the Pivot Table Builder's Values field
        31. pivot_filter_column_reset -> resets the Pivot Table Builder's Filter field
        32. pivot_filter_value_reset -> resets the Pivot Table Builder's Filter Values field
        33. pivot_table_clear  -> clears any previously generated pivot table
        34. pivot_status_clear -> clears any previous pivot table status message
        35. missing_stats_html -> Cleaning tab: compact missing-value stat row
        36. missing_search_reset -> clears the missing-values search box for the new dataset
        37. export_banner_reset -> hides the "your cleaned dataset is ready" banner
    """
    file_path = file.name if file is not None else None
    df, message = load_dataset(file_path)

    # Whenever a new file is loaded, the Visualization Studio controls
    # should reset back to a clean "Histogram" starting state so old
    # feature selections from a previous dataset are never reused.
    # The AI Dataset Chat session and Data Cleaning results are reset
    # too, since both would be misleading if left over from a
    # previous, different dataset.
    default_chart_type = "Histogram"

    if df is None:
        # Loading failed — clear everything and show the error.
        # We do NOT log a history entry for a failed upload.
        empty_dashboard_html = format_stats_as_kpi_cards({})
        empty_metadata_top = render_metadata_top_html(None)
        empty_metric_dropdown = gr.update(choices=[], value=None)
        empty_bi_kpis = format_bi_kpis_html({})
        empty_metadata_bottom = render_metadata_bottom_html(None)
        feature1_update, feature2_update = get_feature_dropdown_updates(default_chart_type, None)
        history_df = get_upload_history()
        empty_missing_summary = get_missing_summary(None)
        empty_visual_summary_dropdown = gr.update(choices=[], value=None)
        empty_visual_summary_html = generate_column_visual_summary_html(None, None)
        empty_cleaning_plan = preview_cleaning_plan(None, "Do Nothing", "Do Nothing", None)
        empty_missing_stats_html = format_missing_value_stats_html(get_missing_value_stats(None))
        (
            empty_pivot_rows, empty_pivot_columns, empty_pivot_values,
            empty_pivot_filter_column, empty_pivot_filter_value,
        ) = get_pivot_dropdown_updates(None)
        return (
            message, None, empty_dashboard_html, None,
            empty_metadata_top, empty_metric_dropdown, empty_bi_kpis, empty_metadata_bottom,
            gr.update(value=default_chart_type), feature1_update, feature2_update, None, "",
            history_df, [], [],
            empty_missing_summary, gr.update(choices=[], value=None), None, "",
            None, gr.update(value=None, visible=False),
            "", gr.update(visible=True),
            empty_visual_summary_dropdown, empty_visual_summary_html, empty_cleaning_plan,
            empty_pivot_rows, empty_pivot_columns, empty_pivot_values,
            empty_pivot_filter_column, empty_pivot_filter_value, None, "",
            empty_missing_stats_html, "", gr.update(visible=False),
        )

    stats = get_dashboard_stats(df)
    dashboard_html = format_stats_as_kpi_cards(stats)
    preview_df = get_preview(df, n_rows=10)

    # ---- Metadata tab: top section, interactive BI KPIs, bottom section ----
    metadata_top_html = render_metadata_top_html(df)
    business_metric_choices = get_business_metric_columns(df)
    default_metrics = get_default_business_metrics(df, top_n=1)
    default_metric = default_metrics[0] if default_metrics else None
    metric_dropdown_update = gr.update(choices=business_metric_choices, value=default_metric)
    bi_kpis_html = format_bi_kpis_html(get_bi_kpis(df, columns=default_metrics if default_metrics else None))
    metadata_bottom_html = render_metadata_bottom_html(df)

    feature1_update, feature2_update = get_feature_dropdown_updates(default_chart_type, df)
    viz_warning = get_feature_availability_warning(default_chart_type, df)

    # ---- Log this upload as a brand-new history entry (never overwritten) ----
    original_filename = os.path.basename(file_path) if file_path else "unknown_file"
    total_rows, total_columns = df.shape
    log_upload(filename=original_filename, rows=total_rows, columns=total_columns)
    history_df = get_upload_history()

    # ---- Data Cleaning tab: fresh missing-value summary + outlier column choices ----
    missing_summary_df = get_missing_summary(df)
    missing_stats_html = format_missing_value_stats_html(get_missing_value_stats(df))
    outlier_column_update = gr.update(choices=get_numeric_feature_choices(df), value=None)
    cleaning_plan_html = preview_cleaning_plan(df, "Do Nothing", "Do Nothing", None)

    # ---- Metadata tab: Quick Visual Summary column selector ----
    all_columns = df.columns.tolist()
    default_visual_column = all_columns[0] if all_columns else None
    visual_summary_dropdown_update = gr.update(choices=all_columns, value=default_visual_column)
    visual_summary_html = generate_column_visual_summary_html(df, default_visual_column)

    # ---- Visualization Studio: fresh Pivot Table Builder fields for the new dataset ----
    (
        pivot_rows_update, pivot_columns_update, pivot_values_update,
        pivot_filter_column_update, pivot_filter_value_update,
    ) = get_pivot_dropdown_updates(df)

    return (
        message, df, dashboard_html, preview_df,
        metadata_top_html, metric_dropdown_update, bi_kpis_html, metadata_bottom_html,
        gr.update(value=default_chart_type), feature1_update, feature2_update, None, "",
        history_df, [], [],
        missing_summary_df, outlier_column_update, None, "",
        None, gr.update(value=None, visible=False),
        viz_warning, gr.update(visible=True),
        visual_summary_dropdown_update, visual_summary_html, cleaning_plan_html,
        pivot_rows_update, pivot_columns_update, pivot_values_update,
        pivot_filter_column_update, pivot_filter_value_update, None, "",
        missing_stats_html, "", gr.update(visible=False),
    )


def handle_feedback_submit(rating, ai_helpful, viz_useful, would_recommend, suggestions):
    """
    Triggered when the user clicks "Submit Feedback".

    Saves the feedback to feedback.csv, then immediately refreshes the
    About tab so the community stats always reflect the latest data
    without requiring a separate manual refresh.

    Returns (in this order):
        1. feedback_status   -> confirmation/error message
        2. about_md          -> refreshed About tab markdown
        3. latest_entries_df -> refreshed "latest 5 entries" table
        4-8. Reset values for the 5 feedback form fields, so the form
             is clean and ready for the next submission.
    """
    status_message = submit_feedback(rating, ai_helpful, viz_useful, would_recommend, suggestions)

    stats, latest_entries = get_feedback_stats()
    about_html = format_about_kpi_cards(stats)

    return (
        status_message, about_html, latest_entries,
        gr.update(value=None),  # reset rating
        gr.update(value=None),  # reset AI helpful
        gr.update(value=None),  # reset viz useful
        gr.update(value=None),  # reset would recommend
        gr.update(value=""),    # reset suggestions
    )


def handle_about_refresh():
    """
    Triggered when the user clicks "Refresh Stats" on the About tab.
    Simply re-reads feedback.csv and recalculates everything.
    """
    stats, latest_entries = get_feedback_stats()
    about_html = format_about_kpi_cards(stats)
    return about_html, latest_entries


def handle_chat_message(question, chat_history, df):
    """
    Triggered when the user sends a message in the AI Dataset Chat tab
    (whether typed directly or sent via a suggestion chip).

    Every answer comes from modules.ai_chat.answer_question(), which
    always returns a real, grounded answer - either from offline
    pandas analysis or (if configured) OpenAI-assisted routing - and
    never raises, so this handler can't crash the chat.

    Parameters
    ----------
    question : str            the text the user just typed (or a
                                 suggestion chip's preset prompt)
    chat_history : list        session-level chat memory: a list of
                                 {"role": ..., "content": ...} dicts,
                                 the format Gradio's Chatbot requires.
                                 Kept in a gr.State so it persists
                                 across turns for this browser session.
    df : pandas.DataFrame or None   the currently loaded dataset

    Returns
    -------
    (updated_chatbot_display, updated_chat_history_state, cleared_input,
     suggestion_chips_visibility_update)
    """
    answer = answer_question(question, df)

    # Gradio's Chatbot expects a list of dicts with 'role' and
    # 'content' keys - NOT the old [user, ai] tuple format, which
    # raises "Data incompatible with messages format".
    chat_history = chat_history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    # Once the user has sent a message (typed or via a chip), the
    # suggestion chips are no longer needed - hide them so the chat
    # window stays the clear focus, matching modern AI chat products.
    return chat_history, chat_history, "", gr.update(visible=False)


def handle_chat_clear():
    """Triggered by the 'Clear Chat' button - wipes session memory and
    brings the suggestion chips back, since the conversation is fresh again."""
    return [], [], "", gr.update(visible=True)


def handle_metric_change(selected_metric, df):
    """
    Triggered when the user picks a different column in the Metadata
    tab's "Select Metric to Analyze" dropdown - recomputes ONLY the
    Business Intelligence KPI cards for that one column, without
    touching the rest of the Metadata report.
    """
    return format_bi_kpis_html(get_bi_kpis(df, columns=selected_metric))


def handle_visual_summary_change(selected_column, df):
    """
    Triggered when the user picks a column in the Metadata tab's Quick
    Visual Summary selector - regenerates just that column's chart(s)
    (histogram+box plot for numeric, top-categories bar for
    categorical) without touching anything else on the page.
    """
    return generate_column_visual_summary_html(df, selected_column)


def handle_chart_type_change(chart_type, df):
    """
    Triggered whenever the user picks a different Chart Type in
    Visualization Studio. Updates the Feature 1/2 dropdowns AND a
    proactive warning message explaining upfront if this dataset is
    missing what that chart type needs (e.g. no numeric columns at
    all) - instead of leaving the user to guess why a dropdown is empty.
    """
    feature1_update, feature2_update = get_feature_dropdown_updates(chart_type, df)
    warning_message = get_feature_availability_warning(chart_type, df)
    return feature1_update, feature2_update, warning_message


def handle_pivot_filter_column_change(df, filter_column):
    """
    Triggered when the user picks a different Filter field in the
    Pivot Table Builder - repopulates the Filter Values picker with
    that column's actual distinct values (like opening the checklist
    under an Excel filter dropdown).
    """
    choices = get_pivot_filter_value_choices(df, filter_column)
    return gr.update(choices=choices, value=[])


def handle_generate_pivot(df, rows, columns, values, aggregation, filter_column, filter_values):
    """
    Triggered by the "Generate Pivot Table" button - builds the pivot
    table ONLY on click, never automatically, matching the philosophy
    of the chart builder above it in the same tab.
    """
    pivot_df, status_message = generate_pivot_table(
        df, rows, columns, values, aggregation, filter_column, filter_values
    )
    return pivot_df, status_message


def handle_detect_outliers(df, column):
    """
    Triggered by the "Detect Outliers" button. Read-only - does NOT
    modify the dataset, just reports what the IQR method would flag.
    """
    _, _, _, message = detect_outliers_iqr(df, column)
    return message


def handle_cleaning_plan_preview(df, missing_strategy, outlier_strategy, outlier_column):
    """
    Triggered whenever any Cleaning tab control changes - shows a live
    "here's what will happen" summary WITHOUT actually cleaning
    anything. Purely a preview; the real cleaning only happens when
    "Apply Cleaning" is clicked.
    """
    return preview_cleaning_plan(df, missing_strategy, outlier_strategy, outlier_column)


def handle_missing_search(df, search_term):
    """
    Triggered as the user types in the missing-values search box -
    filters the table to columns whose name matches, without touching
    the underlying data or the mini-stat row above it (those describe
    the WHOLE dataset regardless of what's currently filtered/visible).
    """
    return search_missing_summary(df, search_term)


def handle_apply_cleaning(df, missing_strategy, outlier_strategy, outlier_column):
    """
    Triggered by the "Apply Cleaning" button. Runs the full cleaning
    pipeline and stores the result in cleaned_dataset_state - the
    ORIGINAL dataset (used by Dashboard/Visualization/AI Chat) is never
    touched, so cleaning is always non-destructive and undoable by
    simply re-loading the file. Also refreshes the missing-value mini-
    stats (the numbers may have changed) and hides any previous export
    download, since it no longer matches the freshly cleaned data.

    Returns
    -------
    (cleaned_df_for_state, summary_markdown, cleaned_preview_df,
     missing_summary_df, missing_stats_html, export_banner_hide, export_file_hide)
    """
    cleaned_df, summary_md, missing_df = clean_dataset(
        df, missing_strategy, outlier_strategy, outlier_column
    )
    hide_banner = gr.update(visible=False)
    hide_file = gr.update(value=None, visible=False)

    if cleaned_df is None:
        empty_stats_html = format_missing_value_stats_html(get_missing_value_stats(None))
        return None, summary_md, None, get_missing_summary(None), empty_stats_html, hide_banner, hide_file

    preview_df = cleaned_df.head(10)
    stats_html = format_missing_value_stats_html(get_missing_value_stats(cleaned_df))
    return cleaned_df, summary_md, preview_df, missing_df, stats_html, hide_banner, hide_file


def handle_export_csv(cleaned_df):
    """
    Triggered by the "CSV" export option. On success, hides the plain
    button row's need for explanation and instead shows a compact
    success banner + the real download chip - no dashed empty
    drag-and-drop box is ever shown, since the File component stays
    hidden until there's an actual file to hand over.
    """
    if cleaned_df is None:
        gr.Warning("Please click 'Apply Cleaning' first - there's no cleaned dataset to export yet.")
        return gr.update(visible=False), gr.update(value=None, visible=False)
    file_path = export_dataset(cleaned_df, "CSV (.csv)")
    banner_html = (
        "<div class='export-success-banner-inner'>"
        "<span class='export-success-icon'>✅</span>"
        "<span>Your cleaned dataset is ready — CSV format.</span>"
        "</div>"
    )
    return gr.update(value=banner_html, visible=True), gr.update(value=file_path, visible=True)


def handle_export_excel(cleaned_df):
    """Triggered by the "Excel" export option - same flow as CSV, above."""
    if cleaned_df is None:
        gr.Warning("Please click 'Apply Cleaning' first - there's no cleaned dataset to export yet.")
        return gr.update(visible=False), gr.update(value=None, visible=False)
    file_path = export_dataset(cleaned_df, "Excel (.xlsx)")
    banner_html = (
        "<div class='export-success-banner-inner'>"
        "<span class='export-success-icon'>✅</span>"
        "<span>Your cleaned dataset is ready — Excel format.</span>"
        "</div>"
    )
    return gr.update(value=banner_html, visible=True), gr.update(value=file_path, visible=True)


# ----------------------------------------------------------------------
# CUSTOM CSS
# Full visual redesign: soft glass-style cards, an indigo/purple/slate
# accent palette (deliberately avoiding an all-blue look), a styled
# upload zone, pill-style tabs, hover animations, and the AI Readiness
# gauge card. Nothing here touches app logic - only appearance.
# ----------------------------------------------------------------------
CUSTOM_CSS = """
:root {
    /* Brand accents - identical in both themes; this is what keeps the
       product recognizable regardless of light/dark. */
    --pa-indigo: #6366f1;
    --pa-purple: #8b5cf6;
    --pa-cyan: #22d3ee;
    --pa-blue-accent: #3b82f6;
    --pa-charcoal: #1e1b2e;
    --pa-slate: #64748b;
    --pa-success: #22c55e;
    --pa-danger: #f43f5e;
    --pa-warning: #f59e0b;

    /* Tooltip surface stays intentionally dark in BOTH themes (a
       common, always-readable pattern), so it gets its own fixed
       colors instead of following --pa-text-primary / --pa-card-bg. */
    --pa-tooltip-bg: #12131c;
    --pa-tooltip-text: #f1f5f9;

    --pa-radius-lg: 18px;
    --pa-radius-md: 14px;
    --pa-radius-sm: 10px;

    /* ============================================================
       DARK THEME (default - matches the app's original look)
       ============================================================ */
    --pa-bg: #0a0a12;
    --pa-bg-elevated: #131320;
    --pa-card-bg: rgba(255, 255, 255, 0.045);
    --pa-card-bg-solid: #211f36;
    --pa-card-bg-secondary: rgba(255, 255, 255, 0.03);
    --pa-subtle-fill: rgba(255, 255, 255, 0.035);
    --pa-border: rgba(255, 255, 255, 0.09);
    --pa-border-strong: rgba(255, 255, 255, 0.18);
    --pa-text-primary: #f1f5f9;
    --pa-text-secondary: rgba(241, 245, 249, 0.7);
    --pa-text-muted: rgba(241, 245, 249, 0.5);
    --pa-text-disabled: rgba(241, 245, 249, 0.3);

    --readiness-track-color: rgba(148, 163, 184, 0.22);
    --pa-shadow-resting: 0 1px 2px rgba(0, 0, 0, 0.2), 0 2px 10px rgba(0, 0, 0, 0.16);
    --pa-shadow-soft: 0 4px 20px rgba(0, 0, 0, 0.18);
    --pa-shadow-hover: 0 10px 30px rgba(99, 102, 241, 0.22);

    --pa-table-header-bg: rgba(255, 255, 255, 0.05);
    --pa-table-row-alt-bg: rgba(255, 255, 255, 0.02);
    --pa-table-row-hover-bg: rgba(99, 102, 241, 0.12);
    --pa-table-border: var(--pa-border);
    --pa-table-selected-bg: rgba(99, 102, 241, 0.18);

    --pa-btn-primary-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
    --pa-btn-secondary-bg: rgba(255, 255, 255, 0.05);
    --pa-btn-secondary-border: var(--pa-border);
    --pa-btn-disabled-bg: rgba(255, 255, 255, 0.03);
    --pa-btn-disabled-text: rgba(241, 245, 249, 0.28);

    --pa-scrollbar-thumb: rgba(99, 102, 241, 0.3);
    --pa-scrollbar-thumb-hover: rgba(99, 102, 241, 0.5);
    --pa-scrollbar-track: transparent;

    --pa-upload-border: rgba(255, 255, 255, 0.18);
    --pa-upload-bg: rgba(255, 255, 255, 0.02);
    --pa-upload-bg-hover: rgba(99, 102, 241, 0.07);
    --pa-upload-icon-bg: rgba(99, 102, 241, 0.16);

    /* Re-point Gradio's OWN theme variables at our palette, so native
       components (inputs, dropdowns, radios, tables, sliders...) stay
       visually consistent with our custom cards instead of falling
       back to gr.themes.Soft's built-in defaults. */
    --body-background-fill: var(--pa-bg);
    --background-fill-primary: var(--pa-bg-elevated);
    --background-fill-secondary: var(--pa-card-bg-solid);
    --border-color-primary: var(--pa-border);
    --border-color-accent: var(--pa-border);
    --body-text-color: var(--pa-text-primary);
    --body-text-color-subdued: var(--pa-text-muted);
    --block-background-fill: var(--pa-card-bg-solid);
    --block-border-color: var(--pa-border);
    --input-background-fill: var(--pa-bg-elevated);
    --table-even-background-fill: var(--pa-bg-elevated);
    --table-odd-background-fill: var(--pa-card-bg-solid);
    --table-border: var(--pa-border);
}

/* ============================================================
   LIGHT THEME - toggled via the sun/moon button (adds/removes
   `.light-mode` on <body> AND every `.gradio-container`, see
   THEME_TOGGLE_JS near the bottom of this file).

   This is a DELIBERATELY, INDEPENDENTLY designed light palette -
   not dark-theme values with backgrounds flipped. Key decisions:
     - Page background is a soft off-white (#F7F8FC), NOT pure white,
       so white cards have something to visually sit ON TOP OF.
     - Cards are pure white with a light border + a resting shadow
       (visible even without hover) so they never disappear into the
       page background.
     - A secondary "soft blue" accent (--pa-blue-accent) supplements
       the indigo/purple brand pair for info states and chart accents.
     - Text uses three solid, hand-tuned grays (not translucent
       opacity over black) tuned for real contrast at each tier, per
       "text should never look faded."
   ============================================================ */
body.light-mode, .gradio-container.light-mode {
    --pa-bg: #f7f8fc;
    --pa-bg-elevated: #ffffff;
    --pa-card-bg: #ffffff;
    --pa-card-bg-solid: #ffffff;
    --pa-card-bg-secondary: #f1f2f8;
    --pa-subtle-fill: #f1f2f8;
    --pa-border: #e4e6f0;
    --pa-border-strong: #d5d8e5;
    --pa-text-primary: #14151f;
    --pa-text-secondary: #565973;
    --pa-text-muted: #85889f;
    --pa-text-disabled: #b7b9c6;

    --readiness-track-color: #e4e6f0;
    --pa-shadow-resting: 0 1px 2px rgba(15, 23, 42, 0.045), 0 2px 8px rgba(15, 23, 42, 0.05);
    --pa-shadow-soft: 0 1px 3px rgba(15, 23, 42, 0.05), 0 8px 22px rgba(15, 23, 42, 0.07);
    --pa-shadow-hover: 0 12px 28px rgba(99, 102, 241, 0.14), 0 2px 8px rgba(15, 23, 42, 0.06);

    --pa-table-header-bg: #f1f2f8;
    --pa-table-row-alt-bg: #fafbfd;
    --pa-table-row-hover-bg: #eef0fc;
    --pa-table-border: #e4e6f0;
    --pa-table-selected-bg: rgba(99, 102, 241, 0.09);

    --pa-btn-primary-shadow: 0 4px 14px rgba(99, 102, 241, 0.22);
    --pa-btn-secondary-bg: #ffffff;
    --pa-btn-secondary-border: #d5d8e5;
    --pa-btn-disabled-bg: #f1f2f8;
    --pa-btn-disabled-text: #b7b9c6;

    --pa-scrollbar-thumb: rgba(99, 102, 241, 0.25);
    --pa-scrollbar-thumb-hover: rgba(99, 102, 241, 0.45);
    --pa-scrollbar-track: #f1f2f8;

    --pa-upload-border: #d5d8e5;
    --pa-upload-bg: #fafbfd;
    --pa-upload-bg-hover: #f1f2fc;
    --pa-upload-icon-bg: rgba(99, 102, 241, 0.1);

    --body-background-fill: var(--pa-bg);
    --background-fill-primary: var(--pa-bg-elevated);
    --background-fill-secondary: #f1f2f8;
    --border-color-primary: var(--pa-border);
    --border-color-accent: var(--pa-border);
    --body-text-color: var(--pa-text-primary);
    --body-text-color-subdued: var(--pa-text-muted);
    --block-background-fill: var(--pa-bg-elevated);
    --block-border-color: var(--pa-border);
    --input-background-fill: #ffffff;
    --table-even-background-fill: #ffffff;
    --table-odd-background-fill: var(--pa-table-row-alt-bg);
    --table-border: var(--pa-border);
}

html, body, .gradio-container {
    background: var(--pa-bg) !important;
}

/* Baseline "resting" elevation for every card-like surface, in BOTH
   themes - visible even before hover, so a white card in light mode
   never blends into the page. Hover lifts to the stronger shadow. */
.kpi-card, .card, .rec-card, .cleaning-tool-card, .export-option-card,
.mini-stat, .chart-card, .insight-card {
    box-shadow: var(--pa-shadow-resting);
}
.info-tooltip::after, .info-tooltip::before {
    box-shadow: var(--pa-shadow-soft);
}

/* ---- Light mode: catch-all safety net for Gradio's own wrapper
   classes ---- `.block` and `.form` are the fundamental containers
   Gradio puts around EVERY component. This is a fallback for any
   native widget NOT covered by a more specific rule below (which,
   being declared later at equal-or-higher specificity, still wins). */
body.light-mode .block, .gradio-container.light-mode .block,
body.light-mode .form, .gradio-container.light-mode .form {
    background: var(--pa-bg-elevated) !important;
    border-color: var(--pa-border) !important;
    color: var(--pa-text-primary) !important;
}

/* ...but undo that catch-all wherever it's nested INSIDE one of our
   own already-styled card containers (.card and its many variants:
   .kpi-card, .cleaning-tool-card, .pivot-card, .cleaning-export-card,
   .cleaning-plan-card, .cleaning-result-card...). Otherwise every
   Row of controls inside a card grows a stray, redundant grey/white
   box around it - a "boxes within boxes" look, since the card itself
   already provides the border/background/shadow. Higher specificity
   than the catch-all above (one extra class), so this always wins. */
body.light-mode .card .block, .gradio-container.light-mode .card .block,
body.light-mode .card .form, .gradio-container.light-mode .card .form {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
/* The dataframe/table INSIDE a card still gets its own background via
   the dedicated table system further below - only the outer wrapper
   box is removed here, not the table itself. */


/* ============================================================
   UPLOAD DROPZONE - a premium drag & drop area instead of Gradio's
   default look. Scoped to `.upload-zone` (the elem_classes wrapper
   already on the file upload Group in app.py) so this can't leak
   onto unrelated `.wrap` elements elsewhere (dropdowns, radios...).
   Theme-aware via --pa-upload-* tokens defined per-theme above, so
   ONE set of rules looks intentionally designed in BOTH themes.
   (The outer dashed gradient card itself is styled separately, in
   the "Upload zone" block below - these rules style what's INSIDE it:
   Gradio's own drop target, which used to stay hard-coded dark.) */
.upload-zone .wrap,
.upload-zone [data-testid="file-upload"],
.upload-zone .file-preview,
.upload-zone .empty.wrap,
.upload-zone .empty {
    background: var(--pa-upload-bg) !important;
    border: 1.5px dashed var(--pa-upload-border) !important;
    border-radius: var(--pa-radius-md) !important;
    color: var(--pa-text-secondary) !important;
    transition: background 0.2s ease, border-color 0.2s ease;
}
.upload-zone .wrap *,
.upload-zone [data-testid="file-upload"] * {
    color: var(--pa-text-secondary) !important;
}
.upload-zone .wrap:hover,
.upload-zone [data-testid="file-upload"]:hover {
    background: var(--pa-upload-bg-hover) !important;
    border-color: var(--pa-indigo) !important;
}
.upload-zone .icon-wrap {
    background: var(--pa-upload-icon-bg) !important;
    border-radius: 999px !important;
    color: var(--pa-indigo) !important;
    padding: 10px !important;
}
.upload-zone .icon-wrap svg {
    fill: var(--pa-indigo) !important;
    stroke: var(--pa-indigo) !important;
}

/* ============================================================
   ACCORDIONS (Upload History, "More examples", etc.) - consistent,
   theme-aware styling everywhere an Accordion appears.
   ============================================================ */
.accordion, .label-wrap, .accordion > .label-wrap, .accordion-header {
    background: var(--pa-bg-elevated) !important;
    color: var(--pa-text-primary) !important;
    border: 1px solid var(--pa-border) !important;
    border-radius: var(--pa-radius-md) !important;
}
.label-wrap:hover {
    color: var(--pa-indigo) !important;
}

/* ============================================================
   TABLES - a proper Power BI / Notion-style system: distinct header,
   zebra striping, row hover, clear borders, and a selected-cell
   treatment - instead of Gradio's flat default table. Applies to
   every gr.Dataframe in the app; theme-aware via --pa-table-* tokens.
   ============================================================ */
table, .table-wrap, .cell-wrap {
    background: var(--pa-bg-elevated) !important;
    color: var(--pa-text-primary) !important;
    border-color: var(--pa-table-border) !important;
}
thead, thead th, .thead {
    background: var(--pa-table-header-bg) !important;
    color: var(--pa-text-primary) !important;
    font-weight: 700 !important;
    border-bottom: 1.5px solid var(--pa-table-border) !important;
}
tbody tr, tbody tr td {
    background: var(--pa-bg-elevated) !important;
    border-color: var(--pa-table-border) !important;
    transition: background 0.12s ease;
}
tbody tr:nth-child(even) td {
    background: var(--pa-table-row-alt-bg) !important;
}
tbody tr:hover td {
    background: var(--pa-table-row-hover-bg) !important;
}
td[aria-selected="true"], .cell-selected, td.selected {
    background: var(--pa-table-selected-bg) !important;
    outline: 1.5px solid var(--pa-indigo) !important;
    outline-offset: -1.5px;
}
.table-wrap {
    border: 1px solid var(--pa-table-border) !important;
    border-radius: var(--pa-radius-md) !important;
    overflow: hidden;
}

/* ---- Theme toggle button ---- */
.theme-toggle-btn {
    border-radius: 999px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 8px 16px !important;
    border: 1px solid var(--pa-border) !important;
    background: var(--pa-card-bg) !important;
    color: var(--pa-text-secondary) !important;
    box-shadow: none !important;
    white-space: nowrap;
}
.theme-toggle-btn:hover {
    border-color: rgba(99, 102, 241, 0.4) !important;
    color: var(--pa-text-primary) !important;
    transform: translateY(-1px);
}


* {
    transition: background-color 0.2s ease, border-color 0.2s ease,
                box-shadow 0.25s ease, transform 0.2s ease, opacity 0.2s ease;
}

/* ---- App header ---- */
.app-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 6px 4px 18px 4px;
}
.app-header-icon {
    flex-shrink: 0;
    width: 48px;
    height: 48px;
    border-radius: 14px;
    background: linear-gradient(135deg, var(--pa-indigo), var(--pa-purple));
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    color: white;
    box-shadow: var(--pa-shadow-hover);
}
.app-header-text h1 {
    margin: 0 !important;
    font-size: 27px !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, var(--pa-indigo), var(--pa-purple) 70%, var(--pa-blue-accent));
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: var(--pa-text-primary);
}
.app-header-text p {
    margin: 4px 0 0 0;
    font-size: 14px;
    color: var(--pa-text-secondary);
}

/* ---- Generic card / section ---- */
.card {
    background: var(--pa-card-bg) !important;
    border: 1px solid var(--pa-border) !important;
    border-radius: var(--pa-radius-lg);
    padding: 22px !important;
    margin-bottom: 20px !important;
    box-shadow: var(--pa-shadow-soft);
    color: var(--pa-text-primary);
}
.card:hover {
    box-shadow: var(--pa-shadow-hover);
    border-color: rgba(99, 102, 241, 0.4) !important;
}
.card h4 {
    margin-top: 0 !important;
    margin-bottom: 12px !important;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: var(--pa-text-primary);
}

/* ---- Feedback tab: header ---- */
.feedback-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 4px 0 8px 0;
}
.feedback-header-icon {
    flex-shrink: 0;
    width: 38px;
    height: 38px;
    border-radius: 11px;
    background: linear-gradient(135deg, var(--pa-indigo), var(--pa-purple));
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 17px;
    color: white;
    box-shadow: var(--pa-btn-primary-shadow);
}
.feedback-header h3 {
    margin: 0 !important;
    padding-left: 0 !important;
    font-size: 19px;
}
.feedback-header h3::before { display: none; }
.feedback-header p {
    margin: 2px 0 0 0;
    font-size: 14px;
    color: var(--pa-text-secondary);
}

/* ---- Feedback tab: stacked-section flow ----
   Deliberately NOT using bordered card boxes here - each question is
   its own whitespace-separated section with a thin hairline divider,
   closer to a Stripe/Linear settings page than a boxy survey form. */
.feedback-flow {
    max-width: 640px;
    margin: 0 auto;
}
.feedback-section {
    padding: 22px 0 !important;
    border-bottom: 1px solid var(--pa-border);
}
.feedback-section-last {
    border-bottom: none !important;
    padding-bottom: 6px !important;
}
.feedback-section p {
    font-size: 15px !important;
    font-weight: 600;
    color: var(--pa-text-primary);
    margin-bottom: 12px !important;
}

/* Turn each Yes/No/rating choice into a clean, lightweight chip -
   transparent by default, accent-colored only when selected. One
   consistent accent color throughout (indigo), no green/red clutter. */
.feedback-flow .wrap label,
.feedback-flow div[role="radiogroup"] label {
    border-radius: 10px !important;
    border: 1px solid var(--pa-border) !important;
    background: transparent !important;
    color: var(--pa-text-secondary) !important;
    font-weight: 500 !important;
    padding: 8px 18px !important;
}
.feedback-flow .wrap label:hover,
.feedback-flow div[role="radiogroup"] label:hover {
    border-color: rgba(99, 102, 241, 0.5) !important;
    color: var(--pa-text-primary) !important;
    transform: translateY(-1px);
}
.feedback-flow .wrap label.selected,
.feedback-flow div[role="radiogroup"] label.selected {
    background: rgba(99, 102, 241, 0.14) !important;
    border-color: var(--pa-indigo) !important;
    color: var(--pa-indigo) !important;
    font-weight: 600 !important;
}

/* ---- Upload zone ---- */
.upload-zone {
    border: 1.5px dashed rgba(99, 102, 241, 0.4) !important;
    border-radius: var(--pa-radius-lg) !important;
    background: linear-gradient(180deg, rgba(99, 102, 241, 0.08), rgba(139, 92, 246, 0.04)) !important;
    padding: 22px !important;
    text-align: center;
}
.upload-zone:hover {
    border-color: var(--pa-indigo) !important;
    background: linear-gradient(180deg, rgba(99, 102, 241, 0.13), rgba(139, 92, 246, 0.06)) !important;
}

/* ---- Buttons ---- */
button.primary, .cta-button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
    background: linear-gradient(135deg, var(--pa-indigo), var(--pa-purple)) !important;
    border: none !important;
    color: white !important;
    box-shadow: var(--pa-btn-primary-shadow);
    transition: box-shadow 0.15s ease, transform 0.15s ease, opacity 0.15s ease;
}
button.primary:hover, .cta-button:hover {
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.45);
    transform: translateY(-1px);
}
button.primary:active, .cta-button:active {
    transform: translateY(0);
    box-shadow: var(--pa-btn-primary-shadow);
}

/* Secondary buttons (the default Gradio button variant) - a clean
   outlined "ghost" style instead of a flat gray box, theme-aware. */
button.secondary {
    background: var(--pa-btn-secondary-bg) !important;
    border: 1px solid var(--pa-btn-secondary-border) !important;
    color: var(--pa-text-secondary) !important;
    font-weight: 600 !important;
    box-shadow: none !important;
    transition: border-color 0.15s ease, color 0.15s ease, transform 0.15s ease, background 0.15s ease;
}
button.secondary:hover {
    border-color: var(--pa-indigo) !important;
    color: var(--pa-indigo) !important;
    transform: translateY(-1px);
}
button.secondary:active {
    transform: translateY(0);
}

/* Disabled buttons - clearly muted, never mistaken for clickable. */
button:disabled, button.primary:disabled, button.secondary:disabled {
    background: var(--pa-btn-disabled-bg) !important;
    color: var(--pa-btn-disabled-text) !important;
    border-color: var(--pa-border) !important;
    box-shadow: none !important;
    cursor: not-allowed !important;
    transform: none !important;
    opacity: 1 !important;
}

button {
    border-radius: 10px !important;
}

/* ---- Tabs ----
   Pill-style navigation (Notion/Linear inspired) instead of plain
   underlined text tabs. Labels stay short with a single minimal
   (monochrome) icon glyph each, so the whole row still fits on one
   line without triggering Gradio's own "..." overflow menu - flex-wrap
   remains as a fallback for very narrow windows. */
.tab-nav {
    gap: 6px;
    border-bottom: 1px solid var(--pa-border) !important;
    padding: 4px 2px 10px 2px;
    flex-wrap: wrap !important;
    row-gap: 8px;
    overflow: visible !important;
}
.tab-nav button {
    border-radius: 10px !important;
    font-weight: 500;
    font-size: 13.5px !important;
    padding: 8px 16px !important;
    opacity: 0.72;
    flex: 0 0 auto !important;
    letter-spacing: -0.005em;
    border: 1px solid transparent !important;
    transition: opacity 0.15s ease, background 0.15s ease, color 0.15s ease, transform 0.15s ease, border-color 0.15s ease;
}
.tab-nav button:hover:not(.selected) {
    opacity: 1;
    background: var(--pa-subtle-fill) !important;
    color: var(--pa-text-primary) !important;
    transform: translateY(-1px);
}
.tab-nav button.selected {
    opacity: 1;
    font-weight: 700;
    color: #ffffff !important;
    background: linear-gradient(135deg, var(--pa-indigo), var(--pa-purple)) !important;
    border: 1px solid transparent !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
}
/* Safety net: hide any "more tabs" overflow toggle Gradio might still
   render - with the row kept narrow, this should rarely be needed. */
.tab-nav > button[aria-label*="tab" i],
.tab-nav > button[title*="tab" i],
.tab-nav .dropdown-arrow {
    display: none !important;
}

/* ---- KPI Cards (Dashboard + About tabs) ---- */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 16px;
    margin-top: 10px;
}
.kpi-card {
    background: var(--pa-card-bg);
    border: 1px solid var(--pa-border);
    border-radius: var(--pa-radius-md);
    padding: 20px 16px;
    text-align: center;
    box-shadow: var(--pa-shadow-soft);
    position: relative;
    overflow: hidden;
    color: var(--pa-text-primary);
}
.kpi-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--pa-indigo), var(--pa-purple), var(--pa-cyan));
    opacity: 0.85;
}
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--pa-shadow-hover);
    border-color: rgba(99, 102, 241, 0.4);
}
.kpi-icon {
    font-size: 24px;
    margin-bottom: 8px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    border-radius: 12px;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.18), rgba(139, 92, 246, 0.18));
    border: 1px solid rgba(99, 102, 241, 0.2);
}
.kpi-value {
    font-size: 27px;
    font-weight: 800;
    line-height: 1.15;
    letter-spacing: -0.025em;
    color: var(--pa-text-primary);
}
.kpi-label {
    font-size: 12.5px;
    color: var(--pa-text-secondary);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.kpi-subchips {
    display: flex;
    justify-content: center;
    gap: 6px;
    margin-top: 10px;
    flex-wrap: wrap;
}
.kpi-subchip {
    font-size: 11px;
    font-weight: 600;
    padding: 3px 9px;
    border-radius: 999px;
    background: rgba(99, 102, 241, 0.12);
    color: var(--pa-indigo);
    text-transform: none;
    letter-spacing: 0;
}
.kpi-empty {
    padding: 40px 24px;
    border-radius: var(--pa-radius-lg);
    background: linear-gradient(180deg, rgba(99, 102, 241, 0.06), rgba(139, 92, 246, 0.03));
    border: 1.5px dashed rgba(99, 102, 241, 0.35);
    text-align: center;
    color: var(--pa-text-secondary);
    font-size: 14.5px;
    position: relative;
}
.kpi-empty::before {
    content: "◆";
    display: block;
    font-size: 26px;
    margin-bottom: 10px;
    background: linear-gradient(135deg, var(--pa-indigo), var(--pa-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.kpi-empty b {
    color: var(--pa-text-primary);
}

/* ---- Section headers (used throughout every tab) ----
   Every gr.Markdown "### Heading" / "#### Heading" in the app picks
   this up automatically - no per-tab markup changes needed. This is
   what gives every section a consistent, intentional look instead of
   plain floating text. */
h3, h4 {
    position: relative;
    padding-left: 14px !important;
    margin: 20px 0 12px 0 !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
    color: var(--pa-text-primary) !important;
}
h3::before, h4::before {
    content: "";
    position: absolute;
    left: 0;
    top: 3px;
    bottom: 3px;
    width: 4px;
    border-radius: 4px;
    background: linear-gradient(180deg, var(--pa-indigo), var(--pa-purple));
}
h4::before {
    width: 3px;
    background: linear-gradient(180deg, var(--pa-purple), var(--pa-cyan));
}
.card h4, .card h4::before {
    /* Headings already inside a card don't need the accent bar again -
       the card itself provides the visual separation. */
    padding-left: 0 !important;
}
.card h4::before { display: none; }

/* ---- AI Readiness score card ---- */
.readiness-card {
    display: flex;
    align-items: center;
    gap: 32px;
    flex-wrap: wrap;
    background: var(--pa-card-bg);
    border: 1px solid var(--pa-border);
    border-radius: var(--pa-radius-lg);
    padding: 28px;
    box-shadow: var(--pa-shadow-soft);
    color: var(--pa-text-primary);
}
.readiness-gauge {
    flex-shrink: 0;
    width: 140px;
    height: 140px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: var(--pa-shadow-soft), inset 0 0 0 1px var(--pa-border);
}
.readiness-gauge-inner {
    width: 108px;
    height: 108px;
    border-radius: 50%;
    background: var(--pa-card-bg-solid);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    box-shadow: var(--pa-shadow-resting);
}
.readiness-score {
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1;
    color: var(--pa-text-primary);
}
.readiness-score-max {
    font-size: 12px;
    color: var(--pa-text-muted);
    margin-top: 2px;
}
.readiness-details {
    flex: 1;
    min-width: 240px;
}
.readiness-badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 13px;
    margin-bottom: 16px;
}
.readiness-bars {
    display: flex;
    flex-direction: column;
    gap: 12px;
}
.readiness-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    margin-bottom: 4px;
    color: var(--pa-text-secondary);
}
.readiness-bar-value {
    font-weight: 600;
    color: var(--pa-text-muted);
}
.readiness-bar-track {
    width: 100%;
    height: 7px;
    border-radius: 999px;
    background: var(--readiness-track-color);
    overflow: hidden;
}
.readiness-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--pa-purple), var(--pa-indigo));
}

/* ---- Metadata tab: BI-style analytics dashboard ---- */
.meta-section-heading {
    margin-top: 34px !important;
}
.meta-card {
    background: var(--pa-card-bg);
    border: 1px solid var(--pa-border);
    border-radius: var(--pa-radius-lg);
    padding: 20px;
    box-shadow: var(--pa-shadow-soft);
    overflow-x: auto;
}
.meta-subsection-title {
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--pa-text-secondary);
    margin: 4px 0 10px 0;
}
.meta-subsection-title:not(:first-child) {
    margin-top: 20px;
}
.meta-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin-bottom: 8px;
}
.meta-table th {
    text-align: left;
    padding: 8px 12px;
    color: var(--pa-text-secondary);
    font-weight: 600;
    border-bottom: 1px solid var(--pa-border);
    white-space: nowrap;
}
.meta-table td {
    padding: 8px 12px;
    color: var(--pa-text-primary);
    border-bottom: 1px solid var(--pa-border);
    white-space: nowrap;
}
.meta-table tr:hover td {
    background: rgba(99, 102, 241, 0.05);
}

/* ---- Dataset Profile cards (kept for reuse - no longer shown on the live Data Insights report) ---- */
.profile-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 14px;
}
.profile-card {
    background: var(--pa-card-bg);
    border: 1px solid var(--pa-border);
    border-radius: var(--pa-radius-md);
    padding: 16px;
    box-shadow: var(--pa-shadow-soft);
}
.profile-card:hover {
    border-color: rgba(99, 102, 241, 0.35);
    transform: translateY(-2px);
}
.profile-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}
.profile-col-name {
    font-weight: 700;
    font-size: 14px;
    color: var(--pa-text-primary);
}
.profile-dtype-badge {
    font-size: 10.5px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 999px;
    background: rgba(99, 102, 241, 0.14);
    color: var(--pa-indigo);
}
.profile-card-body {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-bottom: 10px;
}
.profile-stat {
    display: flex;
    justify-content: space-between;
    font-size: 12.5px;
    color: var(--pa-text-secondary);
}
.profile-stat b {
    color: var(--pa-text-primary);
}
.profile-usage {
    font-size: 12px;
    color: var(--pa-text-muted);
    border-top: 1px solid var(--pa-border);
    padding-top: 8px;
    font-style: italic;
}

/* ---- BI KPI blocks (Section 3) ---- */
.bi-kpi-block {
    margin-bottom: 22px;
}
.bi-kpi-column-title {
    font-weight: 700;
    font-size: 14px;
    color: var(--pa-indigo);
    margin-bottom: 8px;
}
.bi-kpi-grid .kpi-card {
    padding: 12px 8px;
}
.bi-kpi-grid .kpi-value {
    font-size: 17px;
}

/* ---- Automatic Visual Analytics charts (Section 4) ---- */
.chart-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px;
}
.chart-card {
    background: var(--pa-card-bg);
    border: 1px solid var(--pa-border);
    border-radius: var(--pa-radius-md);
    padding: 14px;
    box-shadow: var(--pa-shadow-soft);
    text-align: center;
}
.chart-card:hover {
    border-color: rgba(99, 102, 241, 0.35);
}
.chart-card-title {
    font-weight: 600;
    font-size: 13px;
    color: var(--pa-text-secondary);
    margin-bottom: 8px;
}
.chart-card-img {
    width: 100%;
    height: auto;
    border-radius: 8px;
}

/* ---- AI-generated insight shown under the Quick Visual Summary chart(s) ---- */
.chart-insight-box {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    margin-top: 14px;
    background: rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-left: 3px solid var(--pa-indigo);
    border-radius: var(--pa-radius-md);
    padding: 14px 16px;
}
.chart-insight-icon {
    font-size: 17px;
    flex-shrink: 0;
    line-height: 1.4;
}
.chart-insight-text {
    font-size: 12.5px;
    line-height: 1.6;
    color: var(--pa-text-secondary);
}
.chart-insight-text b {
    color: var(--pa-text-primary);
}

/* ---- Smart AI Insight cards (Section 5) ---- */
.insight-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 14px;
}
.insight-card {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    background: var(--pa-card-bg);
    border: 1px solid var(--pa-border);
    border-radius: var(--pa-radius-md);
    padding: 16px;
    box-shadow: var(--pa-shadow-soft);
}
.insight-card:hover {
    border-color: rgba(99, 102, 241, 0.35);
    transform: translateY(-2px);
}
.insight-icon {
    font-size: 20px;
    flex-shrink: 0;
}
.insight-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
}
.insight-title {
    font-weight: 700;
    font-size: 13.5px;
    color: var(--pa-text-primary);
    margin-bottom: 0;
}
.insight-badge {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    padding: 2px 8px;
    border-radius: 999px;
    background: rgba(99, 102, 241, 0.14);
    color: var(--pa-indigo);
    white-space: nowrap;
    flex-shrink: 0;
}
.insight-description {
    font-size: 12.5px;
    color: var(--pa-text-secondary);
    line-height: 1.4;
}

/* ---- AI Workspace: expanded, single-focus chat card ----
   The AI Assistant is the only thing on this page now, so it gets a
   roomier card (instead of the compact panel used when it shared the
   page with the readiness score + recommendation sections). */
.workspace-chat-card {
    padding: 22px !important;
}
.workspace-chat-card .chatbot,
.workspace-chat-card [data-testid="chatbot"] {
    border-radius: var(--pa-radius-md) !important;
}

/* ---- Pivot Table Builder ----
   Same visual language as the chart builder card above it: a purple
   label pill sitting directly above a plain dark box, grouped in one
   shared card - no separate bordered "tiles" per field, just clean,
   consistent spacing so it reads as part of the same page. */
.pivot-card {
    padding: 20px !important;
    display: flex;
    flex-direction: column;
    gap: 18px;
}
.pivot-field-row {
    gap: 20px !important;
}
.pivot-generate-button {
    margin-top: 2px !important;
}

/* Selected-value chips inside every pivot multiselect - these are the
   "blue boxes" that stood out against the dark UI. Re-themed to a
   soft indigo pill, consistent with .kpi-subchip / .suggestion-chip
   elsewhere in the app. Several selector shapes are targeted since
   Gradio's internal token markup can vary slightly by version. */
.pivot-multiselect .token,
.pivot-multiselect [class*="token"],
.pivot-select .token,
.pivot-select [class*="token"] {
    background: rgba(99, 102, 241, 0.16) !important;
    border: 1px solid rgba(99, 102, 241, 0.35) !important;
    color: var(--pa-indigo) !important;
    border-radius: 999px !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    padding: 3px 10px !important;
    box-shadow: none !important;
}
.pivot-multiselect .token:hover,
.pivot-multiselect [class*="token"]:hover {
    border-color: var(--pa-indigo) !important;
    background: rgba(99, 102, 241, 0.24) !important;
}
.pivot-multiselect .token svg,
.pivot-multiselect [class*="token"] svg,
.pivot-multiselect .token button,
.pivot-multiselect [class*="token"] button {
    color: var(--pa-indigo) !important;
    opacity: 0.75;
}

/* Dropdown shell itself - crisper border/radius, no harsh default blue
   focus ring, matching the rest of the app's inputs. */
.pivot-multiselect .wrap-inner,
.pivot-select .wrap-inner,
.pivot-multiselect [data-testid="dropdown"],
.pivot-select [data-testid="dropdown"] {
    background: var(--pa-card-bg-solid) !important;
    border-radius: var(--pa-radius-md) !important;
    border: 1px solid var(--pa-border) !important;
}
.pivot-multiselect .wrap-inner:focus-within,
.pivot-select .wrap-inner:focus-within {
    border-color: rgba(99, 102, 241, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.12) !important;
}

.pivot-result-card {
    padding: 20px !important;
}
.pivot-result-card .dataframe {
    border-radius: var(--pa-radius-md) !important;
}

/* ---- Status badges (used inline in markdown/status text) ---- */
.status-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 12.5px;
    font-weight: 600;
}

/* ---- VizBot: suggestion chips ----
   Small, rounded, single-click prompts (ChatGPT/Claude/Perplexity
   style) instead of a big expandable list of bullet points. */
.suggestion-chip-row {
    flex-wrap: wrap !important;
    gap: 8px !important;
    margin-bottom: 4px;
}
.suggestion-chip {
    border-radius: 999px !important;
    border: 1px solid var(--pa-border) !important;
    background: transparent !important;
    color: var(--pa-text-secondary) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 6px 14px !important;
    flex: 0 0 auto !important;
    box-shadow: none !important;
}
.suggestion-chip:hover {
    border-color: var(--pa-indigo) !important;
    color: var(--pa-indigo) !important;
    background: rgba(99, 102, 241, 0.08) !important;
    transform: translateY(-1px);
}

/* ---- VizBot: compact "more examples" popover -----
   A capped-height, scrollable panel instead of a long accordion that
   pushes the whole page down when opened. */
.examples-popover {
    max-height: 260px;
    overflow-y: auto;
    border-radius: var(--pa-radius-md) !important;
}

/* Generic muted one-line caption, used under section headers across
   several tabs (Metadata's Quick Visual Summary, etc.) */
.cleaning-widget-note {
    font-size: 12px;
    color: var(--pa-text-muted);
    display: block;
    margin: -4px 0 10px 0;
}

/* ---- Page intro header (Cleaning tab) ----
   A small eyebrow + title + subtitle block, the same pattern used by
   premium SaaS dashboards (Stripe, Linear) instead of a plain
   Markdown heading. */
.page-intro {
    margin-bottom: 20px;
}
.page-intro-eyebrow {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.08em;
    background: linear-gradient(90deg, var(--pa-indigo), var(--pa-blue-accent));
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: var(--pa-indigo);
    margin-bottom: 4px;
}
.page-intro-title {
    font-size: 21px !important;
    font-weight: 800 !important;
    letter-spacing: -0.01em;
    color: var(--pa-text-primary) !important;
    margin: 0 0 6px 0 !important;
}
.page-intro-subtitle {
    font-size: 13px;
    color: var(--pa-text-secondary);
    line-height: 1.6;
    max-width: 640px;
    margin: 0;
}

/* ---- Cleaning Toolkit section header ---- */
.toolkit-header {
    margin: 26px 0 14px 0;
}
.toolkit-title {
    font-size: 16px !important;
    font-weight: 800 !important;
    letter-spacing: -0.01em;
    color: var(--pa-text-primary) !important;
    margin: 0 0 3px 0 !important;
}
.toolkit-subtitle {
    font-size: 12.5px;
    color: var(--pa-text-secondary);
    margin: 0;
}

/* ---- Cleaning Toolkit grid: two equal-height, self-contained tool
   cards instead of one long stacked column. Grid's default
   align-items: stretch keeps both cards the same height even when
   their content differs slightly. ---- */
.cleaning-grid {
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 18px;
    margin: 0 0 20px 0;
    align-items: stretch;
}
.cleaning-grid > .card.cleaning-tool-card {
    margin-bottom: 0 !important;
    padding: 20px !important;
    display: flex;
    flex-direction: column;
    gap: 10px;
}

/* Icon + title + one-line description header, shared by every tool
   card - the "clear icon, title, one-line description" pattern. */
.tool-card-header {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding-bottom: 12px;
    margin-bottom: 2px;
    border-bottom: 1px solid var(--pa-border);
}
.tool-card-icon {
    font-size: 18px;
    line-height: 1;
    flex-shrink: 0;
    width: 34px;
    height: 34px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.16), rgba(139, 92, 246, 0.16));
    border: 1px solid rgba(99, 102, 241, 0.18);
    color: var(--pa-indigo);
    border-radius: var(--pa-radius-md);
}
.tool-card-title {
    font-size: 15px;
    font-weight: 800;
    letter-spacing: -0.005em;
    color: var(--pa-text-primary);
    line-height: 1.3;
}
.tool-card-desc {
    font-size: 11.5px;
    color: var(--pa-text-secondary);
    margin-top: 2px;
}

/* Short static explanation of the detection method (Outlier card). */
.tool-method-note {
    font-size: 11.5px;
    color: var(--pa-text-muted);
    line-height: 1.5;
    background: var(--pa-subtle-fill);
    border: 1px solid var(--pa-border);
    border-radius: var(--pa-radius-md);
    padding: 8px 10px;
}
.tool-method-note b {
    color: var(--pa-text-secondary);
}

.outlier-controls-row {
    gap: 10px !important;
    align-items: end !important;
}

.cleaning-plan-card {
    border-color: rgba(99, 102, 241, 0.3) !important;
    background: linear-gradient(180deg, rgba(99, 102, 241, 0.06), rgba(139, 92, 246, 0.02)) !important;
    padding: 20px !important;
}

/* A quiet caption sitting above a cleaning control - used instead of
   Gradio's default label pill wherever an info tooltip needs to sit
   right next to the option name. */
.cleaning-field-label {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    font-weight: 700;
    color: var(--pa-text-secondary);
    margin-top: 2px;
}

/* Small "ⓘ" badge with a pure-CSS hover tooltip - no JS needed. */
.info-tooltip {
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 15px;
    height: 15px;
    border-radius: 50%;
    background: rgba(99, 102, 241, 0.16);
    color: var(--pa-indigo);
    font-size: 10.5px;
    font-weight: 700;
    cursor: help;
    flex-shrink: 0;
}
.info-tooltip::after {
    content: attr(data-tooltip);
    position: absolute;
    bottom: 135%;
    left: 50%;
    transform: translateX(-50%);
    background: var(--pa-tooltip-bg);
    color: var(--pa-tooltip-text);
    border: 1px solid var(--pa-border);
    border-radius: 8px;
    padding: 8px 10px;
    font-size: 11.5px;
    font-weight: 500;
    line-height: 1.45;
    white-space: normal;
    width: max-content;
    max-width: 230px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    opacity: 0;
    visibility: hidden;
    transition: opacity 0.15s ease;
    z-index: 60;
    pointer-events: none;
}
.info-tooltip::before {
    content: "";
    position: absolute;
    bottom: 125%;
    left: 50%;
    transform: translateX(-50%);
    border: 5px solid transparent;
    border-top-color: var(--pa-tooltip-bg);
    opacity: 0;
    visibility: hidden;
    transition: opacity 0.15s ease;
    z-index: 60;
}
.info-tooltip:hover::after,
.info-tooltip:hover::before {
    opacity: 1;
    visibility: visible;
}

/* Segmented-control look for the Outlier Strategy radio - pill
   buttons instead of default circular radio inputs. */
.segmented-control .wrap {
    display: flex !important;
    flex-wrap: wrap;
    gap: 8px !important;
    background: transparent !important;
}
/* Gradio's own radio label sets color:transparent on the unchecked
   state (it normally relies on a background swap to "reveal" text) -
   our pill background doesn't do that, so the label text was
   invisible until this explicit override. Also hide the native radio
   dot entirely; the pill background/border already show selection. */
.segmented-control label {
    border: 1px solid var(--pa-border) !important;
    border-radius: 999px !important;
    padding: 6px 14px !important;
    background: var(--pa-subtle-fill) !important;
    color: var(--pa-text-secondary) !important;
    font-size: 12.5px !important;
    font-weight: 600 !important;
    transition: border-color 0.15s ease, background 0.15s ease, color 0.15s ease;
}
.segmented-control label * {
    color: inherit !important;
}
.segmented-control input[type="radio"] {
    display: none !important;
}
.segmented-control label:hover {
    border-color: rgba(99, 102, 241, 0.4) !important;
    color: var(--pa-text-primary) !important;
}
.segmented-control label:has(input:checked) {
    background: rgba(99, 102, 241, 0.18) !important;
    border-color: var(--pa-indigo) !important;
    color: var(--pa-indigo) !important;
}

/* Legend under the Outlier Strategy radio, explaining the two real
   (non "Do Nothing") options without cluttering the radio itself. */
.outlier-strategy-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    margin-top: 2px;
}
.outlier-strategy-legend-item {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 11.5px;
    color: var(--pa-text-muted);
}

/* ---- Missing Values mini-stat row ---- */
.mini-stat-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
}
.mini-stat {
    background: var(--pa-subtle-fill);
    border: 1px solid var(--pa-border);
    border-radius: var(--pa-radius-md);
    padding: 8px 10px;
    text-align: center;
}
.mini-stat-value {
    font-size: 16px;
    font-weight: 800;
    color: var(--pa-text-primary);
    line-height: 1.2;
}
.mini-stat-label {
    font-size: 9.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--pa-text-muted);
    margin-top: 2px;
}
.mini-stat-empty {
    font-size: 12px;
    color: var(--pa-text-muted);
    padding: 8px 2px;
}
.mini-stat-clean {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: var(--pa-text-secondary);
    background: rgba(34, 197, 94, 0.08);
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-radius: var(--pa-radius-md);
    padding: 9px 12px;
}
.mini-stat-clean-icon {
    color: #22c55e;
    font-weight: 800;
}

/* Search box + compact table for the Missing Values card - a fixed
   max-height with internal scroll so a wide dataset never blows the
   card out to an unreadable length. */
.cleaning-search-box input {
    font-size: 12.5px !important;
    border-radius: 999px !important;
    padding: 9px 16px !important;
    background: var(--pa-subtle-fill) !important;
    border: 1px solid var(--pa-border) !important;
    transition: border-color 0.15s ease, background 0.15s ease, box-shadow 0.15s ease;
}
.cleaning-search-box input:focus {
    background: var(--pa-bg-elevated) !important;
    border-color: var(--pa-indigo) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.12) !important;
}
.cleaning-compact-table {
    max-height: 220px;
    overflow-y: auto !important;
}
.cleaning-compact-table table {
    font-size: 12px !important;
}

/* ---- Chatbot ---- */
.form, .block { border-radius: var(--pa-radius-md) !important; }

/* ---- Export card (Data Cleaning tab) ----
   Compact icon-led header + two option "cards" (not a bare dropdown
   + generic button), and a hidden-until-ready success state - no
   dashed empty drop-zone is ever shown. */
.cleaning-export-card {
    margin-bottom: 4px !important;
    padding: 20px !important;
}
.export-card-header {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 14px;
}
.export-card-icon {
    font-size: 22px;
    line-height: 1;
    flex-shrink: 0;
    width: 38px;
    height: 38px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.16), rgba(139, 92, 246, 0.16));
    border: 1px solid rgba(99, 102, 241, 0.18);
    border-radius: var(--pa-radius-md);
}
.export-card-title {
    font-size: 15px;
    font-weight: 800;
    letter-spacing: -0.005em;
    color: var(--pa-text-primary);
}
.export-card-subtitle {
    font-size: 12px;
    color: var(--pa-text-secondary);
    margin-top: 2px;
}
.export-options-row {
    gap: 12px !important;
}
.export-option-card {
    border: 1px solid var(--pa-border);
    border-radius: var(--pa-radius-md);
    padding: 10px 10px 12px 10px !important;
    text-align: center;
    transition: border-color 0.15s ease, transform 0.15s ease;
    background: var(--pa-subtle-fill);
}
.export-option-card:hover {
    border-color: rgba(99, 102, 241, 0.4);
    transform: translateY(-1px);
}
.export-option-btn {
    font-weight: 700 !important;
    width: 100%;
}
.export-option-caption {
    font-size: 10.5px;
    color: var(--pa-text-muted);
    margin-top: 6px;
}

/* Success banner + real download chip, both hidden until export()
   actually succeeds - eliminates the dashed empty file drop-zone. */
.export-success-banner-inner {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12.5px;
    font-weight: 600;
    color: var(--pa-text-primary);
    background: rgba(34, 197, 94, 0.1);
    border: 1px solid rgba(34, 197, 94, 0.35);
    border-radius: var(--pa-radius-md);
    padding: 10px 14px;
    margin-top: 4px;
}
.export-success-icon {
    font-size: 14px;
}
.export-output {
    margin-top: 6px;
}

/* ---- Page-level spacing & premium framing ---- */
.gradio-container {
    max-width: 1320px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding: 28px 32px 48px 32px !important;
}
@media (max-width: 900px) {
    .gradio-container { padding: 18px 16px 32px 16px !important; }
}

/* App header sits in a row with the theme toggle on the right, instead
   of the toggle floating awkwardly on its own line. */
.app-header-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
}

/* ---- Subtle, deliberate motion ----
   Tab content fades/slides in on switch, and KPI-style cards ease in
   on first paint - both short and understated, not flashy. */
@keyframes pa-fade-in {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}
.tabitem {
    animation: pa-fade-in 0.28s ease;
}
.kpi-card, .rec-card, .cleaning-tool-card, .export-option-card {
    animation: pa-fade-in 0.32s ease;
}

/* Smooth open/close for accordions (Upload History, "More examples",
   etc.) instead of an abrupt snap. */
.accordion, .gr-accordion {
    transition: all 0.25s ease;
}

/* ---- Scrollbar polish (theme-aware) ---- */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--pa-scrollbar-track); }
::-webkit-scrollbar-thumb {
    background: var(--pa-scrollbar-thumb);
    border-radius: 999px;
}
::-webkit-scrollbar-thumb:hover { background: var(--pa-scrollbar-thumb-hover); }
* { scrollbar-width: thin; scrollbar-color: var(--pa-scrollbar-thumb) var(--pa-scrollbar-track); }
"""


# ----------------------------------------------------------------------
# THEME TOGGLE (Light / Dark)
# Pure client-side JS - no server round trip. Adds/removes a
# `light-mode` class on BOTH <body> AND every `.gradio-container`
# element found on the page. Gradio's own component tree doesn't
# always live in a scope where a class on <body> alone reaches every
# descendant selector reliably, so we cover both anchors - every rule
# in CUSTOM_CSS keyed off `.light-mode` matches against either. The
# choice is remembered for the rest of the browser session via
# sessionStorage (cleared when the tab closes, per the "remember
# during the session" requirement) and restored on load.
# ----------------------------------------------------------------------

THEME_INIT_JS = """
() => {
    const applyClass = (isLight) => {
        document.body.classList.toggle('light-mode', isLight);
        document.querySelectorAll('.gradio-container').forEach((el) => {
            el.classList.toggle('light-mode', isLight);
        });
    };
    const saved = sessionStorage.getItem('datanest-theme');
    const isLight = saved === 'light';

    applyClass(isLight);
    // .gradio-container may not have mounted yet at this exact point -
    // retry briefly so the saved theme still applies once it does.
    let attempts = 0;
    const retry = setInterval(() => {
        applyClass(isLight);
        attempts += 1;
        if (attempts > 15) clearInterval(retry);
    }, 150);

    const wrapper = document.getElementById('theme-toggle-btn');
    const btn = wrapper ? (wrapper.querySelector('button') || wrapper) : null;
    if (btn) {
        btn.textContent = isLight ? '🌙  Dark Mode' : '☀️  Light Mode';
    }
}
"""

THEME_TOGGLE_JS = """
() => {
    const currentlyLight = document.body.classList.contains('light-mode');
    const isLight = !currentlyLight;
    document.body.classList.toggle('light-mode', isLight);
    document.querySelectorAll('.gradio-container').forEach((el) => {
        el.classList.toggle('light-mode', isLight);
    });
    sessionStorage.setItem('datanest-theme', isLight ? 'light' : 'dark');
    const wrapper = document.getElementById('theme-toggle-btn');
    const btn = wrapper ? (wrapper.querySelector('button') || wrapper) : null;
    if (btn) {
        btn.textContent = isLight ? '🌙  Dark Mode' : '☀️  Light Mode';
    }
}
"""


# ----------------------------------------------------------------------
# BUILD THE GRADIO APP
# ----------------------------------------------------------------------

with gr.Blocks(
    title="DataNest",
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    ),
    css=CUSTOM_CSS,
    js=THEME_INIT_JS,
) as demo:

    with gr.Row(elem_classes="app-header-row"):
        gr.HTML(
            """
            <div class="app-header">
                <div class="app-header-icon">◆</div>
                <div class="app-header-text">
                    <h1>DataNest</h1>
                    <p>Upload a CSV or Excel file to explore, clean, visualize, and chat with your data.</p>
                </div>
            </div>
            """
        )
        theme_toggle_button = gr.Button(
            "☀️  Light Mode", elem_id="theme-toggle-btn", elem_classes="theme-toggle-btn",
        )

    # -----------------------------------------------------------------
    # Shared state: the loaded DataFrame is stored here so every tab
    # (Dashboard, Metadata, AI Readiness, Visualization Studio...) can
    # access the SAME dataset without reloading the file.
    # -----------------------------------------------------------------
    dataset_state = gr.State(value=None)

    # -----------------------------------------------------------------
    # File upload section (shared across the whole app) - styled as a
    # drag-and-drop zone card rather than a bare file input + button.
    # -----------------------------------------------------------------
    with gr.Group(elem_classes="upload-zone"):
        gr.Markdown("**Upload your dataset** · CSV or Excel (.xlsx / .xls)")
        file_input = gr.File(
            label="",
            show_label=False,
            file_types=[".csv", ".xlsx", ".xls"],
        )
        load_button = gr.Button("Load Dataset", variant="primary", elem_classes="cta-button")

    status_box = gr.Markdown("No file loaded yet.")

    # -----------------------------------------------------------------
    # Upload History: every successful upload is logged as a NEW row
    # (never overwritten) in upload_history.csv. Collapsed by default
    # so it doesn't clutter the main flow, but always up to date.
    # -----------------------------------------------------------------
    with gr.Accordion("Upload History", open=False):
        upload_history_output = gr.Dataframe(
            value=get_upload_history(),
            label="Every dataset you've loaded, most recent first",
            headers=["Filename", "Timestamp", "Rows", "Columns"],
            wrap=True,
        )

    # -----------------------------------------------------------------
    # Tabs
    # -----------------------------------------------------------------
    with gr.Tabs():

        # ============================ TAB 1: DASHBOARD ============================
        with gr.Tab("▦  Dashboard"):
            gr.Markdown("### Overview of your dataset")
            dashboard_output = gr.HTML(format_stats_as_kpi_cards({}))

            gr.Markdown("### 👀 Dataset Preview (first 10 rows)")
            preview_output = gr.Dataframe(label="Preview", wrap=True)

        # ============================ TAB 2: DATASET METADATA ============================
        with gr.Tab("≡ Data Insights"):
            gr.Markdown(
                "### Dataset Analytics Report\n"
                "An automatically generated analytical report — statistical summaries, dataset-aware "
                "charts, BI-style KPIs, and smart insights."
            )
            metadata_top_output = gr.HTML(render_metadata_top_html(None))

            gr.Markdown("### Business Intelligence KPIs")
            metric_selector_dropdown = gr.Dropdown(
                choices=[], value=None, label="Select Metric to Analyze",
            )
            bi_kpis_output = gr.HTML(format_bi_kpis_html({}))

            gr.Markdown("### Quick Visual Summary")
            gr.Markdown(
                "<span class='cleaning-widget-note'>Explore any column without leaving this tab — "
                "pick one below to see its distribution, plus a short AI-generated insight.</span>"
            )
            visual_summary_column_dropdown = gr.Dropdown(
                choices=[], value=None, label="Select a Column",
            )
            visual_summary_output = gr.HTML(
                "<div class='kpi-empty'>Load a dataset to explore its columns visually.</div>"
            )

            metadata_bottom_output = gr.HTML("")

        # ============================ TAB 3: DATA CLEANING ============================
        with gr.Tab("⟲  PreProcessing"):
            gr.HTML(
                "<div class='page-intro'>"
                "<div class='page-intro-eyebrow'>DATA PREPARATION</div>"
                "<h3 class='page-intro-title'>Clean your dataset</h3>"
                "<p class='page-intro-subtitle'>A focused toolkit: handle missing values and "
                "outliers, preview the plan, then export the result. Your original dataset in the "
                "other tabs is never changed.</p>"
                "</div>"
            )

            # Holds the cleaned DataFrame separately from the main
            # dataset_state - cleaning never overwrites the original.
            cleaned_dataset_state = gr.State(value=None)

            # ---- Export card: compact, icon-led, no dashed drop-zone ever ----
            with gr.Group(elem_classes="card cleaning-export-card"):
                gr.HTML(
                    "<div class='export-card-header'>"
                    "<div class='export-card-icon'>📦</div>"
                    "<div>"
                    "<div class='export-card-title'>Export Cleaned Dataset</div>"
                    "<div class='export-card-subtitle'>Pick a format once you've applied changes "
                    "below.</div>"
                    "</div>"
                    "</div>"
                )
                with gr.Row(elem_classes="export-options-row"):
                    with gr.Column(elem_classes="export-option-card", min_width=140):
                        export_csv_button = gr.Button("📄  CSV", elem_classes="export-option-btn")
                        gr.HTML("<div class='export-option-caption'>Universal spreadsheet format</div>")
                    with gr.Column(elem_classes="export-option-card", min_width=140):
                        export_excel_button = gr.Button("📊  Excel", elem_classes="export-option-btn")
                        gr.HTML("<div class='export-option-caption'>.xlsx with column formatting</div>")

                # Hidden until an export actually succeeds - no dashed
                # empty drop-zone is ever shown to the user.
                export_success_banner = gr.HTML(visible=False)
                export_file_output = gr.File(
                    label="Your cleaned dataset",
                    interactive=False,
                    visible=False,
                    elem_classes="export-output",
                )

            gr.HTML(
                "<div class='toolkit-header'>"
                "<h3 class='toolkit-title'>Cleaning Toolkit</h3>"
                "<p class='toolkit-subtitle'>Two focused tools — hover the ⓘ next to any option "
                "for a quick explanation.</p>"
                "</div>"
            )

            # ---- Widget grid: each card is a self-contained, equal-height
            #      workspace tool instead of one long stacked column. ----
            with gr.Column(elem_classes="cleaning-grid"):

                # ============ TOOL 1: MISSING VALUES ============
                with gr.Group(elem_classes="card cleaning-tool-card"):
                    gr.HTML(
                        "<div class='tool-card-header'>"
                        "<div class='tool-card-icon'>◫</div>"
                        "<div>"
                        "<div class='tool-card-title'>Missing Values</div>"
                        "<div class='tool-card-desc'>Find and fill gaps in your data.</div>"
                        "</div>"
                        "</div>"
                    )
                    missing_stats_output = gr.HTML(format_missing_value_stats_html(get_missing_value_stats(None)))

                    missing_search_box = gr.Textbox(
                        placeholder="🔎 Search columns...",
                        show_label=False,
                        elem_classes="cleaning-search-box",
                    )
                    missing_summary_output = gr.Dataframe(
                        value=get_missing_summary(None),
                        headers=["Column", "Missing Count", "Missing %"],
                        wrap=True,
                        show_label=False,
                        elem_classes="cleaning-compact-table",
                    )

                    gr.HTML(
                        "<div class='cleaning-field-label'>Cleaning Strategy"
                        "<span class='info-tooltip' data-tooltip='Choose how missing values are "
                        "handled: drop incomplete rows, or automatically fill numeric columns "
                        "(mean/median) or categorical columns (most frequent value).'>ⓘ</span></div>"
                    )
                    missing_strategy_dropdown = gr.Dropdown(
                        choices=MISSING_VALUE_STRATEGIES,
                        value="Do Nothing",
                        show_label=False,
                    )

                # ============ TOOL 2: OUTLIER HANDLING ============
                with gr.Group(elem_classes="card cleaning-tool-card"):
                    gr.HTML(
                        "<div class='tool-card-header'>"
                        "<div class='tool-card-icon'>◉</div>"
                        "<div>"
                        "<div class='tool-card-title'>Outlier Handling</div>"
                        "<div class='tool-card-desc'>Spot and neutralize extreme values.</div>"
                        "</div>"
                        "</div>"
                    )
                    gr.HTML(
                        "<div class='tool-method-note'>Detection uses the <b>1.5×IQR rule</b>: any "
                        "value below Q1 − 1.5×IQR or above Q3 + 1.5×IQR is flagged as an outlier.</div>"
                    )

                    with gr.Row(elem_classes="outlier-controls-row"):
                        outlier_column_dropdown = gr.Dropdown(
                            choices=[], value=None, label="Numeric Column", scale=3,
                        )
                        detect_outliers_button = gr.Button("Detect", elem_classes="export-option-btn", scale=1)
                    outlier_status_output = gr.Markdown("")

                    gr.HTML("<div class='cleaning-field-label'>Outlier Strategy</div>")
                    outlier_strategy_radio = gr.Radio(
                        choices=OUTLIER_STRATEGIES,
                        value="Do Nothing",
                        show_label=False,
                        elem_classes="segmented-control",
                    )
                    gr.HTML(
                        "<div class='outlier-strategy-legend'>"
                        "<span class='outlier-strategy-legend-item'>Remove Outliers"
                        "<span class='info-tooltip' data-tooltip='Permanently removes rows "
                        "containing detected outliers from the cleaned dataset.'>ⓘ</span></span>"
                        "<span class='outlier-strategy-legend-item'>Replace with Maximum Value"
                        "<span class='info-tooltip' data-tooltip='Replaces each detected outlier "
                        "with the maximum valid (non-outlier) value found in that column, keeping "
                        "the row instead of deleting it.'>ⓘ</span></span>"
                        "</div>"
                    )

            with gr.Group(elem_classes="card cleaning-plan-card"):
                gr.Markdown("#### 📋 Planned Changes")
                cleaning_plan_output = gr.Markdown("Upload a dataset to see the cleaning plan.")

            apply_cleaning_button = gr.Button("Apply Cleaning", variant="primary", size="lg")
            cleaning_summary_output = gr.Markdown("")

            gr.Markdown("#### Cleaned Data Preview (first 10 rows)")
            cleaned_preview_output = gr.Dataframe(label="Cleaned Preview", wrap=True)

        # ============================ TAB 4: AI WORKSPACE ============================
        # A focused, single-purpose page: the AI Assistant (VizBot) is the
        # whole story here — no readiness score or recommendation cards
        # competing for attention above it.
        with gr.Tab("⬡  AI Workspace"):

            openai_status = (
                "🟢 OpenAI-assisted routing is active for tricky questions."
                if is_openai_available()
                else "⚪ Running in rule-based mode — still fully functional for common questions."
            )

            gr.HTML(
                f"""
                <div class="feedback-header">
                    <div class="feedback-header-icon">◆</div>
                    <div>
                        <h3>AI Assistant</h3>
                        <p>Ask anything about your dataset — answers are computed directly from your data. {openai_status}</p>
                    </div>
                </div>
                """
            )

            # Suggestion chips (ChatGPT/Claude/Perplexity style). Clicking
            # one sends that question immediately. They disappear the
            # moment a conversation starts, so the chat stays the focus.
            with gr.Row(elem_classes="suggestion-chip-row") as suggestion_chips_row:
                chip_summarize = gr.Button("Summarize Dataset", elem_classes="suggestion-chip", size="sm")
                chip_missing = gr.Button("Explain Missing Values", elem_classes="suggestion-chip", size="sm")
                chip_outliers = gr.Button("Find Outliers", elem_classes="suggestion-chip", size="sm")
                chip_correlation = gr.Button("Correlation Analysis", elem_classes="suggestion-chip", size="sm")
                chip_cleaning = gr.Button("Cleaning Suggestions", elem_classes="suggestion-chip", size="sm")
                chip_target = gr.Button("Best Target Column", elem_classes="suggestion-chip", size="sm")

            with gr.Group(elem_classes="card workspace-chat-card"):
                chat_history_state = gr.State(value=[])
                chatbot_display = gr.Chatbot(
                    label="VizBot",
                    height=560,
                    placeholder="💬 **Ask a question about your dataset to get started.**\n\nExamples: *\"What is the average salary?\"* · *\"Summarize this dataset\"* · *\"Which columns have missing values?\"*",
                )

                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder="Ask a question about your dataset...",
                        show_label=False,
                        scale=4,
                    )
                    chat_send_button = gr.Button("📤 Send", variant="primary", scale=1)

                chat_clear_button = gr.Button("🗑 Clear Chat")

            with gr.Accordion("More examples", open=False, elem_classes="examples-popover"):
                gr.Markdown(get_example_questions_markdown())

        # ============================ TAB 5: VISUALIZATION STUDIO ============================
        with gr.Tab("▤  Visualization"):
            gr.Markdown(
                "### Build your own chart\n"
                "Pick a chart type and feature(s), then click **Generate Visualization**. "
                "Nothing is drawn automatically."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    chart_type_dropdown = gr.Dropdown(
                        choices=CHART_TYPES,
                        value="Histogram",
                        label="Chart Type",
                    )
                with gr.Column(scale=1):
                    feature1_dropdown = gr.Dropdown(
                        choices=[], value=None, label="Feature 1 (Numeric Column)",
                        visible=True, interactive=True,
                    )
                with gr.Column(scale=1):
                    feature2_dropdown = gr.Dropdown(
                        choices=[], value=None, label="Feature 2 (not needed for this chart)",
                        visible=True, interactive=False,
                    )

            # Proactively explains why a dropdown might be empty for this
            # chart type + dataset (e.g. no numeric columns at all) -
            # updates the moment the chart type or dataset changes, so
            # the user isn't left guessing.
            viz_availability_warning = gr.Markdown("")

            generate_viz_button = gr.Button("🎨 Generate Visualization", variant="primary")

            viz_status_output = gr.Markdown("")
            viz_plot_output = gr.Plot(label="Chart")

            # ---- Pivot Table Builder (Excel-style) ----
            gr.Markdown(
                "### Pivot Table Builder\n"
                "Configure **Rows**, **Columns**, **Values**, and an **Aggregation** — just like "
                "Excel's PivotTable — then click **Generate Pivot Table**. Add an optional **Filter** "
                "to narrow the data first."
            )

            with gr.Group(elem_classes="card pivot-card"):
                with gr.Row(elem_classes="pivot-field-row"):
                    pivot_rows_dropdown = gr.Dropdown(
                        choices=[], value=[], multiselect=True,
                        label="Rows", elem_classes="pivot-multiselect",
                    )
                    pivot_columns_dropdown = gr.Dropdown(
                        choices=[], value=[], multiselect=True,
                        label="Columns (optional)", elem_classes="pivot-multiselect",
                    )
                with gr.Row(elem_classes="pivot-field-row"):
                    pivot_values_dropdown = gr.Dropdown(
                        choices=[], value=[], multiselect=True,
                        label="Values", elem_classes="pivot-multiselect",
                    )
                    pivot_aggregation_dropdown = gr.Dropdown(
                        choices=PIVOT_AGGREGATIONS, value="Sum",
                        label="Aggregation", elem_classes="pivot-select",
                    )
                with gr.Row(elem_classes="pivot-field-row"):
                    pivot_filter_column_dropdown = gr.Dropdown(
                        choices=[], value=None,
                        label="Filter Field (optional)", elem_classes="pivot-select",
                    )
                    pivot_filter_value_dropdown = gr.Dropdown(
                        choices=[], value=[], multiselect=True,
                        label="Filter Values", elem_classes="pivot-multiselect",
                    )

                generate_pivot_button = gr.Button(
                    "📊 Generate Pivot Table", variant="primary", elem_classes="pivot-generate-button"
                )

            with gr.Group(elem_classes="card pivot-result-card"):
                pivot_status_output = gr.Markdown("")
                pivot_table_output = gr.Dataframe(label="Pivot Table", wrap=True)

        # ============================ TAB 6: FEEDBACK ============================
        with gr.Tab("✎  Feedback"):
            gr.HTML(
                """
                <div class="feedback-header">
                    <div class="feedback-header-icon">◆</div>
                    <div>
                        <h3>We'd love your feedback</h3>
                        <p>Help us improve DataNest — it takes less than a minute.</p>
                    </div>
                </div>
                """
            )

            with gr.Column(elem_classes="feedback-flow"):

                # ---- Step 1: Overall rating ----
                with gr.Column(elem_classes="feedback-section"):
                    gr.Markdown("**How was your experience?**")
                    rating_input = gr.Radio(
                        choices=[
                            ("1 · Poor", 1),
                            ("2", 2),
                            ("3", 3),
                            ("4", 4),
                            ("5 · Excellent", 5),
                        ],
                        show_label=False,
                        value=None,
                    )

                # ---- Step 2: AI helpfulness ----
                with gr.Column(elem_classes="feedback-section"):
                    gr.Markdown("**Were the AI insights useful?**")
                    ai_helpful_input = gr.Radio(
                        choices=[("Yes", "Yes"), ("No", "No")],
                        show_label=False,
                        value=None,
                    )

                # ---- Step 3: Visualization usefulness ----
                with gr.Column(elem_classes="feedback-section"):
                    gr.Markdown("**Did the visualizations help you understand your data?**")
                    viz_useful_input = gr.Radio(
                        choices=[("Yes", "Yes"), ("No", "No")],
                        show_label=False,
                        value=None,
                    )

                # ---- Step 4: Recommendation ----
                with gr.Column(elem_classes="feedback-section"):
                    gr.Markdown("**Would you recommend this platform?**")
                    recommend_input = gr.Radio(
                        choices=[("Yes", "Yes"), ("No", "No")],
                        show_label=False,
                        value=None,
                    )

                # ---- Step 5: Suggestions ----
                with gr.Column(elem_classes="feedback-section feedback-section-last"):
                    gr.Markdown("**Anything you'd like us to improve?**")
                    suggestions_input = gr.Textbox(
                        show_label=False,
                        placeholder="Optional — share your thoughts...",
                        lines=3,
                    )

                submit_feedback_button = gr.Button(
                    "Submit feedback", variant="primary", elem_classes="cta-button", size="lg"
                )
                feedback_status_output = gr.Markdown("")

        # ============================ TAB 7: ABOUT ============================
        with gr.Tab("ⓘ  About"):
            initial_stats, initial_latest_entries = get_feedback_stats()

            gr.Markdown(format_about_intro_markdown())

            gr.Markdown("### 📈 Community Feedback Summary")
            about_output = gr.HTML(format_about_kpi_cards(initial_stats))

            gr.Markdown("### 🕐 Latest 5 Feedback Entries")
            latest_entries_output = gr.Dataframe(
                value=initial_latest_entries,
                label="Recent Feedback",
                wrap=True,
            )

            refresh_about_button = gr.Button("🔄 Refresh Stats")

    # -----------------------------------------------------------------
    # Wire up the "Load Dataset" button to our callback function
    # -----------------------------------------------------------------
    load_button.click(
        fn=handle_file_upload,
        inputs=[file_input],
        outputs=[
            status_box, dataset_state, dashboard_output, preview_output,
            metadata_top_output, metric_selector_dropdown, bi_kpis_output, metadata_bottom_output,
            chart_type_dropdown, feature1_dropdown, feature2_dropdown, viz_plot_output, viz_status_output,
            upload_history_output, chatbot_display, chat_history_state,
            missing_summary_output, outlier_column_dropdown, cleaned_dataset_state,
            cleaning_summary_output, cleaned_preview_output, export_file_output,
            viz_availability_warning, suggestion_chips_row,
            visual_summary_column_dropdown, visual_summary_output, cleaning_plan_output,
            pivot_rows_dropdown, pivot_columns_dropdown, pivot_values_dropdown,
            pivot_filter_column_dropdown, pivot_filter_value_dropdown,
            pivot_table_output, pivot_status_output,
            missing_stats_output, missing_search_box, export_success_banner,
        ],
    )

    # -----------------------------------------------------------------
    # Metadata tab: picking a different metric in "Select Metric to
    # Analyze" recomputes ONLY the BI KPI cards - the rest of the
    # report (Statistical Summary, charts, Insights) stays untouched.
    # -----------------------------------------------------------------
    metric_selector_dropdown.change(
        fn=handle_metric_change,
        inputs=[metric_selector_dropdown, dataset_state],
        outputs=[bi_kpis_output],
    )

    # -----------------------------------------------------------------
    # Metadata tab: Quick Visual Summary - picking a column regenerates
    # just that column's chart(s), independent of everything else.
    # -----------------------------------------------------------------
    visual_summary_column_dropdown.change(
        fn=handle_visual_summary_change,
        inputs=[visual_summary_column_dropdown, dataset_state],
        outputs=[visual_summary_output],
    )

    # -----------------------------------------------------------------
    # When the user changes the Chart Type, refresh the Feature 1 /
    # Feature 2 dropdowns (different chart types need different
    # features, e.g. Bar Chart needs a categorical column while
    # Histogram needs a numeric one).
    # -----------------------------------------------------------------
    chart_type_dropdown.change(
        fn=handle_chart_type_change,
        inputs=[chart_type_dropdown, dataset_state],
        outputs=[feature1_dropdown, feature2_dropdown, viz_availability_warning],
    )

    # -----------------------------------------------------------------
    # The actual chart is ONLY built when this button is clicked -
    # never automatically after upload or after changing dropdowns.
    # -----------------------------------------------------------------
    generate_viz_button.click(
        fn=generate_visualization,
        inputs=[dataset_state, chart_type_dropdown, feature1_dropdown, feature2_dropdown],
        outputs=[viz_plot_output, viz_status_output],
    )

    # -----------------------------------------------------------------
    # Pivot Table Builder: picking a different Filter Field repopulates
    # the Filter Values checklist with that column's real values. The
    # pivot table itself is ONLY (re)built when "Generate Pivot Table"
    # is clicked - never automatically.
    # -----------------------------------------------------------------
    pivot_filter_column_dropdown.change(
        fn=handle_pivot_filter_column_change,
        inputs=[dataset_state, pivot_filter_column_dropdown],
        outputs=[pivot_filter_value_dropdown],
    )

    generate_pivot_button.click(
        fn=handle_generate_pivot,
        inputs=[
            dataset_state, pivot_rows_dropdown, pivot_columns_dropdown, pivot_values_dropdown,
            pivot_aggregation_dropdown, pivot_filter_column_dropdown, pivot_filter_value_dropdown,
        ],
        outputs=[pivot_table_output, pivot_status_output],
    )

    # -----------------------------------------------------------------
    # Submitting feedback saves it to feedback.csv AND immediately
    # refreshes the About tab so the stats never look stale.
    # -----------------------------------------------------------------
    submit_feedback_button.click(
        fn=handle_feedback_submit,
        inputs=[rating_input, ai_helpful_input, viz_useful_input, recommend_input, suggestions_input],
        outputs=[
            feedback_status_output, about_output, latest_entries_output,
            rating_input, ai_helpful_input, viz_useful_input, recommend_input, suggestions_input,
        ],
    )

    # -----------------------------------------------------------------
    # Manual refresh button on the About tab (e.g. if feedback.csv was
    # updated by someone else while the app was already open).
    # -----------------------------------------------------------------
    refresh_about_button.click(
        fn=handle_about_refresh,
        inputs=[],
        outputs=[about_output, latest_entries_output],
    )

    # -----------------------------------------------------------------
    # AI Dataset Chat (VizBot): the Send button, pressing Enter in the
    # textbox, and clicking any suggestion chip all trigger the same
    # underlying handler. Session memory lives in chat_history_state
    # (a gr.State), so it persists across turns for this browser
    # session but resets whenever a new dataset is loaded. The
    # suggestion chips hide themselves after the first message from
    # ANY of these paths, and reappear on Clear Chat or a new upload.
    # -----------------------------------------------------------------
    chat_send_button.click(
        fn=handle_chat_message,
        inputs=[chat_input, chat_history_state, dataset_state],
        outputs=[chatbot_display, chat_history_state, chat_input, suggestion_chips_row],
    )
    chat_input.submit(
        fn=handle_chat_message,
        inputs=[chat_input, chat_history_state, dataset_state],
        outputs=[chatbot_display, chat_history_state, chat_input, suggestion_chips_row],
    )
    chat_clear_button.click(
        fn=handle_chat_clear,
        inputs=[],
        outputs=[chatbot_display, chat_history_state, chat_input, suggestion_chips_row],
    )

    # Each suggestion chip sends its own fixed prompt through the exact
    # same handler as typing + Send - functools.partial pre-fills the
    # "question" argument so the click only needs to supply the two
    # remaining inputs (chat history + current dataset).
    _CHIP_PROMPTS = {
        chip_summarize: "Summarize this dataset",
        chip_missing: "Which columns have missing values?",
        chip_outliers: "Does the dataset contain outliers?",
        chip_correlation: "Show correlation",
        chip_cleaning: "Recommend preprocessing steps",
        chip_target: "What is the best target column?",
    }
    for chip_button, preset_prompt in _CHIP_PROMPTS.items():
        chip_button.click(
            fn=functools.partial(handle_chat_message, preset_prompt),
            inputs=[chat_history_state, dataset_state],
            outputs=[chatbot_display, chat_history_state, chat_input, suggestion_chips_row],
        )

    # -----------------------------------------------------------------
    # Data Cleaning tab: detecting outliers is read-only (never
    # modifies data). Applying cleaning writes to its OWN state
    # (cleaned_dataset_state) rather than the shared dataset_state, so
    # Dashboard / Visualization / AI Chat always keep working on the
    # original, untouched data. Export only works once cleaning has
    # been applied at least once.
    # -----------------------------------------------------------------
    detect_outliers_button.click(
        fn=handle_detect_outliers,
        inputs=[dataset_state, outlier_column_dropdown],
        outputs=[outlier_status_output],
    )
    apply_cleaning_button.click(
        fn=handle_apply_cleaning,
        inputs=[
            dataset_state, missing_strategy_dropdown,
            outlier_strategy_radio, outlier_column_dropdown,
        ],
        outputs=[
            cleaned_dataset_state, cleaning_summary_output, cleaned_preview_output, missing_summary_output,
            missing_stats_output, export_success_banner, export_file_output,
        ],
    )
    export_csv_button.click(
        fn=handle_export_csv,
        inputs=[cleaned_dataset_state],
        outputs=[export_success_banner, export_file_output],
    )
    export_excel_button.click(
        fn=handle_export_excel,
        inputs=[cleaned_dataset_state],
        outputs=[export_success_banner, export_file_output],
    )

    # Missing-values search box: filters the table only - the mini-stat
    # row above it always describes the whole dataset, so it's left
    # untouched by this event.
    missing_search_box.change(
        fn=handle_missing_search,
        inputs=[dataset_state, missing_search_box],
        outputs=[missing_summary_output],
    )

    # -----------------------------------------------------------------
    # Cleaning Summary: a live "here's what will happen" preview that
    # updates instantly whenever ANY cleaning control changes - purely
    # informational, never modifies data (that only happens when
    # "Apply Cleaning" is clicked).
    # -----------------------------------------------------------------
    _cleaning_plan_inputs = [
        dataset_state, missing_strategy_dropdown,
        outlier_strategy_radio, outlier_column_dropdown,
    ]
    missing_strategy_dropdown.change(
        fn=handle_cleaning_plan_preview, inputs=_cleaning_plan_inputs, outputs=[cleaning_plan_output],
    )
    outlier_column_dropdown.change(
        fn=handle_cleaning_plan_preview, inputs=_cleaning_plan_inputs, outputs=[cleaning_plan_output],
    )
    outlier_strategy_radio.change(
        fn=handle_cleaning_plan_preview, inputs=_cleaning_plan_inputs, outputs=[cleaning_plan_output],
    )

    # -----------------------------------------------------------------
    # Theme toggle: pure client-side JS (fn=None means no server round
    # trip) - see THEME_TOGGLE_JS above for what it actually does.
    # -----------------------------------------------------------------
    theme_toggle_button.click(fn=None, js=THEME_TOGGLE_JS)


# ----------------------------------------------------------------------
# LAUNCH
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import os

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )