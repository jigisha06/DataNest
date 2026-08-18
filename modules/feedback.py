"""
feedback.py
-----------
This module powers BOTH the "Feedback" tab and the "About" tab.

Feedback tab responsibilities:
    - Take the user's Overall Rating (1-5), Was AI Helpful?,
      Were Visualizations Useful?, Would you Recommend this App?,
      and free-text Suggestions.
    - Append that as one row to `feedback.csv` (created automatically
      the first time someone submits feedback).

About tab responsibilities:
    - Read `feedback.csv` back in.
    - Calculate: Total Users, Average Rating, % who found AI helpful,
      % satisfied with visualizations, % who'd recommend the app.
    - Show the latest 5 feedback entries (most recent first).

Keeping storage + aggregation together in one module makes sense here
because both tabs revolve around the exact same CSV file/schema.
"""

import os
import pandas as pd
from datetime import datetime


# The CSV columns, in the order we will always write/read them.
FEEDBACK_COLUMNS = [
    "Timestamp",
    "Rating",
    "AI_Helpful",
    "Visualizations_Useful",
    "Would_Recommend",
    "Suggestions",
]


def _get_feedback_file_path() -> str:
    """
    Returns the absolute path to feedback.csv, placed at the project
    root (one level above this `modules/` folder). Using an absolute
    path means feedback is saved/read correctly no matter what folder
    the app was launched from.
    """
    modules_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(modules_dir)
    return os.path.join(project_root, "feedback.csv")


def submit_feedback(rating, ai_helpful, viz_useful, would_recommend, suggestions) -> str:
    """
    Appends one feedback entry to feedback.csv (creating the file with
    a header row the very first time).

    Parameters
    ----------
    rating : int or str          e.g. 5
    ai_helpful : str              "Yes" or "No"
    viz_useful : str               "Yes" or "No"
    would_recommend : str          "Yes" or "No"
    suggestions : str               free text, may be empty

    Returns
    -------
    status_message : str  - shown to the user after clicking Submit.
    """

    # ---- Basic validation so we never save a half-filled entry ----
    if rating is None:
        return "⚠ Please select an Overall Rating before submitting."
    if not ai_helpful:
        return "⚠ Please answer 'Was AI Helpful?' before submitting."
    if not viz_useful:
        return "⚠ Please answer 'Were Visualizations Useful?' before submitting."
    if not would_recommend:
        return "⚠ Please answer 'Would you Recommend this App?' before submitting."

    file_path = _get_feedback_file_path()

    new_entry = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Rating": int(rating),
        "AI_Helpful": ai_helpful,
        "Visualizations_Useful": viz_useful,
        "Would_Recommend": would_recommend,
        "Suggestions": suggestions.strip() if suggestions else "",
    }], columns=FEEDBACK_COLUMNS)

    try:
        file_exists = os.path.isfile(file_path)
        # mode="a" appends; header is only written the first time the file is created.
        new_entry.to_csv(file_path, mode="a", header=not file_exists, index=False)
        return "✅ Thank you! Your feedback has been recorded."
    except Exception as e:
        return f" Could not save feedback: {str(e)}"


def _load_feedback_dataframe() -> pd.DataFrame:
    """
    Reads feedback.csv from disk. Returns an empty (but correctly
    headed) DataFrame if the file doesn't exist yet or is unreadable.
    """
    file_path = _get_feedback_file_path()

    if not os.path.isfile(file_path):
        return pd.DataFrame(columns=FEEDBACK_COLUMNS)

    try:
        df = pd.read_csv(file_path)
        return df
    except Exception:
        return pd.DataFrame(columns=FEEDBACK_COLUMNS)


def get_feedback_stats():
    """
    Computes the aggregated stats shown on the About tab.

    Returns
    -------
    stats : dict       summary numbers (see keys below)
    latest_entries : pandas.DataFrame   the 5 most recent submissions
    """
    df = _load_feedback_dataframe()

    if df.empty:
        stats = {
            "total_users": 0,
            "average_rating": 0,
            "pct_ai_helpful": 0,
            "pct_viz_useful": 0,
            "pct_recommend": 0,
        }
        return stats, pd.DataFrame(columns=FEEDBACK_COLUMNS)

    total_users = len(df)
    average_rating = round(df["Rating"].mean(), 2)
    pct_ai_helpful = round((df["AI_Helpful"] == "Yes").mean() * 100, 1)
    pct_viz_useful = round((df["Visualizations_Useful"] == "Yes").mean() * 100, 1)
    pct_recommend = round((df["Would_Recommend"] == "Yes").mean() * 100, 1)

    stats = {
        "total_users": total_users,
        "average_rating": average_rating,
        "pct_ai_helpful": pct_ai_helpful,
        "pct_viz_useful": pct_viz_useful,
        "pct_recommend": pct_recommend,
    }

    # Most recent submissions first.
    latest_entries = df.tail(5).iloc[::-1].reset_index(drop=True)

    return stats, latest_entries


def format_about_markdown(stats: dict) -> str:
    """
    Converts the aggregated feedback stats dict into a Markdown block
    for display on the About tab.

    (Kept for backward compatibility - the About tab itself now uses
    format_about_intro_markdown() + format_about_kpi_cards() below,
    which render as a nicer HTML card grid instead of a plain table.)
    """
    if not stats or stats.get("total_users", 0) == 0:
        return (
            "### ℹ️ About This App\n\n"
            "**DataNest** helps you explore, understand, and visualize "
            "your datasets in minutes — no code required.\n\n"
            "No feedback has been submitted yet. Be the first! 🎉"
        )

    md = "### ℹ️ About This App\n\n"
    md += (
        "**DataNest** helps you explore, understand, and visualize "
        "your datasets in minutes — no code required.\n\n"
    )
    md += "### 📈 Community Feedback Summary\n\n"
    md += "| Metric | Value |\n"
    md += "|---|---|\n"
    md += f"| **Total Users** | {stats['total_users']} |\n"
    md += f"| **Average Rating** | {stats['average_rating']} / 5 ⭐ |\n"
    md += f"| **AI Found Helpful** | {stats['pct_ai_helpful']}% |\n"
    md += f"| **Satisfied with Visualizations** | {stats['pct_viz_useful']}% |\n"
    md += f"| **Would Recommend the App** | {stats['pct_recommend']}% |\n"

    return md


def format_about_intro_markdown() -> str:
    """
    The static app description shown at the top of the About tab -
    doesn't depend on any feedback data, so it's separated out from
    the dynamic KPI cards below.
    """
    return (
        "### ℹ️ About This App\n\n"
        "**DataNest** helps you explore, understand, and visualize "
        "your datasets in minutes — no code required."
    )


def format_about_kpi_cards(stats: dict) -> str:
    """
    Converts the aggregated feedback stats into an HTML grid of KPI
    cards - the same modern look used on the Dashboard tab - instead
    of a plain markdown table.

    Returned as raw HTML so it can be dropped straight into a Gradio
    gr.HTML() component. Styling (.kpi-grid / .kpi-card / .kpi-empty)
    lives in app.py's CUSTOM_CSS, shared with the Dashboard tab.
    """
    if not stats or stats.get("total_users", 0) == 0:
        return (
            "<div class='kpi-empty'>"
            "No feedback has been submitted yet. Be the first! 🎉"
            "</div>"
        )

    # (icon, label, value) - the order they'll appear on the About tab.
    kpi_definitions = [
        ("👥", "Total Users", stats["total_users"]),
        ("⭐", "Average Rating", f"{stats['average_rating']} / 5"),
        ("🤖", "Found AI Helpful", f"{stats['pct_ai_helpful']}%"),
        ("📊", "Satisfied with Visualizations", f"{stats['pct_viz_useful']}%"),
        ("👍", "Would Recommend", f"{stats['pct_recommend']}%"),
    ]

    cards_html = ""
    for icon, label, value in kpi_definitions:
        cards_html += f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{label}</div>
        </div>
        """

    return f'<div class="kpi-grid">{cards_html}</div>'
    md += "|---|---|\n"
    md += f"| **Total Users** | {stats['total_users']} |\n"
    md += f"| **Average Rating** | {stats['average_rating']} / 5 ⭐ |\n"
    md += f"| **AI Found Helpful** | {stats['pct_ai_helpful']}% |\n"
    md += f"| **Satisfied with Visualizations** | {stats['pct_viz_useful']}% |\n"
    md += f"| **Would Recommend the App** | {stats['pct_recommend']}% |\n"

    return md
