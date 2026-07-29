"""
ai_readiness.py
---------------
This module calculates the "AI Readiness Score" (0-100) for the
AI Readiness tab — a single number that tells the user, at a glance,
how ready their dataset is to be used for AI / Machine Learning work.

SCORING LOGIC (starts at 100, points are deducted for problems):

    1. Missing Values Penalty   (up to -40 points)
       More missing data -> harder for models to learn -> bigger penalty.

    2. Duplicate Records Penalty (up to -30 points)
       Duplicate rows can bias a model and inflate accuracy metrics.

    3. Feature Type Penalty      (up to -15 points)
       A dataset with ZERO numeric columns is much harder to model
       directly (needs heavy preprocessing), so we penalize that.

Final score is clamped between 0 and 100, then mapped to a status label:

    90 - 100  -> Excellent
    75 - 89   -> Good
    50 - 74   -> Average
    0  - 49   -> Poor

Keeping the *formula* in one place means we can tune the thresholds
later without touching the UI code in app.py.
"""

import pandas as pd


def calculate_ai_readiness(df: pd.DataFrame) -> dict:
    """
    Calculates the AI Readiness score and status for a given dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset currently loaded in the app.

    Returns
    -------
    result : dict
        {
            "score": int (0-100),
            "status": str ("Excellent" / "Good" / "Average" / "Poor"),
            "missing_penalty": float,
            "duplicate_penalty": float,
            "feature_penalty": float,
        }
        Returns a "no data" style result if df is None.
    """

    if df is None or df.empty:
        return {
            "score": 0,
            "status": "No Data",
            "missing_penalty": 0,
            "duplicate_penalty": 0,
            "feature_penalty": 0,
        }

    total_rows, total_columns = df.shape
    total_cells = total_rows * total_columns

    # ------------------------------------------------------------
    # 1. Missing Values Penalty (max 40 points)
    # ------------------------------------------------------------
    total_missing = df.isnull().sum().sum()
    missing_percentage = (total_missing / total_cells) * 100 if total_cells > 0 else 0
    # Every 1% missing costs 1 point, capped at 40 points total.
    missing_penalty = min(missing_percentage, 40)

    # ------------------------------------------------------------
    # 2. Duplicate Records Penalty (max 30 points)
    # ------------------------------------------------------------
    duplicate_count = df.duplicated().sum()
    duplicate_percentage = (duplicate_count / total_rows) * 100 if total_rows > 0 else 0
    # Every 1% duplicate rows costs 1 point, capped at 30 points total.
    duplicate_penalty = min(duplicate_percentage, 30)

    # ------------------------------------------------------------
    # 3. Feature Type Penalty (max 15 points)
    # ------------------------------------------------------------
    numeric_features = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_ratio = len(numeric_features) / total_columns if total_columns > 0 else 0

    if len(numeric_features) == 0:
        # No numeric columns at all -> hardest case for most ML models.
        feature_penalty = 15
    elif numeric_ratio < 0.2:
        # Very few numeric columns -> moderate penalty.
        feature_penalty = 8
    else:
        feature_penalty = 0

    # ------------------------------------------------------------
    # Final Score
    # ------------------------------------------------------------
    raw_score = 100 - missing_penalty - duplicate_penalty - feature_penalty
    score = int(round(max(0, min(100, raw_score))))

    # ------------------------------------------------------------
    # Status Label
    # ------------------------------------------------------------
    if score >= 90:
        status = "Excellent"
    elif score >= 75:
        status = "Good"
    elif score >= 50:
        status = "Average"
    else:
        status = "Poor"

    return {
        "score": score,
        "status": status,
        "missing_penalty": round(missing_penalty, 2),
        "duplicate_penalty": round(duplicate_penalty, 2),
        "feature_penalty": round(feature_penalty, 2),
    }


def format_ai_readiness_as_markdown(result: dict) -> str:
    """
    Converts the AI Readiness result dict into a clean Markdown block
    for display inside the AI Readiness tab.

    (Kept for backward compatibility - the AI Readiness tab itself now
    uses format_ai_readiness_as_html() below, which renders as a
    circular gauge + score card instead of plain text.)
    """
    if not result or result.get("status") == "No Data":
        return "### ⚠ No dataset loaded yet.\nPlease upload a CSV or Excel file and click **Load Dataset**."

    # Pick an emoji that matches the status, purely for visual feedback.
    status_emojis = {
        "Excellent": "🟢",
        "Good": "🟩",
        "Average": "🟡",
        "Poor": "🔴",
    }
    emoji = status_emojis.get(result["status"], "⚪")

    md = "### 🧠 AI Readiness Report\n\n"
    md += f"## {emoji} Score: {result['score']} / 100\n"
    md += f"### Status: **{result['status']}**\n\n"
    md += "---\n"
    md += "**How this score was calculated (points deducted):**\n\n"
    md += "| Factor | Points Deducted |\n"
    md += "|---|---|\n"
    md += f"| Missing Values | -{result['missing_penalty']} |\n"
    md += f"| Duplicate Records | -{result['duplicate_penalty']} |\n"
    md += f"| Feature Type Balance | -{result['feature_penalty']} |\n"

    return md


# Status -> accent color, used by the HTML score card below. Kept in
# one place so the gauge ring, badge pill, and status text always
# agree on which color represents which status.
_STATUS_COLORS = {
    "Excellent": "#22c55e",   # emerald
    "Good": "#6366f1",        # indigo
    "Average": "#f59e0b",     # amber
    "Poor": "#f43f5e",        # rose
}


def format_ai_readiness_as_html(result: dict) -> str:
    """
    Renders the AI Readiness result as an analytics-style "score card":
    a circular gauge (built with a conic-gradient, no JS/canvas needed)
    showing the 0-100 score, a colored status badge, and three small
    progress bars breaking down exactly where points were lost.

    Returned as raw HTML for a gr.HTML() component. Styling classes
    (.readiness-*) live in app.py's CUSTOM_CSS.
    """
    if not result or result.get("status") == "No Data":
        return (
            "<div class='kpi-empty'>"
            "No dataset loaded yet. Please upload a CSV or Excel file and click "
            "<b>Load Dataset</b>."
            "</div>"
        )

    score = result["score"]
    status = result["status"]
    color = _STATUS_COLORS.get(status, "#6366f1")

    # The conic-gradient needs an angle proportional to the score
    # (100 points = full circle = 360deg).
    angle = round((score / 100) * 360, 1)

    # Each penalty is shown as a small bar out of its own max points,
    # so a "-15 / 15" penalty reads as a full (bad) bar while a
    # "-2 / 40" penalty reads as a mostly-empty (good) bar.
    penalty_bars = [
        ("Missing Values", result["missing_penalty"], 40),
        ("Duplicate Records", result["duplicate_penalty"], 30),
        ("Feature Type Balance", result["feature_penalty"], 15),
    ]

    bars_html = ""
    for label, penalty, max_points in penalty_bars:
        fill_pct = min(100, round((penalty / max_points) * 100)) if max_points > 0 else 0
        bars_html += f"""
        <div class="readiness-bar-row">
            <div class="readiness-bar-label">
                <span>{label}</span>
                <span class="readiness-bar-value">-{penalty}</span>
            </div>
            <div class="readiness-bar-track">
                <div class="readiness-bar-fill" style="width: {fill_pct}%;"></div>
            </div>
        </div>
        """

    html = f"""
    <div class="readiness-card">
        <div class="readiness-gauge" style="background: conic-gradient({color} {angle}deg, var(--readiness-track-color) {angle}deg);">
            <div class="readiness-gauge-inner">
                <div class="readiness-score">{score}</div>
                <div class="readiness-score-max">/ 100</div>
            </div>
        </div>
        <div class="readiness-details">
            <span class="readiness-badge" style="background: {color}22; color: {color}; border: 1px solid {color}55;">
                {status}
            </span>
            <div class="readiness-bars">
                {bars_html}
            </div>
        </div>
    </div>
    """
    return html


# ----------------------------------------------------------------------
# AI WORKSPACE — SECTION 1 EXTRAS: explanation / strengths / blockers /
# suggested improvements (shown right below the gauge card)
# ----------------------------------------------------------------------

def get_readiness_explanation(result: dict) -> str:
    """One short paragraph explaining what the score means in plain English."""
    if not result or result.get("status") == "No Data":
        return ""

    status = result["status"]
    score = result["score"]
    explanations = {
        "Excellent": (
            f"With a score of {score}/100, this dataset is in excellent shape — minimal missing "
            "data, no duplicate records, and a healthy mix of feature types. It's ready for most "
            "AI/ML workflows with little to no additional cleanup."
        ),
        "Good": (
            f"With a score of {score}/100, this dataset is in good condition. A few minor issues "
            "exist, but they're unlikely to seriously impact most modeling tasks."
        ),
        "Average": (
            f"With a score of {score}/100, this dataset has some real quality issues worth "
            "addressing before serious model training — see the recommendations below."
        ),
        "Poor": (
            f"With a score of {score}/100, this dataset has significant quality issues (missing "
            "data, duplicates, and/or feature imbalance). Training directly on this data is likely "
            "to produce unreliable results — fix the issues below first."
        ),
    }
    return explanations.get(status, "")


def get_dataset_strengths(df: pd.DataFrame, result: dict) -> list:
    """Things the dataset is doing well, in plain English."""
    if df is None or df.empty:
        return []

    strengths = []
    total_rows, total_columns = df.shape
    total_missing = int(df.isnull().sum().sum())
    duplicate_count = int(df.duplicated().sum())
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    if total_missing == 0:
        strengths.append("No missing values anywhere in the dataset.")
    elif result.get("missing_penalty", 0) < 5:
        strengths.append("Very few missing values overall.")

    if duplicate_count == 0:
        strengths.append("No duplicate records detected.")

    if numeric_cols and categorical_cols:
        strengths.append(f"Healthy mix of {len(numeric_cols)} numeric and {len(categorical_cols)} categorical column(s).")
    elif numeric_cols and not categorical_cols:
        strengths.append("Every column is numeric — no encoding required before modeling.")

    if total_rows >= 500:
        strengths.append(f"Good sample size with {total_rows:,} rows.")

    return strengths


def get_dataset_blockers(df: pd.DataFrame, result: dict) -> list:
    """Things currently preventing the dataset from being fully AI-ready."""
    if df is None or df.empty:
        return []

    blockers = []
    total_rows, total_columns = df.shape
    total_cells = total_rows * total_columns
    total_missing = int(df.isnull().sum().sum())
    duplicate_count = int(df.duplicated().sum())
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if result.get("missing_penalty", 0) > 0:
        missing_pct = round((total_missing / total_cells) * 100, 1) if total_cells > 0 else 0
        blockers.append(f"{missing_pct}% missing values across the dataset.")
    if duplicate_count > 0:
        blockers.append(f"{duplicate_count} duplicate record(s) found.")
    if not numeric_cols:
        blockers.append("No numeric columns — most ML models need at least some numeric features.")
    elif total_columns > 0 and len(numeric_cols) / total_columns < 0.2:
        blockers.append("Very few numeric columns relative to the total number of columns.")

    return blockers


def get_suggested_improvements(df: pd.DataFrame, result: dict) -> list:
    """A short, prioritized list (max 4 items) for Section 1 - the
    detailed, priority-badged version lives in Section 2 below."""
    if df is None or df.empty:
        return []

    improvements = []
    if result.get("missing_penalty", 0) > 0:
        improvements.append("Handle missing values (drop rows, or fill with mean/median/mode).")
    if result.get("duplicate_penalty", 0) > 0:
        improvements.append("Remove duplicate records.")
    if result.get("feature_penalty", 0) > 0:
        improvements.append("Improve feature-type balance (e.g. ensure some numeric columns exist).")
    if not improvements:
        improvements.append("No major improvements needed — this dataset is ready to use.")
    return improvements[:4]


def format_readiness_details_html(df: pd.DataFrame, result: dict) -> str:
    """
    Renders the explanation + strengths/blockers/improvements that sit
    below the gauge card in Section 1. Returns an empty string when no
    dataset is loaded (the gauge card above already shows its own
    empty state, so we don't want a second one stacked underneath it).
    """
    if df is None or df.empty or not result or result.get("status") == "No Data":
        return ""

    explanation = get_readiness_explanation(result)
    strengths = get_dataset_strengths(df, result)
    blockers = get_dataset_blockers(df, result)
    improvements = get_suggested_improvements(df, result)

    strengths_html = "".join(f"<li>{s}</li>" for s in strengths) or "<li>No specific strengths detected.</li>"
    blockers_html = "".join(f"<li>{b}</li>" for b in blockers) or "<li>No blocking issues detected — dataset looks solid.</li>"
    improvements_html = "".join(f"<li>{i}</li>" for i in improvements)

    return f"""
    <div class="readiness-explanation">{explanation}</div>
    <div class="readiness-columns">
        <div class="readiness-column">
            <div class="readiness-column-title strengths-title">Key Strengths</div>
            <ul class="readiness-list strengths-list">{strengths_html}</ul>
        </div>
        <div class="readiness-column">
            <div class="readiness-column-title blockers-title">Blocking Issues</div>
            <ul class="readiness-list blockers-list">{blockers_html}</ul>
        </div>
    </div>
    <div class="readiness-improvements">
        <div class="readiness-column-title">Suggested Improvements</div>
        <ul class="readiness-list">{improvements_html}</ul>
    </div>
    """


# ----------------------------------------------------------------------
# AI WORKSPACE — SECTION 2: Smart Recommendations (priority-badged cards)
# ----------------------------------------------------------------------

# Priority label -> accent color, shared by every recommendation card.
_PRIORITY_COLORS = {
    "High Priority": "#f43f5e",
    "Medium Priority": "#f59e0b",
    "Optional": "#6366f1",
}


def get_smart_recommendations(df: pd.DataFrame) -> list:
    """
    Generates actionable, prioritized recommendations based on what's
    actually detected in the dataset - each one only appears if it's
    actually relevant (e.g. no "remove duplicates" card if there are
    none).

    Returns a list of dicts: {"title": str, "description": str, "priority": str}
    """
    if df is None or df.empty:
        return []

    recommendations = []
    total_rows, total_columns = df.shape
    total_cells = total_rows * total_columns
    total_missing = int(df.isnull().sum().sum())
    missing_pct = round((total_missing / total_cells) * 100, 1) if total_cells > 0 else 0
    duplicate_count = int(df.duplicated().sum())
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # ---- Missing values ----
    if total_missing > 0:
        priority = "High Priority" if missing_pct > 15 else "Medium Priority"
        recommendations.append({
            "title": "Handle Missing Values",
            "description": (
                f"{missing_pct}% of cells are missing. Consider dropping affected rows, or filling "
                "numeric columns with mean/median and categorical columns with the most frequent value."
            ),
            "priority": priority,
        })

    # ---- Duplicates ----
    if duplicate_count > 0:
        dup_pct = round((duplicate_count / total_rows) * 100, 1) if total_rows > 0 else 0
        priority = "High Priority" if dup_pct > 5 else "Medium Priority"
        recommendations.append({
            "title": "Remove Duplicate Records",
            "description": (
                f"{duplicate_count} duplicate row(s) found ({dup_pct}% of the dataset). Removing them "
                "prevents skewed statistics and biased model training."
            ),
            "priority": priority,
        })

    # ---- Categorical encoding ----
    if categorical_cols:
        example_cols = ", ".join(categorical_cols[:4])
        recommendations.append({
            "title": "Encode Categorical Columns",
            "description": (
                f"{len(categorical_cols)} categorical column(s) ({example_cols}) will need encoding "
                "(one-hot or label encoding) before most ML models can use them."
            ),
            "priority": "Medium Priority",
        })

    # ---- Scaling ----
    if len(numeric_cols) >= 2:
        recommendations.append({
            "title": "Scale Numeric Features",
            "description": (
                "Numeric columns often have very different ranges. Standardizing or normalizing "
                "them can improve performance for distance-based or gradient-based models."
            ),
            "priority": "Optional",
        })

    # ---- Identifier columns ----
    id_candidates = [
        col for col in df.columns
        if total_rows > 0 and df[col].nunique(dropna=True) == total_rows
    ]
    if id_candidates:
        recommendations.append({
            "title": "Remove Identifier Columns",
            "description": (
                f"<b>{id_candidates[0]}</b> appears to be a unique identifier for every row. "
                "Exclude it from modeling — it won't generalize to new data."
            ),
            "priority": "Medium Priority",
        })

    # ---- Outliers (simple IQR check across up to 8 numeric columns) ----
    outlier_hits = []
    for col in numeric_cols[:8]:
        series = df[col].dropna()
        if len(series) < 4:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_count = int(((series < lower_bound) | (series > upper_bound)).sum())
        if outlier_count > 0:
            outlier_hits.append((col, outlier_count))
    if outlier_hits:
        top_col, top_count = max(outlier_hits, key=lambda x: x[1])
        recommendations.append({
            "title": "Detect and Review Outliers",
            "description": (
                f"<b>{top_col}</b> has {top_count} potential outlier(s) (1.5×IQR rule). Review "
                "whether these are legitimate values or data errors before training."
            ),
            "priority": "Medium Priority",
        })

    # ---- Feature engineering (generic, always relevant as an optional idea) ----
    recommendations.append({
        "title": "Consider Feature Engineering",
        "description": (
            "Look for opportunities to create new features — combining columns, extracting date "
            "parts, or grouping high-cardinality categories — to improve model performance."
        ),
        "priority": "Optional",
    })

    if not recommendations:
        recommendations.append({
            "title": "Dataset Looks Ready",
            "description": "No major issues detected. You can proceed to modeling with minimal preprocessing.",
            "priority": "Optional",
        })

    return recommendations


def format_smart_recommendations_html(recommendations: list) -> str:
    """Renders each recommendation as a card with a colored priority badge."""
    if not recommendations:
        return "<div class='kpi-empty'>Upload a dataset to see smart recommendations.</div>"

    cards = ""
    for rec in recommendations:
        color = _PRIORITY_COLORS.get(rec["priority"], "#6366f1")
        cards += f"""
        <div class="rec-card">
            <div class="rec-card-header">
                <span class="rec-title">{rec['title']}</span>
                <span class="rec-priority-badge" style="background:{color}22;color:{color};border:1px solid {color}55;">{rec['priority']}</span>
            </div>
            <div class="rec-description">{rec['description']}</div>
        </div>
        """
    return f"<div class='rec-grid'>{cards}</div>"
