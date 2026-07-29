"""
upload_history.py
------------------
This module keeps a permanent log of every dataset the user has
uploaded, so nothing is ever lost or overwritten.

Every time a file is SUCCESSFULLY loaded, we append ONE new row to
`upload_history.csv` (created automatically on first use) containing:
    - Filename
    - Timestamp
    - Rows
    - Columns

This mirrors the same pattern used in feedback.py: an absolute path
so it works regardless of the current working directory, and an
append-only write so history is never lost.
"""

import os
import pandas as pd
from datetime import datetime


# The CSV columns, in the order we will always write/read them.
HISTORY_COLUMNS = ["Filename", "Timestamp", "Rows", "Columns"]


def _get_history_file_path() -> str:
    """
    Returns the absolute path to upload_history.csv, placed at the
    project root (one level above this `modules/` folder) - same
    pattern as feedback.csv.
    """
    modules_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(modules_dir)
    return os.path.join(project_root, "upload_history.csv")


def log_upload(filename: str, rows: int, columns: int) -> bool:
    """
    Appends ONE new upload record to upload_history.csv.
    This NEVER overwrites previous entries - every upload becomes a
    brand new row, even if the same file is uploaded twice.

    Parameters
    ----------
    filename : str   the original name of the uploaded file
    rows : int        number of rows in the loaded dataset
    columns : int     number of columns in the loaded dataset

    Returns
    -------
    success : bool   True if the entry was saved, False on error.
    """
    file_path = _get_history_file_path()

    new_entry = pd.DataFrame([{
        "Filename": filename,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Rows": rows,
        "Columns": columns,
    }], columns=HISTORY_COLUMNS)

    try:
        file_exists = os.path.isfile(file_path)
        # mode="a" appends; header is only written the first time the file is created.
        new_entry.to_csv(file_path, mode="a", header=not file_exists, index=False)
        return True
    except Exception:
        return False


def get_upload_history(latest_first: bool = True) -> pd.DataFrame:
    """
    Reads the full upload history back from disk.

    Parameters
    ----------
    latest_first : bool   if True, most recent upload appears at the top.

    Returns
    -------
    history_df : pandas.DataFrame
        Returns an empty (but correctly headed) DataFrame if no
        uploads have been logged yet or the file can't be read.
    """
    file_path = _get_history_file_path()

    if not os.path.isfile(file_path):
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    try:
        df = pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    if latest_first:
        df = df.iloc[::-1].reset_index(drop=True)

    return df
