"""
data_handler.py
----------------
This module is responsible for ONE job only: loading a dataset
(CSV or Excel) into a pandas DataFrame.

Keeping this logic in its own file (instead of inside app.py) is what
makes the project "modular" - every other module/tab will simply
receive the DataFrame that this module produces, they never need to
know HOW the file was read.
"""

import pandas as pd
import os


def load_dataset(file_path: str):
    """
    Loads a dataset from a CSV or Excel file into a pandas DataFrame.

    Parameters
    ----------
    file_path : str
        The path to the uploaded file (Gradio gives us a temp file path).

    Returns
    -------
    df : pandas.DataFrame or None
        The loaded dataset. None if loading failed.
    message : str
        A human-readable status message (success or error) that we can
        show directly in the Gradio UI.
    """

    # Guard clause: if no file was uploaded yet, do nothing.
    if file_path is None:
        return None, "⚠ Please upload a CSV or Excel file first."

    # Get the file extension (e.g. ".csv" or ".xlsx") in lowercase
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    try:
        if file_extension == ".csv":
            # Standard CSV read. `low_memory=False` avoids mixed dtype
            # warnings on large files.
            df = pd.read_csv(file_path, low_memory=False)

        elif file_extension in [".xlsx", ".xls"]:
            # openpyxl is used automatically by pandas as the engine
            # for .xlsx files (make sure it's installed).
            df = pd.read_excel(file_path, engine="openpyxl")

        else:
            return None, (
                f"❌ Unsupported file type '{file_extension}'. "
                "Please upload a .csv or .xlsx file."
            )

        # Basic sanity check: make sure the file wasn't empty.
        if df.empty:
            return None, "⚠ The uploaded file is empty. Please check your data."

        rows, cols = df.shape
        success_message = f"✅ File loaded successfully! ({rows} rows, {cols} columns)"
        return df, success_message

    except Exception as e:
        # Catch-all so the app never crashes on a bad file — instead
        # the user sees a friendly error message.
        return None, f"❌ Error while reading the file: {str(e)}"
