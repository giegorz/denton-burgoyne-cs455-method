import os

import pandas as pd


def write_sheet_to_excel(
    filepath: str,
    df: pd.DataFrame,
    sheet_name: str,
) -> None:
    """
    Write `df` as a sheet named `sheet_name` in the .xlsx workbook at
    `filepath`. If the workbook already exists, the sheet is added to it
    (replacing any existing sheet with the same name) and every other
    sheet is left untouched. If the workbook doesn't exist yet, it is
    created with just this one sheet.
    """
    mode = "a" if os.path.exists(filepath) else "w"
    writer_kwargs = {"engine": "openpyxl", "mode": mode}
    if mode == "a":
        writer_kwargs["if_sheet_exists"] = "replace"

    with pd.ExcelWriter(filepath, **writer_kwargs) as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
