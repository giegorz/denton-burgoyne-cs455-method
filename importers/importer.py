from typing import List

import pandas as pd
from enum import Enum

class SheetNames(str, Enum):
    NODES = "nodes"
    ELEMENTS = "elements"
    RESULTS = "results"

class ColumnsNames(str, Enum):
    NODES = "A:C"
    ELEMENTS = "A,G:N"
    RESULTS = "A,B,C,J:L"

def import_data(source, sheet_name: str) -> pd.DataFrame:
    _map = {
        SheetNames.NODES: ColumnsNames.NODES.value,
        SheetNames.ELEMENTS: ColumnsNames.ELEMENTS.value,
        SheetNames.RESULTS: ColumnsNames.RESULTS.value,
    }

    try:
        return pd.read_excel(source, sheet_name=sheet_name, usecols=_map[sheet_name])
    except KeyError:
        raise KeyError(f"The sheet {sheet_name.value} does not exist")

def import_nodes(path: str) -> pd.DataFrame:
    df = import_data(path, SheetNames.NODES)

    df.rename(columns={"Node": "node",
                       "X(m)": "x",
                       "Y(m)": "y"},
              inplace=True
              )
    try:
        return df
    except KeyError:
        raise KeyError(f"The sheet {SheetNames.NODES} does not exist")

def import_elements(path: str) -> pd.DataFrame:
    df = import_data(path, SheetNames.ELEMENTS)
    df.columns = df.columns.str.lower()
    df.rename(
        columns={"element": "elem"},
        inplace=True
    )

    try:
        return df
    except KeyError:
        raise KeyError(f"The sheet {SheetNames.ELEMENTS} does not exist")

def import_results(path: str) -> pd.DataFrame:
    df = import_data(path, SheetNames.RESULTS)
    df.columns = df.columns.str.lower()
    cols = df.columns.tolist()
    df.rename(
        columns={
            cols[3]: "mxx",
            cols[4]: "myy",
            cols[5]: "mxy"
        },
        inplace=True)

    try:
        return df
    except KeyError:
        raise KeyError(f"The sheet {SheetNames.RESULTS} does not exist")

def list_of_loads(df: pd.DataFrame) -> List:
    return df["load"].unique().tolist()

def filter_by_loads(
    df: pd.DataFrame,
    loads: str | List[str] | None,
) -> pd.DataFrame:
    """Keep only the given load case(s) from a results dataframe.

    `loads=None` returns `df` unchanged (all load cases). A single load
    name or a list of names restricts it to those - use `list_of_loads`
    first to see what's available.
    """
    if loads is None:
        return df

    if isinstance(loads, str):
        loads = [loads]

    available = set(df["load"].unique())
    missing = set(loads) - available
    if missing:
        raise ValueError(
            f"Unknown load case(s): {sorted(missing)}. "
            f"Available: {sorted(available)}"
        )

    return df[df["load"].isin(loads)].reset_index(drop=True)

def average_forces(df: pd.DataFrame) -> pd.DataFrame:
    """Average duplicate (elem, load, node) result rows.

    A single element can report the same node more than once (e.g. corner
    values from adjacent Gauss points); this collapses those into one row
    per (elem, load, node) before any further processing.
    """
    return (
        df.groupby(["elem", "load", "node"], as_index=False)[["mxx", "myy", "mxy"]]
        .mean()
    )
