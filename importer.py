from typing import Any

import pandas as pd
from enum import Enum

class SheetTabs(str, Enum):
    NODES = "nodes"
    ELEMENTS = "elements"
    RESULTS = "results"

class ColumnsNames(str, Enum):
    NODES = "A:C"
    ELEMENTS = "A,G:N"
    RESULTS = "A,B,C,J:L"

def import_data(path: str, sheet_name: str) -> pd.DataFrame:

    _map = {
        SheetTabs.NODES: ColumnsNames.NODES.value,
        SheetTabs.ELEMENTS: ColumnsNames.ELEMENTS.value,
        SheetTabs.RESULTS: ColumnsNames.RESULTS.value,
    }

    try:
        return pd.read_excel(path, sheet_name=sheet_name, usecols=_map[sheet_name])
    except KeyError:
        raise KeyError(f"The sheet {sheet_name} does not exist")

def import_nodes(path: str) -> pd.DataFrame | None | Any:
    df = import_data(path, SheetTabs.NODES)

    df.rename(columns={"Node": "node",
                       "X(m)": "x",
                       "Y(m)": "y"}, inplace=True)

    df.set_index("node", inplace=True)

    try:
        return df
    except KeyError:
        raise KeyError(f"The sheet {SheetTabs.NODES} does not exist")

def import_elements(path: str) -> pd.DataFrame:
    df = import_data(path, SheetTabs.ELEMENTS)

    try:
        return df
    except KeyError:
        raise KeyError(f"The sheet {SheetTabs.ELEMENTS} does not exist")

def import_results(path: str) -> pd.DataFrame:
    df = import_data(path, SheetTabs.RESULTS)
    try:
        return df
    except KeyError:
        raise KeyError(f"The sheet {SheetTabs.RESULTS} does not exist")

def list_of_loadcases(df: pd.DataFrame) -> list[Any]:
    return df["Load"].unique().tolist()

def main():
    d = import_results("dane_z_midasa.xlsx")

    print(list_of_loadcases(d))



if __name__ == "__main__":
    main()


