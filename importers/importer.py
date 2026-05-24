from typing import Any

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
                       "Y(m)": "y"}, inplace=True)

    try:
        return df
    except KeyError:
        raise KeyError(f"The sheet {SheetNames.NODES} does not exist")

def import_elements(path: str) -> pd.DataFrame:
    df = import_data(path, SheetNames.ELEMENTS)
    df.columns = df.columns.str.lower()
    df.rename(columns={"element": "elem"}, inplace=True)

    try:
        return df
    except KeyError:
        raise KeyError(f"The sheet {SheetNames.ELEMENTS} does not exist")

def import_results(path: str) -> pd.DataFrame:
    df = import_data(path, SheetNames.RESULTS)
    df.columns = df.columns.str.lower()
    cols = df.columns.tolist()

    df.rename(columns={
        cols[3]: "mxx",
        cols[4]: "myy",
        cols[5]: "mxy"
    }, inplace=True)

    try:
        return df
    except KeyError:
        raise KeyError(f"The sheet {SheetNames.RESULTS} does not exist")


def main():
    n = import_nodes("../files/dane_z_midasa.xlsx")
    e = import_elements("../files/dane_z_midasa.xlsx")
    r = import_results("../files/dane_z_midasa.xlsx")


if __name__ == "__main__":
    main()


