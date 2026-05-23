import pandas as pd
import pytest
import importers.importer as importer

@pytest.fixture
def nodes_df():
    return pd.DataFrame(
        {
            "Node": [1, 2],
            "X(m)": [0.0, 1.5],
            "Y(m)": [2.0, 3.5],
        }
    )


@pytest.fixture
def elements_df():
    return pd.DataFrame(
        {
            "Element": [10, 20],
            "Load": ["L1", "L1"],
            "Node": [1, 2],
            "A": [100, 200],
            "B": [300, 400],
            "C": [500, 600],
            "D": [700, 800],
            "E": [900, 1000],
            "F": [1100, 1200],
            "G": [1300, 1400],
            "H": [1500, 1600],
            "I": [1700, 1800],
            "J": [1900, 2000],
            "K": [2100, 2200],
            "L": [2300, 2400],
            "M": [2500, 2600],
            "N": [2700, 2800],
        }
    )


@pytest.fixture
def results_df():
    return pd.DataFrame(
        {
            "Elem": [1, 1, 2, 2],
            "Load": ["L1", "L1", "L2", "L2"],
            "Node": [101, 101, 102, 102],
            "mxx": [10.0, 14.0, 20.0, 22.0],
            "myy": [1.0, 3.0, 4.0, 6.0],
            "mxy": [7.0, 9.0, 11.0, 13.0],
        }
    )


def test_import_data_calls_read_excel(monkeypatch):
    captured = {}

    def fake_read_excel(path, sheet_name=None, usecols=None):
        captured["path"] = path
        captured["sheet_name"] = sheet_name
        captured["usecols"] = usecols
        return pd.DataFrame({"a": [1]})

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    df = importer.import_data("file.xlsx", importer.SheetNames.NODES.value)

    assert not df.empty
    assert captured["path"] == "file.xlsx"
    assert captured["sheet_name"] == "nodes"
    assert captured["usecols"] == "A:C"


def test_import_nodes_renames_columns(monkeypatch, nodes_df):
    monkeypatch.setattr(importer, "import_data", lambda path, sheet_name: nodes_df.copy())

    df = importer.import_nodes("file.xlsx")

    assert list(df.columns) == ["node", "x", "y"]
    assert df.loc[0, "node"] == 1
    assert df.loc[1, "x"] == 1.5


def test_import_elements_renames_columns(monkeypatch, elements_df):
    monkeypatch.setattr(importer, "import_data", lambda path, sheet_name: elements_df.copy())

    df = importer.import_elements("file.xlsx")

    assert "elem" in df.columns
    assert "element" not in df.columns
    assert all(col == col.lower() for col in df.columns)


def test_import_results_renames_force_columns(monkeypatch, results_df):
    monkeypatch.setattr(importer, "import_data", lambda path, sheet_name: results_df.copy())

    df = importer.import_results("file.xlsx")

    assert list(df.columns) == ["elem", "load", "node", "mxx", "myy", "mxy"]


def test_list_of_loads():
    df = pd.DataFrame({"load": ["L1", "L2", "L1"]})

    loads = importer.list_of_loads(df)

    assert loads == ["L1", "L2"]


def test_average_forces():
    df = pd.DataFrame(
        {
            "elem": [1, 1, 2, 2],
            "load": ["L1", "L1", "L2", "L2"],
            "node": [101, 101, 102, 102],
            "mxx": [10.0, 14.0, 20.0, 22.0],
            "myy": [1.0, 3.0, 4.0, 6.0],
            "mxy": [7.0, 9.0, 11.0, 13.0],
        }
    )

    out = importer.average_forces(df)

    assert list(out.columns) == ["elem", "load", "node", "mxx", "myy", "mxy"]
    assert out.loc[0, "mxx"] == 12.0
    assert out.loc[0, "myy"] == 2.0
    assert out.loc[0, "mxy"] == 8.0