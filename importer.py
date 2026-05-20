import pandas as pd

def import_nodes(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath, sheet_name="Nodes")
    df.columns = ["id", "x", "y", "z"]
    return df

def import_elements(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(
            filepath,
            sheet_name="Elements",
            usecols="A,G:N" # type: ignore
        )
    except (pd.errors.EmptyDataError, ValueError) as e:
        raise RuntimeError(
            f"Nie można załadować elementów z pliku: {filepath}: {e}"
        ) from e
    return df

def main():
    nodes = import_nodes("dane_z_midasa.xlsx")

if __name__ == "__main__":
    main()