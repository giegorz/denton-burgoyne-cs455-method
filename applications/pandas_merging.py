import pandas as pd
from typing import Any
import timeit

from scripts.denton_logic import denton_burgoyne_orchestrator
from importers.importer import *
from scripts.denton_logic import Capacity


def list_of_loads(df: pd.DataFrame) -> list[Any]:
    return df["load"].unique().tolist()

def results_mean_by_node(df: pd.DataFrame) -> pd.DataFrame:
    mean_forces = (df
                   .groupby(['load', 'node'])
                   [['mxx', 'myy', 'mxy']]
                   .mean()
                   .reset_index())
    return mean_forces



def merge_results_with_nodes(
        gammas_df: pd.DataFrame,
        nodes: pd.DataFrame
) -> pd.DataFrame:
    return gammas_df.merge(nodes, how='left', on='node')

def main():
    r = import_results("../files/dane_z_midasa.xlsx")
    rm = results_mean_by_node(r)
    n = import_nodes("../files/dane_z_midasa.xlsx")
    e = import_elements("../files/dane_z_midasa.xlsx")

    capacity = Capacity(capacity= [750, 500], angles= [0, 90])
    merged = merge_results_with_nodes(rm, n)

    bbb = denton_burgoyne_orchestrator(rm, capacity=capacity)
    ccc= merge_results_with_nodes(bbb, n)

    # t2 = timeit.timeit(lambda: calculate_denton_vectorized(rm,capacity=capacity), number=10)
    # print(t2)

    # print(ccc)
    # print(e)

    nn = create_polygons(n, e)
    print(nn[623])



if __name__ == "__main__":
    main()