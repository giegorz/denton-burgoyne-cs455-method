from enum import Enum
from unittest import case

from matplotlib import pyplot as plt

from applications.pandas_merging import merge_results_with_nodes
from importers import import_nodes, import_results, import_elements
from scripts.denton_logic import Capacity, denton_burgoyne_orchestrator, group_gammas_by_elements
# from scripts.plotting import plot_contour, create_polygons
from scripts.plot_contour import create_polygons, plot_contour
from scripts.plot_polygons import plot_polygons
import pandas as pd

class PlotType(str, Enum):
    CONTOUR = "contour"
    POLYGON = "polygon"

def load_data(
    filepath: str = "files/dane_z_midasa.xlsx"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load nodes, elements and results from file."""

    nodes = import_nodes(filepath)
    elements = import_elements(filepath)
    results = import_results(filepath)

    return nodes, elements, results

def process_data(
    capacity_list: list[float],
    angles_list: list[float],
    filepath: str = "files/dane_z_midasa.xlsx"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute gamma-related outputs and merge with geometry."""

    nodes, elements, results = load_data(filepath)

    capacity = Capacity(capacity_list, angles_list)
    gammas = denton_burgoyne_orchestrator(results, capacity)

    gammas_by_elements = group_gammas_by_elements(gammas)
    merged_results = merge_results_with_nodes(gammas, nodes)

    return nodes, elements, gammas_by_elements, merged_results


def visualise(
        nodes: pd.DataFrame,
        elements: pd.DataFrame,
        gammas_by_elements: pd.DataFrame,
        merged_results: pd.DataFrame,
        plot_type: PlotType = PlotType.CONTOUR
) -> plt.Figure:

    match plot_type:
        case plot_type.CONTOUR:
            fig = plot_contour(merged_results, colormap_min=0, colormap_max=1)
            return fig
        case plot_type.POLYGON:
            polygons = create_polygons(nodes, elements, gammas_by_elements)
            fig, _ = plot_polygons(polygons)
            return fig
        case _:
            raise KeyError(f"Unknown plot type: {plot_type}")

def main():
    nodes, elements, gammas_by_elements, merged_results = \
    process_data(
        [750, 500],
        [0, 70]
    )
    visualise(
        nodes= nodes,
        elements= elements,
        gammas_by_elements= gammas_by_elements,
        merged_results= merged_results,
        plot_type= PlotType.POLYGON
    )
    plt.show()

if __name__ == "__main__":
    main()