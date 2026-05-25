from applications.pandas_merging import merge_results_with_nodes
from importers import import_nodes, import_results, import_elements
from scripts.denton_logic import Capacity, denton_burgoyne_orchestrator, group_gammas_by_elements
from scripts.plotting import plot_contour, create_polygons, plot_polygons
import pandas as pd

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
) -> None:
    """Generate all plots."""

    # Contour plot
    fig1 = plot_contour(merged_results)
    fig1.show()

    # Polygon plot
    polygons = create_polygons(nodes, elements, gammas_by_elements)
    fig2, _ = plot_polygons(polygons)
    fig2.show()

def main():
    nodes, elements, gammas_by_elements, merged_results = \
    process_data(
        [750, 500],
        [0, 70]
    )
    visualise(nodes, elements, gammas_by_elements, merged_results)

if __name__ == "__main__":
    main()