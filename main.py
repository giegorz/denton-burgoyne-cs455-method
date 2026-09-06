from enum import Enum
from unittest import case

from matplotlib import pyplot as plt

from applications.pandas_merging import merge_results_with_nodes, results_mean_by_node
from importers import (
    import_nodes,
    import_results,
    import_elements,
    list_of_loads,
    filter_by_loads,
    write_sheet_to_excel,
)
from scripts.denton_logic import (
    Capacity,
    denton_burgoyne_orchestrator,
    denton_burgoyne_by_node,
    denton_burgoyne_report_by_node,
    group_gammas_by_elements,
)
from scripts import plot_contour, create_polygons, plot_polygons
import pandas as pd

class PlotType(str, Enum):
    CONTOUR = "contour"
    POLYGON = "polygon"

def load_data(
    filepath: str = "files/dane_z_midasa.xlsx",
    loads: str | list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load nodes, elements and results from file.

    `loads` restricts the results to one or more load cases (by name, as
    they appear in the "load" column) - pass None to keep all of them.
    Use `available_loads(filepath)` to see what's in the file first.
    """

    nodes = import_nodes(filepath)
    elements = import_elements(filepath)
    results = import_results(filepath)
    results = filter_by_loads(results, loads)

    return nodes, elements, results

def available_loads(filepath: str = "files/dane_z_midasa.xlsx") -> list:
    """List the load case names present in the results sheet."""
    return list_of_loads(import_results(filepath))

def process_data(
    capacity_list: list[float],
    angles_list: list[float],
    filepath: str = "files/dane_z_midasa.xlsx",
    loads: str | list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute gamma-related outputs and merge with geometry.

    `loads` restricts the computation to one or more load cases, see
    `load_data`.

    Two separate gamma computations are produced on purpose:
      - `gammas_by_elements`: gamma per (elem, load) from the raw per-element
        results, used for the polygon plot.
      - `merged_results`: gamma per (node, load), computed *after* averaging
        the moments of every element sharing that node (so a shared node
        gets one gamma value, not one per contributing element) - used for
        the contour plot.
    """

    nodes, elements, results = load_data(filepath, loads=loads)
    capacity = Capacity(capacity_list, angles_list)

    gammas = denton_burgoyne_orchestrator(results, capacity)
    gammas_by_elements = group_gammas_by_elements(gammas)

    node_moments = results_mean_by_node(results)
    node_gammas = denton_burgoyne_by_node(node_moments, capacity)
    merged_results = merge_results_with_nodes(node_gammas, nodes)

    return nodes, elements, gammas_by_elements, merged_results


def export_gamma_report(
    capacity_list: list[float],
    angles_list: list[float],
    filepath: str = "files/dane_z_midasa.xlsx",
    loads: str | list[str] | None = None,
    output_filepath: str | None = None,
    sheet_name: str = "denton_report",
) -> pd.DataFrame:
    """
    Build the full per-point report (triad, gamma, theta, and MN(theta) for
    every theta - plus a reference CAPACITY row with MR(theta)) and write
    it as a new sheet, so it can be opened and charted directly in Excel.

    Writes into `filepath` itself (adding/replacing `sheet_name`) unless
    `output_filepath` is given - pass a different path if you don't want
    to modify the source workbook with the raw MIDAS export.

    This table can get very wide (one column per theta) and is meant for
    manual spot-checking, not for anything else in the pipeline.
    """
    nodes, elements, results = load_data(filepath, loads=loads)
    capacity = Capacity(capacity_list, angles_list)

    node_moments = results_mean_by_node(results)
    report = denton_burgoyne_report_by_node(node_moments, capacity)
    report = merge_results_with_nodes(report, nodes)

    fixed_columns = ["node", "load", "x", "y", "mxx", "myy", "mxy", "gamma", "theta_deg"]
    theta_columns = [c for c in report.columns if c not in fixed_columns]
    report = report[fixed_columns + theta_columns]

    write_sheet_to_excel(output_filepath or filepath, report, sheet_name)

    return report


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