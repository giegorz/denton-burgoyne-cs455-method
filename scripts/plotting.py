import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.figure import Figure
from matplotlib.tri import Triangulation

from applications.orchiestrators import denton_orchestrator
from applications.pandas_merging import merge_results_with_nodes
from denton_logic import *
from importers.importer import import_results, import_nodes


def plot_moment_field(denton: Denton) -> Figure:
    fig, ax = plt.subplots()

    angles_field = denton.angles_field
    capacity_field = denton.capacities_field
    moment_field = denton.moment_field

    if capacity_field is not None:
        ax.plot(angles_field, capacity_field, label="Capacity", color="red", lw=3)

    if moment_field is not None:
        ax.plot(angles_field, moment_field, label="Moment", color="blue", lw=2)

    ax.plot(angles_field, moment_field * denton.gamma, label="Moment * gamma", color="green", lw=1, linestyle="--")

    ax.set_xlabel("Theta [degrees]")
    ax.set_ylabel("Moment [degrees]")
    ax.set_title("Moment Field")
    ax.set_xlim(np.min(angles_field), np.max(angles_field))
    ax.grid(linestyle=":")

    return fig

def plot_contour(
    results: pd.DataFrame,
    *,
    colormap_min: float | None = None,
    colormap_max: float | None = None,
    title: str = "Contour plot",
) -> Figure:

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    x = results["x"].to_numpy()
    y = results["y"].to_numpy()
    values = results["gamma"].to_numpy()

    if np.isinf(values).any():
        values = np.clip(values, 0, 1000)

    min_value = np.min(values)
    max_value = np.max(values)
    #TODO
    colormap_min = colormap_min if colormap_min is not None else min_value
    colormap_max = colormap_max if colormap_max is not None else max_value

    if colormap_min <= 1:
        colormap_max = 1.0
    else:
        colormap_max = colormap_min + 1

    # if np.isinf(values).any():
    #     raise ValueError("Values must not be infinite")

    tri = Triangulation(x, y)

    cmin = values.min() if colormap_min is None else colormap_min
    cmax = values.max() if colormap_max is None else colormap_max

    bounds = np.linspace(cmin, cmax, 13)

    cntr = ax.tricontourf(
        tri, values,
        levels=bounds,
        cmap="turbo_r",
        extend="both",
        antialiased=True,
    )

    ax.tricontour(
        tri, values,
        levels=bounds,
        colors="k",
        linewidths=0.5,
        alpha=0.5,
        antialiased=True,
    )

    fig.colorbar(cntr, ax=ax, label="γ (gamma)", ticks=bounds)

    ax.triplot(tri, color="0.5", linewidth=0.3, alpha=0.5, zorder=0)
    ax.set_aspect("equal")
    ax.set_title(title)
    plt.tight_layout()
    return fig

def main():

    c = [800, 500]
    a = [0, 70]
    capacity = Capacity(c,a)
    results = import_results("../files/dane_z_midasa.xlsx")
    nodes = import_nodes("../files/dane_z_midasa.xlsx")

    gammas = calculate_gammas(results, capacity)
    merged_results = merge_results_with_nodes(gammas, nodes)

    plot_contour(merged_results,colormap_min=0)
    plt.show()

    # moments = np.array([35, 15, 10])
    # dent = denton_orchestrator(c,a,moments)
    #
    # p = plot_moment_field(dent)
    # p.show()


if __name__ == "__main__":
    main()
