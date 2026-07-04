import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.pyplot import Figure, Axes

def plot_contour(
    results: pd.DataFrame,
    *,
    colormap_min: float | None = None,
    colormap_max: float | None = None,
    title: str = "Contour plot",
) -> Figure:

    #constants:

    N_LEVELS = 13
    LINE_WIDTH = 0.5

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

    x = results["x"].to_numpy()
    y = results["y"].to_numpy()
    values = results["gamma"].to_numpy()

    if colormap_min is None:
        colormap_min = np.min(values)

    if colormap_max is None:
        colormap_max = 1 if colormap_min <= 1 else colormap_min + 1


    values = np.clip(values, colormap_min, colormap_max)


    if not np.isfinite(values).all():
        raise ValueError("Values must be finite (no NaN or inf)")


    tri = Triangulation(x, y)

    bounds = np.linspace(colormap_min, colormap_max, N_LEVELS)

    cntr = ax.tricontourf(
        tri, values,
        levels=bounds,
        cmap="turbo_r",
        extend="both",
        antialiased=True,
    )

    # Linie konturów (spójne z bounds)
    ax.tricontour(
        tri, values,
        levels=bounds,
        colors="k",
        linewidths=LINE_WIDTH,
        alpha=LINE_WIDTH,
        antialiased=True,
    )

    fig.colorbar(cntr, ax=ax, label="γ (gamma)", ticks=bounds)

    ax.triplot(tri, color="0.5", linewidth=0.3, alpha=0.5, zorder=0)
    ax.set_aspect("equal")
    ax.set_title(title)
    plt.tight_layout()
    return fig

def create_polygons(
    nodes: pd.DataFrame,
    elements: pd.DataFrame,
    gammas_by_elements: pd.DataFrame
) -> pd.DataFrame:
    # node -> (x, y)
    node_coords = {
        int(row.node): (float(row.x), float(row.y))
        for row in nodes.itertuples(index=False)
    }

    # elem -> gamma_value
    gamma_map = dict(zip(gammas_by_elements["elem"], gammas_by_elements["gamma"]))

    result = []

    for el in elements.itertuples(index=False):
        element_id = int(el.elem)

        node_ids = [
            int(getattr(el, f"node{i}"))
            for i in range(1, 9)
            if getattr(el, f"node{i}") != 0
        ]

        coords = [node_coords[node_id] for node_id in node_ids if node_id in node_coords]

        if len(coords) < 3:
            continue

        gamma_value = gamma_map.get(element_id, np.nan)

        result.append({
            "element_id": element_id,
            "coords": coords,
            "value": gamma_value,
        })

    return pd.DataFrame(result)