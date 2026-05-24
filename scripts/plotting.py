import matplotlib.pyplot as plt

from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.tri import Triangulation
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

    if not colormap_min:
        colormap_min = np.min(values)

    if colormap_min <= 1:
        colormap_max = 1
    else:
        colormap_max = colormap_min + 1

    values = np.clip(values, colormap_min, colormap_max)


    if np.isinf(values).any():
        raise ValueError("Values must not be infinite")

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

    # Linie konturów (spójne z bounds)
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
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
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

        # węzły, pomijając 0
        node_ids = [
            int(getattr(el, f"node{i}"))
            for i in range(1, 9)
            if getattr(el, f"node{i}") != 0
        ]

        # współrzędne wielokąta
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


def plot_polygons(polygons: pd.DataFrame, max_value: float = 1.0):
    patches = []
    values = []

    for row in polygons.itertuples(index=False):
        coords = np.array(row.coords)
        value = row.value

        if len(coords) < 3:
            continue

        patches.append(Polygon(coords, closed=True))
        values.append(float(value) if pd.notna(value) else np.nan)

    if not patches:
        raise ValueError("Brak polygonów do narysowania.")

    values = np.array(values, dtype=float)
    values = np.clip(values, a_min=None, a_max=max_value)

    fig, ax = plt.subplots(figsize=(8, 8))

    pc = PatchCollection(
        patches,
        cmap="turbo_r",
        edgecolor="black",
        linewidth=0.8
    )
    pc.set_array(np.array(values, dtype=float))

    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")

    fig.colorbar(pc, ax=ax, label="Gamma")
    plt.tight_layout()
    plt.show()

    return fig, ax


def main():

    c = [750, 500]
    a = [0, 70]
    capacity = Capacity(c,a)
    results = import_results("../files/dane_z_midasa.xlsx")
    nodes = import_nodes("../files/dane_z_midasa.xlsx")
    elements = import_elements("../files/dane_z_midasa.xlsx")

    gammas = denton_burgoyne_orchestrator(results, capacity)
    gammas_by_elements = group_gammas_by_elements(gammas)
    merged_results = merge_results_with_nodes(gammas, nodes)

    print(merged_results.head())

    plot_contour(merged_results)
    # plt.show()

    pc = create_polygons(nodes, elements, gammas_by_elements)
    print(pc)

    plot_polygons(pc)




if __name__ == "__main__":
    main()
