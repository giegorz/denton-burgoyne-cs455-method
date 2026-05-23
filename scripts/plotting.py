import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.figure import Figure
from matplotlib.tri import Triangulation

from applications.orchiestrators import denton_orchestrator
from denton_logic import *

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

def plot_contour(merged_df: pd.DataFrame,
                 *,
                 colormap_min: float = None,
                 colormap_max: float = None,
                 title: str = "Contour plot") -> Figure:

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    x = merged_df["x"].to_numpy()
    y = merged_df["y"].to_numpy()
    values = merged_df["gamma"].to_numpy()
    tri = Triangulation(x, y)

    # --- USTAWIENIA KOLORÓW / NORMY ---
    cmin = min(values) if colormap_min is None else colormap_min
    cmax = max(values) if colormap_max is None else colormap_max
    n_levels = 12                        # ile przedziałów (np. 10 → 10 „kafli”)
    bounds = np.linspace(cmin, cmax, n_levels + 1)  # np. 0,1,2,...,10 (11 krawędzi → 10 przedziałów)
    norm = BoundaryNorm(bounds, ncolors=256, clip=False)  # clip=False → wartości <cmin i >cmax dostaną „extend”

    # --- WYPEŁNIENIE (MUSI dostać norm + levels=bounds) ---
    cntr = ax.tricontourf(
        tri, values,
        levels=bounds,
        norm=norm,
        cmap="turbo_r",  # 'turbo', 'viridis', 'JET_R'
        antialiased=True
    )

    # Linie konturów (spójne z bounds)
    ax.tricontour(
        tri, values,
        levels=bounds,
        colors="k",
        linewidths=0.5,
        alpha=0.5,
        norm=norm,
        antialiased=True
    )

    cb = fig.colorbar(
        cntr, ax=ax, label="γ (gamma)",
        boundaries=bounds,
        ticks=bounds,
        extend="both"
    )

    ax.triplot(
        tri,
        color="0.5",
        linewidth=0.3,
        alpha=0.5,
        zorder=0
    )

    ax.set_aspect("equal")
    ax.set_title(title)
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    plt.tight_layout()
    return fig

def main():

    c = [100, 35]
    a = [0, 70]
    moments = np.array([35, 15, 10])
    dent = denton_orchestrator(c,a,moments)

    p = plot_moment_field(dent)
    p.show()


if __name__ == "__main__":
    main()
