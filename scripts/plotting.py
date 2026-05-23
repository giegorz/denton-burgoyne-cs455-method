import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.figure import Figure
from matplotlib.tri import Triangulation

from denton_logic import *

def plot_moment_field(angles_field: np.ndarray = None,
                      moment_field: np.ndarray = None,
                      capacity_field: np.ndarray = None,
                      convert_to_degrees: bool = True) -> Figure:
    fig, ax = plt.subplots()

    if angles_field is None:
        angles_field = ThetasField().x

    if convert_to_degrees:
        angles_field = np.rad2deg(angles_field)

    if capacity_field is not None:
        ax.plot(angles_field, capacity_field, label="Capacity", color="red", lw=3)

    if moment_field is not None:
        ax.plot(angles_field, moment_field, label="Moment", color="blue", lw=2)

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
    cmin = min(z) if colormap_min is None else colormap_min
    cmax = max(z) if colormap_max is None else colormap_max
    n_levels = 12                        # ile przedziałów (np. 10 → 10 „kafli”)
    bounds = np.linspace(cmin, cmax, n_levels + 1)  # np. 0,1,2,...,10 (11 krawędzi → 10 przedziałów)
    norm = BoundaryNorm(bounds, ncolors=256, clip=False)  # clip=False → wartości <cmin i >cmax dostaną „extend”

    # --- WYPEŁNIENIE (MUSI dostać norm + levels=bounds) ---
    cntr = ax.tricontourf(
        tri, z,
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
    x = ThetasField(number_of_divisions=180).x
    c = [100, 35]
    a = [0, 70]
    moments = np.array([35, 15, 10])
    moment_field = create_moment_field(triad=moments, angles_field=x)
    cap = Capacity(c,a)
    cap.convert_to_radians()
    capacity_triad = cap.to_triad()
    capacity_field = create_moment_field(capacity_triad, angles_field=x)

    p = plot_moment_field(angles_field=x, capacity_field=capacity_field, moment_field=moment_field)
    p.show()


if __name__ == "__main__":
    main()
