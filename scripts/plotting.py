import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from denton_logic import *


def plot_moment_field(angles_field: np.ndarray = None,
                      moment_field: np.ndarray = None,
                      capacity_field: np.ndarray = None,
                      convert_to_degrees: bool = True) -> Figure:
    fig, ax = plt.subplots()

    if angles_field is None:
        angles_field = AnglesField().x

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

def plot_contour(nodes: pd.DataFrame,
                 values: np.ndarray,
                 *,
                 colormap_min: float = None,
                 colormap_max: float = None,
                 title: str = "Contour plot") -> Figure:
    pass


def main():
    x = AnglesField(number_of_divisions=180).x
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
