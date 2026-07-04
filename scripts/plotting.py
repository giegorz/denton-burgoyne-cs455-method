from __future__ import annotations

import matplotlib.pyplot as plt

from matplotlib.figure import Figure

from scripts.denton_logic import *

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