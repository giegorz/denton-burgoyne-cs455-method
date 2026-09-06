from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib.pyplot import Figure, Axes

def plot_polygons(
        polygons: pd.DataFrame,
        max_value: float = 1.0
) -> tuple[Figure, Axes]:
    patches, values = _build_polygon_patches(polygons)

    if not patches:
        raise ValueError("No polygons to display.")

    clipped_values = _clip_values(values, max_value)
    fig, ax = _create_figure_and_axes()
    patch_collection = _create_patch_collection(patches, clipped_values)

    _configure_axes(ax, patch_collection)
    _add_colorbar(fig, ax, patch_collection)

    plt.tight_layout()
    return fig, ax


def _build_polygon_patches(polygons: pd.DataFrame) -> tuple[list[Polygon], np.ndarray]:
    patches: list[Polygon] = []
    values: list[float] = []

    for coords, value in _iterate_valid_polygons(polygons):
        patches.append(Polygon(coords, closed=True))
        values.append(_normalize_value(value))

    return patches, np.array(values, dtype=float)


def _iterate_valid_polygons(polygons: pd.DataFrame) -> Iterable[tuple[np.ndarray, object]]:
    for row in polygons.itertuples(index=False):
        coords = np.array(row.coords)

        if len(coords) < 3:
            continue

        yield coords, row.value


def _normalize_value(value: float) -> float:
    return float(value) if pd.notna(value) else np.nan


def _clip_values(values: np.ndarray, max_value: float) -> np.ndarray:
    return np.clip(values, a_min=None, a_max=max_value)


def _create_figure_and_axes(figsize: tuple[int, int] = (8, 8)) -> tuple[plt.Figure, plt.Axes]:
    return plt.subplots(figsize=figsize)


def _create_patch_collection(
        patches: list[Polygon],
        values: np.ndarray,
        cmap: str = "turbo_r",
        edgecolor: str = "black",
        linewidth: float = 0.8,
) -> PatchCollection:
    patch_collection = PatchCollection(
        patches,
        cmap=cmap,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    patch_collection.set_array(values)
    return patch_collection


def _configure_axes(ax: plt.Axes, patch_collection: PatchCollection) -> None:
    ax.add_collection(patch_collection)
    ax.autoscale_view()
    ax.set_aspect("equal")


def _add_colorbar(
        fig: plt.Figure,
        ax: plt.Axes,
        patch_collection: PatchCollection,
        label: str = "Gamma"
) -> None:
    fig.colorbar(patch_collection, ax=ax, label=label)