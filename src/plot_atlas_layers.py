# -*- coding: utf-8 -*-
"""
Small plotting utilities for atlas-aligned density maps.

The boundary volume is assumed to be a binary image volume:
  1 = atlas boundary pixel
  0 = background
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def _rgba(color: str | tuple[float, float, float], alpha: float) -> tuple[float, float, float, float]:
    r, g, b, _ = mcolors.to_rgba(color)
    return (r, g, b, alpha)


def _flip_image(image: np.ndarray, flip_x: bool, flip_y: bool) -> np.ndarray:
    out = np.array(image, copy=True)
    if flip_y:
        out = np.flipud(out)
    if flip_x:
        out = np.fliplr(out)
    return out


def render_slice(
    output_path: str | Path,
    *,
    boundary2d: np.ndarray,
    kde_layers: list[dict[str, Any]],
    injection_mask: np.ndarray | None = None,
    flip_x: bool = False,
    flip_y: bool = True,
    dpi: int = 300,
    transparent: bool = False,
    draw_boundary: bool = True,
    png_legend: bool = False,
) -> None:
    
    """
    Render one coronal atlas slice with one or more density-map contour layers.

    Parameters
    ----------
    output_path
        Output image path. PNG and SVG are supported.
    boundary2d
        2D binary atlas boundary image.
    kde_layers
        List of dicts with keys: dens2d, levels, color, line_width, fill.
    injection_mask
        Optional 2D injection-site mask already resized to boundary2d shape.
    """
    output_path = Path(output_path)
    boundary_plot = _flip_image(boundary2d, flip_x, flip_y)
    height, width = boundary_plot.shape

    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_axis_off()

    if draw_boundary and np.any(boundary_plot > 0):
        boundary_alpha = (boundary_plot > 0).astype(float)
        ax.imshow(
            np.zeros_like(boundary_alpha),
            origin="lower",
            interpolation="nearest",
            cmap="gray",
            alpha=boundary_alpha,
        )

    if injection_mask is not None and np.any(injection_mask > 0):
        mask_plot = _flip_image(injection_mask, flip_x, flip_y)
        ax.contourf(
            mask_plot.astype(float),
            levels=[0.5, 1.0],
            colors=[_rgba((0.2, 0.2, 0.2), 0.25)],
            origin="lower",
        )

    for layer in kde_layers:
        density = _flip_image(np.asarray(layer["dens2d"], dtype=float), flip_x, flip_y)
        if not np.any(density > 0):
            continue

        levels = np.asarray(layer.get("levels", []), dtype=float)
        levels = np.unique(levels[np.isfinite(levels)])
        if levels.size == 0:
            continue

        color = layer.get("color", "magenta")
        line_width = float(layer.get("line_width", 0.8))

        if bool(layer.get("fill", True)):
            max_value = float(np.max(density))
            if max_value > levels[-1]:
                fill_levels = np.r_[levels, max_value]
                n_intervals = len(fill_levels) - 1
                alphas = np.linspace(0.25, 0.45, n_intervals)
                colors = [_rgba(color, float(alpha)) for alpha in alphas]
                ax.contourf(density, levels=fill_levels, colors=colors, origin="lower")

        ax.contour(
            density,
            levels=levels,
            colors=[_rgba(color, 1.0)],
            linewidths=line_width,
            origin="lower",
        )

        if output_path.suffix.lower() == ".png" and png_legend:
            x0 = 0.98
            y0 = 0.98
            dy = 0.045

            for i, layer in enumerate(kde_layers):
                label = layer.get("label", layer.get("name", f"Layer {i+1}"))
                color = layer.get("color", "black")
                y = y0 - i * dy

                ax.text(
                    x0 - 0.030,
                    y,
                    label,
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=6,
                    color="black",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=1.5),
                )

                ax.add_patch(
                    plt.Rectangle(
                        (x0 - 0.020, y - 0.025),
                        0.018,
                        0.018,
                        transform=ax.transAxes,
                        facecolor=color,
                        edgecolor="black",
                        linewidth=0.5,
                        clip_on=False,
                    )
                )

    fig.savefig(output_path, dpi=dpi, transparent=transparent)
    plt.close(fig)
