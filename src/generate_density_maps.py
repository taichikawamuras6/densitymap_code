#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate atlas-aligned density maps from QuPath-exported coordinate tables.

This script was prepared for public code sharing. It expects:
  1. QuPath TSV/CSV files containing atlas coordinates.
  2. A binary atlas boundary volume saved as a NumPy .npy file.
  3. A YAML config file describing input paths, layer filters, and rendering settings.

Example
-------
python src/generate_density_maps.py --config configs/vsub_phpeb_densitymap.yml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter

from plot_atlas_layers import render_slice


# Kim laboratory / CCFv3 atlas grid resolution used in this study.
# Volume axis order is Z, Y, X.
ATLAS_RESOLUTION_UM = np.array([10.0, 10.0, 10.0], dtype=float)


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def compute_boundary_volume_from_annotation(annotation: np.ndarray) -> np.ndarray:
    """Create a binary atlas-boundary volume from an annotation volume."""
    from skimage.segmentation import find_boundaries

    boundary = np.zeros(annotation.shape, dtype=np.uint8)
    for z in range(annotation.shape[0]):
        boundary[z] = find_boundaries(annotation[z], mode="inner").astype(np.uint8)
    return boundary


def load_or_build_boundary_volume(boundary_path: Path, atlas_name: str = "kim_mouse_10um"):
    """
    Load atlas boundary volume if available.
    If not available, try to build it from the BrainGlobe atlas annotation.
    If this also fails, return None.
    """
    if boundary_path.exists():
        print(f"[INFO] Loaded atlas boundary volume: {boundary_path}")
        return np.load(boundary_path)

    print(f"[WARN] Atlas boundary volume not found: {boundary_path}")
    print(f"[INFO] Trying to build boundary volume from BrainGlobe atlas: {atlas_name}")

    try:
        from bg_atlasapi import BrainGlobeAtlas

        atlas = BrainGlobeAtlas(atlas_name)
        annotation = atlas.annotation
        boundary = compute_boundary_volume_from_annotation(annotation)

        boundary_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(boundary_path, boundary)

        print(f"[INFO] Built and saved atlas boundary volume: {boundary_path}")
        return boundary

    except Exception as e:
        print("[WARN] Could not build atlas boundary volume automatically.")
        print(f"[WARN] Reason: {e}")
        print("[WARN] Density maps will be generated without atlas boundaries.")
        return None

def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", low_memory=False)
    raise ValueError(f"Unsupported input file type: {path}")


def select_rows(df: pd.DataFrame, layer: dict[str, Any], columns: dict[str, str]) -> pd.DataFrame:
    """Select QuPath objects for one layer using object type and optional filters."""
    out = df.copy()

    object_type_col = columns["object_type"]
    if "object_type" in layer:
        out = out[out[object_type_col].astype(str) == str(layer["object_type"])]

    filt = layer.get("filter", {}) or {}
    class_col = columns["classification"]
    name_col = columns["name"]

    if "classification_equals" in filt:
        out = out[out[class_col].astype(str) == str(filt["classification_equals"])]

    if "classification_contains" in filt:
        pattern = str(filt["classification_contains"])
        out = out[out[class_col].astype(str).str.contains(pattern, case=False, na=False)]

    if "name_equals" in filt:
        out = out[out[name_col].astype(str) == str(filt["name_equals"])]

    if "name_contains" in filt:
        pattern = str(filt["name_contains"])
        out = out[out[name_col].astype(str).str.contains(pattern, case=False, na=False)]

    return out


def atlas_coordinates_um(
    df: pd.DataFrame,
    columns: dict[str, str],
    unit_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Atlas_X/Y/Z coordinates in micrometers and a finite-value mask."""
    x = pd.to_numeric(df[columns["x"]], errors="coerce").to_numpy(float) * unit_factor
    y = pd.to_numeric(df[columns["y"]], errors="coerce").to_numpy(float) * unit_factor
    z = pd.to_numeric(df[columns["z"]], errors="coerce").to_numpy(float) * unit_factor
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    return x[finite], y[finite], z[finite], finite


def coarse_shape(boundary_shape: tuple[int, int, int], voxel_xy_um: float, voxel_z_um: float) -> tuple[int, int, int]:
    z_fine, y_fine, x_fine = boundary_shape
    z_um = z_fine * ATLAS_RESOLUTION_UM[0]
    y_um = y_fine * ATLAS_RESOLUTION_UM[1]
    x_um = x_fine * ATLAS_RESOLUTION_UM[2]
    return (
        int(np.ceil(z_um / voxel_z_um)),
        int(np.ceil(y_um / voxel_xy_um)),
        int(np.ceil(x_um / voxel_xy_um)),
    )


def coordinates_to_coarse_indices(
    x_um: np.ndarray,
    y_um: np.ndarray,
    z_um: np.ndarray,
    shape: tuple[int, int, int],
    voxel_xy_um: float,
    voxel_z_um: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zc, yc, xc = shape
    xi = np.floor(x_um / voxel_xy_um).astype(int)
    yi = np.floor(y_um / voxel_xy_um).astype(int)
    zi = np.floor(z_um / voxel_z_um).astype(int)
    return (
        np.clip(zi, 0, zc - 1),
        np.clip(yi, 0, yc - 1),
        np.clip(xi, 0, xc - 1),
    )


def find_intensity_column(df: pd.DataFrame, layer: dict[str, Any]) -> str | None:
    """Find or read the intensity column used for weighted density maps."""
    weight = layer.get("weight", {}) or {}
    if "column" in weight:
        return str(weight["column"])

    candidates = [
        c for c in df.columns
        if any(key in str(c).lower() for key in ["intensity", "cy3", "texas"])
    ]
    return candidates[0] if candidates else None


def build_histogram_volume(
    df: pd.DataFrame,
    layer: dict[str, Any],
    columns: dict[str, str],
    cfg: dict[str, Any],
    boundary_shape: tuple[int, int, int],
) -> tuple[np.ndarray, str | None]:
    """Aggregate selected objects into a coarse 3D histogram."""
    dens_cfg = cfg["densmap"]
    voxel_xy = float(dens_cfg["coarse_voxel_um"]["xy"])
    voxel_z = float(dens_cfg["coarse_voxel_um"]["z"])
    unit_factor = float(cfg["columns"].get("unit_factor", 1000.0))

    shape = coarse_shape(boundary_shape, voxel_xy, voxel_z)
    vol = np.zeros(shape, dtype=np.float32)

    selected = select_rows(df, layer, columns)
    if selected.empty:
        return vol, None

    x_um, y_um, z_um, finite = atlas_coordinates_um(selected, columns, unit_factor)
    zi, yi, xi = coordinates_to_coarse_indices(x_um, y_um, z_um, shape, voxel_xy, voxel_z)

    weight_mode = str((layer.get("weight", {}) or {}).get("mode", "count")).lower()
    intensity_col = None

    if weight_mode == "count":
        np.add.at(vol, (zi, yi, xi), 1.0)
    elif weight_mode == "auto_intensity":
        intensity_col = find_intensity_column(selected, layer)
        if intensity_col is None:
            raise ValueError(f"No intensity column found for layer: {layer['name']}")
        weights = pd.to_numeric(selected[intensity_col], errors="coerce").to_numpy(np.float32)[finite]
        weights[~np.isfinite(weights)] = 0.0
        np.add.at(vol, (zi, yi, xi), weights)
    else:
        raise ValueError(f"Unsupported weight mode: {weight_mode}")

    return vol, intensity_col


def build_direct_kde_volume(
    df: pd.DataFrame,
    layer: dict[str, Any],
    columns: dict[str, str],
    cfg: dict[str, Any],
    boundary_shape: tuple[int, int, int],
) -> tuple[np.ndarray, str | None]:
    """
    Build a coarse-grid KDE volume from continuous atlas coordinates.

    This method adds an anisotropic Gaussian contribution around each selected object.
    It was used to avoid hard voxel-bin assignment before smoothing.
    """
    dens_cfg = cfg["densmap"]
    voxel_xy = float(dens_cfg["coarse_voxel_um"]["xy"])
    voxel_z = float(dens_cfg["coarse_voxel_um"]["z"])
    bandwidth_xy = float(dens_cfg["bandwidth_um"]["xy"])
    bandwidth_z = float(dens_cfg["bandwidth_um"]["z"])
    cutoff_sigma = float(dens_cfg.get("kde_cutoff_sigma", 3.0))
    unit_factor = float(cfg["columns"].get("unit_factor", 1000.0))

    shape = coarse_shape(boundary_shape, voxel_xy, voxel_z)
    zc, yc, xc = shape
    vol = np.zeros(shape, dtype=np.float32)

    selected = select_rows(df, layer, columns)
    if selected.empty:
        return vol, None

    x_um, y_um, z_um, finite = atlas_coordinates_um(selected, columns, unit_factor)

    weight_mode = str((layer.get("weight", {}) or {}).get("mode", "count")).lower()
    intensity_col = None
    if weight_mode == "count":
        weights = np.ones_like(x_um, dtype=np.float32)
    elif weight_mode == "auto_intensity":
        intensity_col = find_intensity_column(selected, layer)
        if intensity_col is None:
            raise ValueError(f"No intensity column found for layer: {layer['name']}")
        weights = pd.to_numeric(selected[intensity_col], errors="coerce").to_numpy(np.float32)[finite]
        weights[~np.isfinite(weights)] = 0.0
    else:
        raise ValueError(f"Unsupported weight mode: {weight_mode}")

    rx = max(1, int(np.ceil(cutoff_sigma * bandwidth_xy / voxel_xy)))
    ry = max(1, int(np.ceil(cutoff_sigma * bandwidth_xy / voxel_xy)))
    rz = max(1, int(np.ceil(cutoff_sigma * bandwidth_z / voxel_z)))

    for x0, y0, z0, w0 in zip(x_um, y_um, z_um, weights):
        if not np.isfinite(w0) or w0 == 0:
            continue

        xi0 = int(np.rint(x0 / voxel_xy - 0.5))
        yi0 = int(np.rint(y0 / voxel_xy - 0.5))
        zi0 = int(np.rint(z0 / voxel_z - 0.5))

        x1, x2 = max(0, xi0 - rx), min(xc - 1, xi0 + rx)
        y1, y2 = max(0, yi0 - ry), min(yc - 1, yi0 + ry)
        z1, z2 = max(0, zi0 - rz), min(zc - 1, zi0 + rz)

        xs = (np.arange(x1, x2 + 1, dtype=np.float32) + 0.5) * voxel_xy
        ys = (np.arange(y1, y2 + 1, dtype=np.float32) + 0.5) * voxel_xy
        zs = (np.arange(z1, z2 + 1, dtype=np.float32) + 0.5) * voxel_z

        gx = np.exp(-0.5 * ((xs - x0) / bandwidth_xy) ** 2)
        gy = np.exp(-0.5 * ((ys - y0) / bandwidth_xy) ** 2)
        gz = np.exp(-0.5 * ((zs - z0) / bandwidth_z) ** 2)

        vol[z1:z2 + 1, y1:y2 + 1, x1:x2 + 1] += (
            float(w0) * gz[:, None, None] * gy[None, :, None] * gx[None, None, :]
        ).astype(np.float32)

    return vol, intensity_col


def gaussian_smooth_volume(vol: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    dens_cfg = cfg["densmap"]
    sigma = (
        float(dens_cfg["bandwidth_um"]["z"]) / float(dens_cfg["coarse_voxel_um"]["z"]),
        float(dens_cfg["bandwidth_um"]["xy"]) / float(dens_cfg["coarse_voxel_um"]["xy"]),
        float(dens_cfg["bandwidth_um"]["xy"]) / float(dens_cfg["coarse_voxel_um"]["xy"]),
    )
    return gaussian_filter(vol.astype(np.float32), sigma=sigma, mode="nearest")


def normalize_volume(vol: np.ndarray, method: str) -> np.ndarray:
    vol = vol.astype(np.float32)
    if method == "max":
        value = float(np.max(vol))
    else:
        value = float(np.sum(vol))
    if value <= 0:
        return np.zeros_like(vol, dtype=np.float32)
    return vol / value


def contour_levels(vol: np.ndarray, quantiles: list[float]) -> np.ndarray:
    values = vol[vol > 0]
    if values.size == 0:
        return np.array([], dtype=float)
    return np.unique(np.quantile(values, quantiles))


def build_injection_mask(
    df: pd.DataFrame,
    layer: dict[str, Any],
    columns: dict[str, str],
    cfg: dict[str, Any],
    boundary_shape: tuple[int, int, int],
) -> np.ndarray:
    """Build a coarse 0/1 injection-site mask from objects selected by name/classification."""
    dens_cfg = cfg["densmap"]
    voxel_xy = float(dens_cfg["coarse_voxel_um"]["xy"])
    voxel_z = float(dens_cfg["coarse_voxel_um"]["z"])
    unit_factor = float(cfg["columns"].get("unit_factor", 1000.0))

    shape = coarse_shape(boundary_shape, voxel_xy, voxel_z)
    vol = np.zeros(shape, dtype=np.uint8)

    detect = layer.get("detect_by", {}) or {}
    selected = df.copy()
    mask = np.zeros(len(selected), dtype=bool)

    if "name_equals" in detect:
        mask |= selected[columns["name"]].astype(str).eq(str(detect["name_equals"])).to_numpy()
    if "classification_contains" in detect:
        pattern = str(detect["classification_contains"])
        mask |= selected[columns["classification"]].astype(str).str.contains(pattern, case=False, na=False).to_numpy()

    selected = selected[mask]
    if selected.empty:
        return vol

    x_um, y_um, z_um, _ = atlas_coordinates_um(selected, columns, unit_factor)
    zi, yi, xi = coordinates_to_coarse_indices(x_um, y_um, z_um, shape, voxel_xy, voxel_z)
    vol[zi, yi, xi] = 1

    morph = layer.get("morphology", {}) or {}
    if morph.get("close", True):
        structure = np.zeros((3, 3, 3), dtype=bool)
        structure[1, 1, :] = True
        structure[1, :, 1] = True
        structure[:, 1, 1] = True
        vol = binary_closing(vol.astype(bool), structure=structure)
    if morph.get("fill_holes", True):
        vol = binary_fill_holes(vol)

    return vol.astype(np.uint8)


def coarse_z_from_fine_z(z_fine: int, coarse_z: int, cfg: dict[str, Any]) -> int:
    voxel_z = float(cfg["densmap"]["coarse_voxel_um"]["z"])
    z_um = z_fine * ATLAS_RESOLUTION_UM[0]
    return int(np.clip(np.rint(z_um / voxel_z), 0, coarse_z - 1))


def resize_to_boundary(slice2d: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Resize a coarse 2D slice to atlas-boundary image size."""
    from scipy.ndimage import zoom

    if slice2d.shape == output_shape:
        return slice2d

    zoom_y = output_shape[0] / slice2d.shape[0]
    zoom_x = output_shape[1] / slice2d.shape[1]
    return zoom(slice2d, (zoom_y, zoom_x), order=1, mode="nearest")


def save_volume_npz(
    path: Path,
    raw_volumes: dict[str, np.ndarray],
    smoothed_volumes: dict[str, np.ndarray],
    normalized_volumes: dict[str, np.ndarray],
    metadata: dict[str, Any],
    injection_mask: np.ndarray | None = None,
) -> None:
    arrays: dict[str, Any] = {}
    for name, vol in raw_volumes.items():
        arrays[f"RAW_{name}"] = vol
    for name, vol in smoothed_volumes.items():
        arrays[f"SMOOTH_{name}"] = vol
    for name, vol in normalized_volumes.items():
        arrays[f"NORM_{name}"] = vol
    if injection_mask is not None:
        arrays["INJECTION_MASK"] = injection_mask
    arrays["metadata_json"] = np.array([json.dumps(metadata, ensure_ascii=False)], dtype=object)
    np.savez_compressed(path, **arrays)


def sparse_voxel_table(volumes: dict[str, np.ndarray]) -> pd.DataFrame:
    """Convert non-zero coarse voxels into a compact table for inspection."""
    rows = []
    for layer, vol in volumes.items():
        z, y, x = np.nonzero(vol)
        if z.size == 0:
            continue
        rows.append(pd.DataFrame({
            "layer": layer,
            "zc": z,
            "yc": y,
            "xc": x,
            "value": vol[z, y, x],
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["layer", "zc", "yc", "xc", "value"])


def process_one_sample(
    input_path: Path,
    boundary_volume: np.ndarray,
    cfg: dict[str, Any],
    output_dir: Path,
) -> dict[str, np.ndarray]:
    """Build, save, and render density maps for one QuPath table."""
    sample_name = input_path.stem
    sample_dir = output_dir / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Processing {sample_name}")
    df = read_table(input_path)

    columns = {
        "x": cfg["columns"].get("x", "Atlas_X"),
        "y": cfg["columns"].get("y", "Atlas_Y"),
        "z": cfg["columns"].get("z", "Atlas_Z"),
        "object_type": cfg["columns"].get("object_type", "Object type"),
        "classification": cfg["columns"].get("classification", "Classification"),
        "name": cfg["columns"].get("name", "Name"),
    }

    layers = cfg["layer_defs"]
    active_layers = cfg["layers"]

    dens_layers = [dict(layers[name], name=name) for name in active_layers if layers[name]["kind"] == "densmap"]
    mask_layers = [dict(layers[name], name=name) for name in active_layers if layers[name]["kind"] in {"threshold", "mask", "injection"}]

    raw_volumes: dict[str, np.ndarray] = {}
    smooth_volumes: dict[str, np.ndarray] = {}
    norm_volumes: dict[str, np.ndarray] = {}

    density_method = cfg["densmap"].get("method", "histogram")
    norm_method = cfg["densmap"].get("pooled", {}).get("pool_norm_method", "sum")

    for layer in dens_layers:
        if density_method == "kde":
            raw, intensity_col = build_direct_kde_volume(df, layer, columns, cfg, boundary_volume.shape)
            smooth = raw
        else:
            raw, intensity_col = build_histogram_volume(df, layer, columns, cfg, boundary_volume.shape)
            smooth = gaussian_smooth_volume(raw, cfg)

        norm = normalize_volume(smooth, norm_method)

        raw_volumes[layer["name"]] = raw
        smooth_volumes[layer["name"]] = smooth
        norm_volumes[layer["name"]] = norm

        if intensity_col:
            print(f"[INFO] {sample_name}: layer {layer['name']} weighted by {intensity_col}")

    injection_mask = None
    if mask_layers:
        injection_mask = build_injection_mask(df, mask_layers[0], columns, cfg, boundary_volume.shape)

    metadata = {
        "sample": sample_name,
        "density_method": density_method,
        "coarse_voxel_um": cfg["densmap"]["coarse_voxel_um"],
        "bandwidth_um": cfg["densmap"]["bandwidth_um"],
        "contour_quantiles": cfg["densmap"].get("contour_quantiles", [0.99]),
        "atlas_resolution_um": ATLAS_RESOLUTION_UM.tolist(),
        "input_file": input_path.name,
    }

    save_volume_npz(
        sample_dir / f"{sample_name}_coarse_volumes.npz",
        raw_volumes=raw_volumes,
        smoothed_volumes=smooth_volumes,
        normalized_volumes=norm_volumes,
        metadata=metadata,
        injection_mask=injection_mask,
    )
    sparse_voxel_table(norm_volumes).to_csv(sample_dir / f"{sample_name}_normalized_voxels.tsv", sep="\t", index=False)

    render_outputs(
        sample_name=sample_name,
        boundary_volume=boundary_volume,
        normalized_volumes=norm_volumes,
        injection_mask=injection_mask,
        cfg=cfg,
        output_dir=sample_dir,
    )

    return norm_volumes


def render_outputs(
    sample_name: str,
    boundary_volume: np.ndarray,
    normalized_volumes: dict[str, np.ndarray],
    injection_mask: np.ndarray | None,
    cfg: dict[str, Any],
    output_dir: Path,
) -> None:
    """Render selected coronal sections for all density layers."""
    z_selection = cfg["atlas"]["z_selection"]
    z_values = range(int(z_selection["start"]), int(z_selection["stop"]) + 1, int(z_selection["step"]))

    quantiles = cfg["densmap"].get("contour_quantiles", [0.99])
    levels = {name: contour_levels(vol, quantiles) for name, vol in normalized_volumes.items()}

    layer_defs = cfg["layer_defs"]
    active_layers = cfg["layers"]
    dens_layers = [dict(layer_defs[name], name=name) for name in active_layers if layer_defs[name]["kind"] == "densmap"]

    flip_x = bool(cfg["atlas"].get("flip_x", False))
    flip_y = bool(cfg["atlas"].get("flip_y", True))
    output_shape = boundary_volume.shape[1:]  # Y, X
    coarse_z_count = next(iter(normalized_volumes.values())).shape[0] if normalized_volumes else 0

    for z_fine in z_values:
        zc = coarse_z_from_fine_z(z_fine, coarse_z_count, cfg)
        if boundary_volume is not None:
            boundary2d = boundary_volume[z_index]
        else:
            boundary2d = None

        kde_layers = []
        for layer in dens_layers:
            name = layer["name"]
            if name not in normalized_volumes:
                continue
            dens2d = resize_to_boundary(normalized_volumes[name][zc], output_shape)
            kde_layers.append({
                "name": name,
                "dens2d": dens2d,
                "levels": levels[name],
                "color": layer.get("color", "magenta"),
                "line_width": float(layer.get("lw", cfg["densmap"].get("lw", 0.8))),
                "fill": bool(layer.get("fill", True)),
            })

        inj2d = None
        if injection_mask is not None:
            inj2d = resize_to_boundary(injection_mask[zc], output_shape)

        stem = f"{sample_name}_z{z_fine:03d}"
        render_slice(
            output_dir / f"{stem}.png",
            boundary2d=boundary2d,
            kde_layers=kde_layers,
            injection_mask=inj2d,
            flip_x=flip_x,
            flip_y=flip_y,
            dpi=300,
            transparent=False,
            png_legend=True,   
        )
        render_slice(
            output_dir / f"{stem}.svg",
            boundary2d=boundary2d,
            kde_layers=kde_layers,
            injection_mask=inj2d,
            flip_x=flip_x,
            flip_y=flip_y,
            dpi=300,
            transparent=True,
            draw_boundary=False,
        )


def save_pooled_outputs(
    sample_volumes: list[dict[str, np.ndarray]],
    boundary_volume: np.ndarray,
    cfg: dict[str, Any],
    output_dir: Path,
) -> None:
    """Average normalized volumes across samples and render pooled density maps."""
    if not sample_volumes:
        return

    pooled_dir = output_dir / "POOLED"
    pooled_dir.mkdir(parents=True, exist_ok=True)

    layer_names = sample_volumes[0].keys()
    pooled = {
        name: np.mean([volumes[name] for volumes in sample_volumes if name in volumes], axis=0)
        for name in layer_names
    }

    save_volume_npz(
        pooled_dir / "POOLED_coarse_volumes.npz",
        raw_volumes={},
        smoothed_volumes={},
        normalized_volumes=pooled,
        metadata={
            "n_samples": len(sample_volumes),
            "note": "Pooled map generated by averaging normalized per-sample coarse volumes.",
        },
    )

    render_outputs(
        sample_name="POOLED",
        boundary_volume=boundary_volume,
        normalized_volumes=pooled,
        injection_mask=None,
        cfg=cfg,
        output_dir=pooled_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate atlas-aligned density maps from QuPath tables.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_config(config_path)

    atlas_name = config.get("atlas", {}).get("name", "kim_mouse_10um")
    boundary_volume = load_or_build_boundary_volume(boundary_path, atlas_name=atlas_name)
  
    output_dir = Path(cfg["output"]["root_dir"]) / str(cfg["output"].get("run_tag", "densitymap_output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = [Path(p) for p in cfg["input"]["paths"]]
    sample_volumes = [
        process_one_sample(path, boundary_volume, cfg, output_dir)
        for path in input_paths
    ]

    if cfg["densmap"].get("pooled", {}).get("enabled", True):
        save_pooled_outputs(sample_volumes, boundary_volume, cfg, output_dir)


if __name__ == "__main__":
    main()
