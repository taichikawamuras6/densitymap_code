# Density map code
This repository contains the Python code used to generate atlas-aligned density maps from QuPath/ABBA-exported coordinate tables.

The code takes QuPath TSV/CSV files containing atlas coordinates (`Atlas_X`, `Atlas_Y`, `Atlas_Z`) and renders density-map contours on a binary atlas-boundary volume. It also saves intermediate coarse-grid volumes for inspection and reuse.

## Contents
```text
src/
  generate_density_maps.py   Main script
  plot_atlas_layers.py       Minimal rendering utilities

configs/
  vsub_phpeb_densitymap_amg.yml  Example config for vSUB dsATT / AAV-PHP.eB data
  vsub_atg_rtg_densitymap_amg.yml Example config for conventional ATG/RTG tracing data

data/qupath_exports/
  Example QuPath-exported TSV files

data/atlas/
  Example binary atlas-boundary volume 
  (based on Kim laboratory "Enhanced and Unified Mouse Brain Atlas") 
  or Place another atlas here
```

## Input data
The script expects QuPath/ABBA-exported TSV or CSV files. The original QuPath column names are kept unchanged. The config file specifies which columns are used:

```yaml
columns:
  x: "Atlas_X"
  y: "Atlas_Y"
  z: "Atlas_Z"
  object_type: "Object type"
  classification: "Classification"
  name: "Name"
```

Coordinates are assumed to be in millimeters and are converted to micrometers using:

```yaml
unit_factor: 1000.0
```

## Atlas boundary volume
The script also requires a binary atlas-boundary volume saved as a NumPy `.npy` file.

Expected format:

- 3D NumPy array
- axis order: `(Z, Y, X)`
- value `1`: atlas boundary pixel
- value `0`: background
- atlas grid resolution: 10 µm isotropic

Place the file at:

```text
data/atlas/kim_mouse_10um_boundaries_uint8.npy
```

or edit `atlas.boundary_npy_path` in the config file.

If the boundary file is missing, the script attempts to generate it from the BrainGlobe atlas.
This requires bg-atlasapi and scikit-image.

## Installation
Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

Tested with Python 3.11.

## Usage
From the repository root:

```bash
python src/generate_density_maps.py --config configs/vsub_phpeb_densitymap.yml
```

or

```bash
python src/generate_density_maps.py --config configs/vsub_atg_rtg_densitymap.yml
```

Before running on the full dataset, edit the following fields in the config file:

```yaml
input:
  paths:
    - "path/to/your_qupath_export.tsv"

atlas:
  boundary_npy_path: "path/to/your_boundary_volume.npy"

output:
  root_dir: "outputs"
```

## Output
For each input sample, the script saves:

- rendered density-map slices (`.png`, `.svg`)
- coarse-grid volume data (`*_coarse_volumes.npz`)
- sparse table of non-zero normalized voxels (`*_normalized_voxels.tsv`)

If pooling is enabled, averaged normalized volumes are saved in:

```text
outputs/<run_tag>/POOLED/
```

## Density map parameters
The key parameters are defined in the YAML config:

```yaml
densmap:
  method: "kde" # or "histogram"
  coarse_voxel_um:
    xy: 50
    z: 300
  bandwidth_um:
    xy: 75
    z: 450
  contour_quantiles: [0.99]
```

For `method: "histogram"`, objects are first aggregated into coarse voxels and then smoothed with a 3D Gaussian filter.

For `method: "kde"`, each object contributes an anisotropic Gaussian kernel directly on the coarse grid.

## Notes on column names
The QuPath-exported column names are intentionally left unchanged to make the processing traceable to the exported tables. If you rename columns, update the `columns:` section in the config file accordingly.

## License
This code is released under the MIT License.
