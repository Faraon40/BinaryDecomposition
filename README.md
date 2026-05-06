# Binary Image Rectangle Decomposition

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Ruff](https://img.shields.io/badge/code%20style-ruff-orange)

Master's thesis project implementing and comparing algorithms for **minimal rectangle decomposition of binary images** — covering all foreground pixels with the minimum number of non-overlapping axis-aligned rectangles.

## Problem

Given a binary image (a 2D array of 0s and 1s), find the smallest set of non-overlapping rectangles that exactly covers all 1-pixels. This is an NP-hard combinatorial optimization problem with applications in document analysis, PCB layout, and image compression.

## Algorithms

| Algorithm | Key | Type | Notes |
|---|---|---|---|
| Delta Method | `dm` | Deterministic | Fast greedy baseline |
| Generalized Delta Method | `gdm` | Deterministic | Improved DM with post-processing |
| Quadtree | `quadtree` | Deterministic | Hierarchical divide-and-conquer |
| Largest Rectangle | `largest_rect` | Deterministic | Histogram-based greedy |
| Graph-Based | `graph_based` | Deterministic | Flow-network optimal (~2.5 min/img) |
| Genetic Algorithm | `ga_gdm`, `ga_dm`, … | Evolutionary | GDM-initialized GA, best results |

All algorithms use **integral images** for O(1) rectangle-sum queries. The GA supports 7 mutation operators (geometry, merge, local, delete, split, shift, largest-rect) and 4 crossover methods (`subset_greedy` recommended).

## Installation

```bash
git clone https://github.com/Faraon40/BinaryDecomposition.git
cd BinaryDecomposition
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

```bash
source .venv/bin/activate

# List available algorithms
python -m src.cli list

# Run GDM on a folder of images
python -m src.cli run --algo gdm \
  --input data/datasets/leafs_selected/npy/ \
  --output results/

# Run GA (GDM-initialized) with fast preset on a single image
python -m src.cli run --algo ga_gdm \
  --ga-preset fast \
  --input data/datasets/leafs_selected/npy/Betula_pendula.npy \
  --output results/

# Binarize your own images first
python -m src.cli preprocess \
  --input ~/my_images/ \
  --output data/datasets/my_dataset/
```

See [`docs/cli_usage.md`](docs/cli_usage.md) for the full CLI reference including all GA hyperparameters, presets, and output format details.

## GA Presets

| Preset | pop_size | generations | patience | Use case |
|---|---|---|---|---|
| `fast` | 10 | 30 | 3 | Quick testing |
| `default` | 15 | 150 | 12 | Best time/quality ratio |
| `research` | 50 | 200 | 20 | Maximum quality |

```bash
python -m src.cli run --algo ga_gdm --ga-preset research \
  --input data/datasets/leafs_selected/npy/ --output results/
```

## Project Structure

```
src/
  algorithms/       # Core algorithm implementations
  solvers/          # ILP solvers (CBC, Gurobi)
  utils/            # Integral images, visualization, types
  cli/              # CLI entry point and presets

experiments/
  scripts/          # Full-dataset production runs
  scripts/analysis/ # Hyperparameter search scripts
  notebooks/        # Analysis dashboards and visualizations
  results/          # CSV logs, rectangle JSONs, PNGs (gitignored)
  src/              # Shared experiment infrastructure (config, runner, logger)

data/datasets/      # Benchmark datasets (.npy arrays + manifest.csv)
docs/               # CLI usage guide
```

## Datasets

| Dataset | N | Purpose |
|---|---|---|
| `leafs_selected` | 204 | Full benchmark |
| `objects_selected` | 282 | Full benchmark (70 object categories) |
| `objects_binary` | 1402 | Source dataset (70 categories, ~20/each) |
| `objects_unique` | 70 | 1 image per category |
| `analysis/leafs_quartile` | 20 | Hyperparameter search (leafs) |
| `analysis/objects_quartile` | 20 | Hyperparameter search (objects) |
| `analysis/large_image_dataset` | 6 | Scalability tests |

Images are stored as `.npy` arrays (values 0/1). Each dataset includes `manifest.csv` with image name, dimensions, and pixel count.

**Sources:**
- Leaf datasets (`leafs_*`) — Department of Image Processing. *Tree Leaf Database MEW2019*. Institute of Information Theory and Automation, Czech Academy of Sciences, 2019. https://zoi.utia.cas.cz/index.php/research/downloads/tree-leaf-database-mew2019
- Object datasets (`objects_*`) — Ralph, R. *MPEG-7 Shape Dataset*. 2000. https://dabi.temple.edu/external/shape/MPEG7/dataset.html

## Reproducing Experiments

```bash
# Full benchmark runs
python -m experiments.scripts.run_gdm
python -m experiments.scripts.run_ga
python -m experiments.scripts.run_graph_based

# Hyperparameter analysis
python -m experiments.scripts.analysis.run_ga_init_comparison
python -m experiments.scripts.analysis.run_ga_crossover_comparison
python -m experiments.scripts.analysis.run_ga_mutation_analysis
```

Results are written to `experiments/results/csv/` and can be explored in `experiments/notebooks/dashboard/experiment_dashboard.ipynb`.

## References

- Ferrari, L., Sankar, P., & Sklansky, J. (1984). Minimal Rectangular Partitions of Digitized Blobs. *Computer Vision, Graphics, and Image Processing*, 28, 58–71. https://doi.org/10.1016/0734-189x(84)90139-7
- Suk, T., Höschl, C., & Flusser, J. (2012). Decomposition of binary images — A survey and comparison. *Pattern Recognition*, 45(12), 4279–4291. https://doi.org/10.1016/j.patcog.2012.05.012

## License

[MIT](LICENSE)
