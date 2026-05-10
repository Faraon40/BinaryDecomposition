# decomp — CLI Reference

Run rectangle decomposition algorithms directly from the command line without editing Python scripts.

## Installation

```bash
git clone https://github.com/Faraon40/BinaryDecomposition.git
```

```bash
cd BinaryDecomposition
```

```bash
python -m venv .venv
```

**Linux / macOS**
```bash
source .venv/bin/activate
```

**Windows**
```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

```bash
pip install -e .
```

> For experiment notebooks and SVG preprocessing, use `pip install -r requirements-dev.txt` instead.

## Setup

```bash
source .venv/bin/activate
```

```bash
decomp <command> [options]
```

---

## Commands

### `list` — list algorithms

```bash
decomp list
```

Prints a table of all available algorithms with their speed rating.

### `help` — algorithm details

```bash
decomp help <algorithm>
```

```bash
decomp help dm
```

```bash
decomp help ga_gdm
```

Prints all available flags and an example call for the given algorithm.

### `preprocess` — binarize images

```bash
decomp preprocess --input <path> --output <dir> [options]
```

Converts PNG/JPG images to binary form and saves them as `.npy` arrays and/or binary PNG files.

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Input image or directory | — |
| `--output` | Output directory | — |
| `--format` | `npy`, `png`, or `both` | `both` |
| `--invert` | Dark object on light background | on |
| `--no-invert` | Light object on dark background | — |

```bash
decomp preprocess --input data/datasets/leafs_unique_color/png/ --output results/preprocessed/
```

### `run` — run an algorithm

```bash
decomp run --algo <algorithm> --input <path> --output <dir> [options]
```

Runs the selected algorithm on a single image or an entire directory.

---

## Available Algorithms

### Deterministic

| Key | Name | Speed |
|-----|------|-------|
| `dm` | Delta Method | fast |
| `gdm` | Generalized Delta Method | fast |
| `quadtree` | Quadtree | fast |
| `largest_rect` | Largest Rectangle | fast |
| `graph_based` | Graph-Based (Optimal) | **very slow** (~2.5 min/img) |

### Genetic Algorithm (GA)

| Key | Initialization |
|-----|---------------|
| `ga_dm` | Delta Method |
| `ga_gdm` | Generalized DM — **best overall results** |
| `ga_random` | Random |
| `ga_qtd` | Quadtree |
| `ga_lrf` | Largest Rectangle |

---

## Common `run` Options

| Option | Description |
|--------|-------------|
| `--algo` | Algorithm key (required) |
| `--input` | Input `.npy` file, PNG/JPG, or directory |
| `--output` | Output directory |
| `--no-viz` | Skip visualization PNG |
| `--no-csv` | Skip CSV logging |
| `--verbose` | Show per-image algorithm output |
| `--limit N` | Process at most N images (useful for quick testing) |
| `--dataset-name` | Override dataset label in output paths |
| `--invert` | Dark object on light background (default for PNG) |
| `--no-invert` | Light object on dark background |

---

## Algorithm-Specific Options

### `quadtree`

| Option | Description |
|--------|-------------|
| `--no-full-decomposition` | Use adaptive min_size instead of subdividing to size 2 |
| `--no-trim` | Skip GDM trim of mixed-content leaf rectangles |

### `largest_rect`

| Option | Description | Default |
|--------|-------------|---------|
| `--coverage-threshold` | Stop when F fraction of pixels is covered | `0.95` |

---

## GA Parameters

GA algorithms use a 3-layer parameter system:

```
1. ExperimentConfig defaults   (lowest priority)
         ↓
2. --ga-preset (TOML file)
         ↓
3. Individual --ga-* flags     (highest priority)
```

### Built-in Presets

| Preset | pop_size | generations | patience | Use case |
|--------|----------|-------------|----------|----------|
| `default` | 15 | 150 | 12 | Best time/quality ratio |
| `fast` | 10 | 30 | 3 | Quick testing |
| `research` | 50 | 200 | 20 | Maximum solution quality |

```bash
decomp run --algo ga_gdm --ga-preset fast --input data/datasets/objects_unique/npy/ --output results/ --limit 5
```

```bash
decomp run --algo ga_gdm --ga-preset fast --ga-pop-size 20 --input data/datasets/objects_unique/npy/apple-10_binary.npy --output results/
```

### All GA Options

| Option | Description | Default (default preset) |
|--------|-------------|--------------------------|
| `--ga-preset` | `default`, `fast`, `research`, or path to `.toml` | `default` |
| `--ga-pop-size` | Population size | `15` |
| `--ga-generations` | Maximum generations | `150` |
| `--ga-patience` | Stop after N generations without improvement | `12` |
| `--ga-seed` | Random seed (random if omitted) | random |
| `--ga-elite-size` | Number of elite individuals preserved | `3` |
| `--ga-penalty` | Invalid-pixel penalty multiplier | `2.0` |
| `--ga-crossover` | Crossover method | `subset_greedy` |

**Mutation probabilities:**

| Option | Mutation | Default |
|--------|----------|---------|
| `--ga-p-geometry` | Geometry (G) | `0.30` |
| `--ga-p-merge` | Merge (M) | `0.10` |
| `--ga-p-local` | Local repartition (L) | `0.50` |
| `--ga-p-largest` | Largest-rect (R) | `0.20` |
| `--ga-p-delete` | Delete (D) | `0.20` |
| `--ga-p-split` | Split (S) | `0.20` |
| `--ga-p-shift` | Shift (H) | `0.05` |
| `--ga-p-repair` | Coverage repair | `0.50` |

**Crossover methods** (`--ga-crossover`):
- `subset_greedy` — Subset crossover with greedy extension *(default, best)*
- `single_point` — Single-point crossover
- `two_point` — Two-point crossover
- `uniform` — Uniform crossover

---

## Custom TOML Preset

Copy a built-in preset and modify it:

```bash
cat src/cli/presets/research.toml
```

```toml
# my_preset.toml
pop_size = 15
patience = 12
generations = 150
elite_size = 3
penalty = 2.0
crossover_method = "subset_greedy"
p_delete = 0.30
p_split = 0.30
p_geometry = 0.30
p_shift = 0.1
p_local = 0.50
p_largest = 0.20
p_merge = 0.20
p_repair = 0.50
```

---

## Input Formats

| Format | Handling |
|--------|----------|
| `.npy` | Loaded directly, normalized to {0, 1} |
| `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif` | Auto-binarized (Otsu + morphology) |
| Directory | Finds all `.npy` and raster images, sorted |

Dark object on **white background** (leaves, silhouettes):
→ use default (`--invert` is on)

Light object on **dark background**:
→ use `--no-invert`

---

## Output Structure

```
<output>/
  csv/<algo>/<dataset>/
      results.csv          ← per-image metrics
      generations.csv      ← generation history (GA only)
      errors.log           ← errors (if any)
  rectangles/<algo>/<dataset>/
      <img>_rects_N.json   ← rectangle coordinates
  visualizations/<algo>/<dataset>/
      <img>_rects_N.png    ← solution visualization
```

`results.csv` columns:
- Deterministic: `image_name, rectangle_count, execution_time_sec`
- GA: additionally `seed, pop_size, final_fitness, generations_used`

---

## Examples

```bash
decomp run --algo dm --input data/datasets/objects_unique/npy/ --output results/ --limit 3
```

```bash
decomp run --algo gdm --input data/datasets/objects_unique/npy/ --output results/ --no-viz
```

```bash
decomp run --algo ga_gdm --ga-preset fast --ga-seed 42 --input data/datasets/objects_unique/npy/apple-10_binary.npy --output results/
```

```bash
decomp run --algo ga_gdm --ga-preset research --input data/datasets/objects_unique/npy/ --output results/ --limit 10
```

```bash
decomp preprocess --input data/datasets/leafs_unique_color/png/ --output results/preprocessed/ --format both
```