# Project Progress Tracker

**Master's Thesis**: Binary Image Rectangle Decomposition
**Last Updated**: 2026-04-04

---

## High-Level Overview

### ✅ Completed

#### Genetic Algorithm (GA)
- **Initialization strategies**: 3 implemented
  - DM-based initialization
  - Random rectangle generation
  - Quadtree-based initialization
  - GDM-based initialization (added with GDM module)
- **Mutation operators**: 3 implemented
  - Geometry mutation (rectangle dimension modification)
  - Merge mutation (combine adjacent rectangles)
  - Local repartition mutation (re-decompose local regions)
  - GDM-guided local mutation (dual-direction evolution)
- **Crossover strategies**: 4 implemented
  - **Subset Greedy** (default, best performance) - Subset Crossover with Greedy Non-overlapping Extension
  - Single-point crossover
  - Two-point crossover
  - Uniform crossover
- **Fitness function**: Rectangle count + penalty for invalid pixels (scaled by image size)
- **Experiment scripts**: `experiments/scripts/run_ga.py`, `experiments/scripts/run_gdm.py` (GA+GDM hybrid)

#### Quadtree Decomposition
- **Algorithm**: Hierarchical divide-and-conquer approach
- **Features**: Configurable min_size, optional trimming
- **Implementation**: `src/algorithms/quadtree.py`
- **Experiment script**: `experiments/scripts/run_quadtree.py`

#### Integer Linear Programming (ILP)
- **Solvers**: 2 implementations
  - CBC solver (open source) - `src/solvers/ilp_solver_cbc.py`
  - Gurobi solver (commercial, faster) - `src/solvers/ilp_solver_gurobi.py`
- **Rectangle modes**: Maximal and Full enumeration
- **Experiment scripts**: `run_ilp_cbc.py`, `run_ilp_gurobi.py`

#### Graph-Based Decomposition
- **Algorithm**: Flow network-based optimal decomposition
- **Implementation**: `src/algorithms/graph_based.py`
- **Features**: Multiple optimization levels (deterministic, optimal)
- **Status**: Literature claims this is optimal approach - implemented and benchmarked
- **Experiment script**: `experiments/scripts/run_graph_based.py`

#### Delta Method (DM) ✅ NEW
- **Algorithm**: Row-wise or column-wise decomposition into 1-pixel-tall/wide strips
- **Direction**: Auto-selected based on image aspect ratio (row-wise if width ≥ height)
- **Implementation**: `src/algorithms/dm.py` (renamed from RLE)
- **Experiment script**: `experiments/scripts/run_dm.py`
- **Status**: Implemented and benchmarked on research dataset

#### Generalized Delta Method (GDM) ✅ NEW
- **Algorithm**: Extends DM by merging adjacent rows/columns with identical intervals into taller rectangles
- **Features**: Auto min_size calculation based on image size, optional trim/full decomposition, GA init support
- **Implementation**: `src/algorithms/gdm.py`
- **Experiment script**: `experiments/scripts/run_gdm.py`
- **Status**: Implemented and benchmarked; used as GA initialization and guided mutation
- **Reference**: Spiliotis & Mertzios (1998), Suk et al. (2012)

#### Largest-Rect Decomposition ✅
- **Algorithm**: Greedy largest-rectangle-first decomposition
- **Features**: At each step, finds and places the largest axis-aligned rectangle fitting in uncovered foreground pixels (histogram-based)
- **Implementation**: `src/algorithms/largest_rect.py`
- **Experiment script**: `experiments/scripts/run_largest_rect.py`
- **Status**: Implemented and benchmarked on research dataset
- **GA hybrid**: Largest-Rect+GDM hybrid with coverage threshold added to GA pipeline

#### Infrastructure
- **Experiment framework**: Config system, runner, metrics, CSV logging
- **Preprocessing**: Image binarization, data loaders, format converters
- **Utilities**: Integral images, rectangle validation, visualization tools
- **Code quality**: Ruff linting/formatting (PEP 8 + PEP 257)
- **Run tracking**: `run_id` attribute for separating multiple runs of same config

#### Data & Environment
- **Research dataset**: `research_leafs_binary/` (extracted from literature, ready for benchmarking)
- **Other datasets**: icons, synthetic, validation sets, `leafs_binary_fix/`
- **CUDA environment**: `cuda_env` prepared in Conda (for future parallelization)
- **Paper results**: `experiments/results/csv/paper/` contains extracted literature results for comparison

### In Progress
- None

### To Do

#### Experiments & Benchmarking
- **ILP benchmarks**: Run ILP (CBC/Gurobi) on `research_leafs_binary/` dataset
- **Literature comparison**: Systematic comparison with extracted paper values (`csv/paper/`)
- **Performance analysis**: Execution time, memory usage, solution quality metrics
- **Visualization**: Generate comparison charts and result visualizations

#### Performance Optimization (Conditional)
- **CUDA parallelization**: Numba CUDA implementation of GA
  - Trigger: If GA is too slow on large images from research dataset
  - Target: Parallelize fitness evaluation and genetic operators
  - Environment: `cuda_env` already prepared

---

## Roadmap

### Phase 1: Genetic Algorithm Enhancement ✅ COMPLETED
**Goal**: Improve GA performance by testing multiple crossover strategies

- [x] **Research crossover strategies**
- [x] **Implement crossover strategies**
  - [x] Subset Greedy Crossover (default, best performance)
  - [x] Single-point crossover
  - [x] Two-point crossover
  - [x] Uniform crossover
- [x] **Experimental comparison** — Subset Greedy selected as default

**Status**: ✅ COMPLETED

---

### Phase 2: Missing Algorithm Implementations ✅ COMPLETED
**Goal**: Implement remaining algorithms for comprehensive comparison

- [x] **Graph-Based Decomposition** ✅ (2026-02-16)
  - Flow network-based approach with multiple optimization levels

- [x] **Delta Method (DM)** ✅ (2026-03-24 — renamed from RLE)
  - Row-wise/column-wise strip decomposition; auto direction based on aspect ratio
  - Implementation: `src/algorithms/dm.py`
  - Experiment script: `experiments/scripts/run_dm.py`

- [x] **Generalized Delta Method (GDM)** ✅ (2026-03-24)
  - Merges adjacent rows/columns with identical intervals for fewer rectangles
  - GDM-guided local mutation integrated into GA
  - Implementation: `src/algorithms/gdm.py`
  - Experiment script: `experiments/scripts/run_gdm.py`

- [x] **Largest-Rect Decomposition** ✅ (2026-03-24)
  - Largest-rectangle-first greedy decomposition (histogram-based)
  - Largest-Rect+GDM hybrid variant added to GA pipeline
  - Implementation: `src/algorithms/largest_rect.py`
  - Experiment script: `experiments/scripts/run_largest_rect.py`

**Notes**: DM and GDM correspond to "Delta Method" and "Generalized Delta Method" from thesis — previously incorrectly called "RLE".

---

### Phase 3: Benchmarking & Experiments 🔄 IN PROGRESS
**Goal**: Run comprehensive experiments and compare all algorithms

- [x] **Dataset preparation** — `research_leafs_binary/` and `leafs_binary_fix/` ready
- [x] **Run benchmark experiments**
  - [x] GA (with GDM-guided mutation, subset_greedy crossover)
  - [x] GA+GDM hybrid
  - [x] Quadtree
  - [x] Graph-Based Decomposition
  - [x] Morphological Decomposition
  - [x] Delta Method (DM)
  - [x] Generalized Delta Method (GDM)
  - [ ] ILP (CBC and Gurobi)
- [ ] **Results analysis**
  - Literature results available in `experiments/results/csv/paper/`
  - Compare rectangle counts with paper values
  - Analyze execution times
  - Generate visualizations and comparison tables
- [ ] **Documentation**
  - Document findings
  - Update thesis with experimental results

---

### Phase 4: Optimization (Conditional)
**Goal**: Parallelize GA if execution time is too slow on large images

- [ ] **Performance profiling**
- [ ] **CUDA implementation** (if needed)
- [ ] **Integration & testing**

**Notes**: CUDA environment `cuda_env` already prepared in Conda. Only implement if GA is too slow.

---

## Key Files & Locations

### Algorithms
- `src/algorithms/genetic.py` - GA implementation ✅
- `src/algorithms/quadtree.py` - Quadtree implementation ✅
- `src/algorithms/dm.py` - Delta Method ✅
- `src/algorithms/gdm.py` - Generalized Delta Method ✅
- `src/algorithms/largest_rect.py` - Largest-Rect (greedy largest-rectangle-first) ✅
- `src/solvers/ilp_solver_cbc.py` - ILP with CBC ✅
- `src/solvers/ilp_solver_gurobi.py` - ILP with Gurobi ✅
- `src/algorithms/graph_based.py` - Graph-Based ✅

### Experiment Scripts
- `experiments/scripts/run_ga.py` ✅
- `experiments/scripts/run_gdm.py` ✅
- `experiments/scripts/run_dm.py` ✅
- `experiments/scripts/run_largest_rect.py` ✅
- `experiments/scripts/run_quadtree.py` ✅
- `experiments/scripts/run_graph_based.py` ✅
- `experiments/scripts/run_ilp_cbc.py` ✅
- `experiments/scripts/run_ilp_gurobi.py` ✅

### Datasets
- `data/datasets/research_leafs_binary/` - Main benchmark dataset ✅
- `data/datasets/leafs_binary_fix/` - Fixed leaf dataset ✅
- `data/datasets/icons/` - Icon dataset
- `data/datasets/validation/` - Validation set
- `data/datasets/synthetic/` - Synthetic test images

### Results
- `experiments/results/csv/` - CSV experiment logs (per algorithm subdirectory)
- `experiments/results/csv/paper/` - Extracted literature results for comparison
- `experiments/results/rectangles/` - Solution JSON files
- `experiments/results/visualizations/` - PNG visualizations

---

## Current Configuration

### GA Configuration
```python
ExperimentConfig(
    algorithm="ga_quadtree",  # or "ga_dm", "ga_gdm", "ga_random"
    pop_size=20,
    generations=100,
    p_geometry=0.2,   # Geometry mutation
    p_merge=0.2,      # Merge mutation
    p_local=0.3,      # Local repartition mutation
    crossover_method="subset_greedy",  # or "single_point", "two_point", "uniform"
)
```

---

## Notes & Decisions

### Crossover Strategies (Phase 1) ✅
**Completed**: 2026-02-03

**Decision**: Subset Greedy Crossover selected as default for production use based on best performance characteristics.

### Graph-Based Decomposition (Phase 2) ✅
**Completed**: 2026-02-16

- Flow network-based approach for optimal rectangle decomposition
- Good time performance; minimal rectangle count achieved in most cases
- Some edge cases where optimal cut is missed

### Delta Method & Generalized Delta Method (Phase 2) ✅
**Completed**: 2026-03-24

- DM renamed from earlier "RLE" implementation — it is the Delta Method from the literature
- GDM produces significantly fewer rectangles than DM by merging identical-interval rows/columns
- GDM integrated into GA as initialization strategy and guided local mutation operator
- Dual-direction (row-wise + column-wise) evolution added for GA exploration

### Morphological Decomposition (Phase 2) ✅
**Completed**: 2026-03-24

- Largest-rectangle-first greedy strategy
- Morphological+GDM hybrid variant with coverage threshold added to GA pipeline
- Benchmarked on research dataset

### Benchmarking Status (Phase 3) 🔄
**Updated**: 2026-04-03

- Results collected for: DM, GDM, GA+GDM, Graph-Based, Morphological, Quadtree
- Literature comparison data available in `experiments/results/csv/paper/`
- Remaining: ILP runs, full analysis

### Algorithm Implementation Decisions (Phase 2)
- GDM supersedes the earlier "Generalized Delta Method" placeholder — it is fully implemented

### Performance Observations (Phase 3)
_To be filled after full benchmarking and comparison with literature._

### CUDA Implementation Notes (Phase 4)
_To be filled if CUDA parallelization is needed._

---

## Thesis Integration

### Comparison Targets
Extracted literature results are in `experiments/results/csv/paper/GDM_M-GDM_SBD_CBD_research_rectangles.csv`. These will serve as benchmark targets for algorithm comparison.

### Dataset
`research_leafs_binary` dataset corresponds to images used in referenced papers, enabling direct performance comparison.

---

**End of Progress Tracker**