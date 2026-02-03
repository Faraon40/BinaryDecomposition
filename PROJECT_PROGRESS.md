# Project Progress Tracker

**Master's Thesis**: Binary Image Rectangle Decomposition
**Last Updated**: 2026-02-03

---

## High-Level Overview

### ✅ Completed

#### Genetic Algorithm (GA)
- **Initialization strategies**: 3 implemented
  - RLE-based initialization
  - Random rectangle generation
  - Quadtree-based initialization
- **Mutation operators**: 3 implemented
  - Geometry mutation (rectangle dimension modification)
  - Merge mutation (combine adjacent rectangles)
  - Local repartition mutation (re-decompose local regions)
- **Crossover strategies**: 4 implemented
  - **Subset Greedy** (default, best performance) - Subset Crossover with Greedy Non-overlapping Extension
  - Single-point crossover
  - Two-point crossover
  - Uniform crossover
- **Fitness function**: Rectangle count + penalty for invalid pixels
- **Experiment script**: `experiments/scripts/run_ga.py`

#### Quadtree Decomposition
- **Algorithm**: Hierarchical divide-and-conquer approach
- **Features**: Configurable min_size, optional trimming
- **Implementation**: `src/algorithms/quadtree.py`

#### Integer Linear Programming (ILP)
- **Solvers**: 2 implementations
  - CBC solver (open source) - `src/solvers/ilp_solver_cbc.py`
  - Gurobi solver (commercial, faster) - `src/solvers/ilp_solver_gurobi.py`
- **Rectangle modes**: Maximal and Full enumeration
- **Experiment scripts**: `run_ilp_cbc.py`, `run_ilp_gurobi.py`

#### Infrastructure
- **Experiment framework**: Config system, runner, metrics, CSV logging
- **Preprocessing**: Image binarization, data loaders, format converters
- **Utilities**: Integral images, rectangle validation, visualization tools
- **Code quality**: Ruff linting/formatting (PEP 8 + PEP 257)

#### Data & Environment
- **Research dataset**: `research_leafs_binary/` (extracted from literature, ready for benchmarking)
- **Other datasets**: icons, synthetic, validation sets
- **CUDA environment**: `cuda_env` prepared in Conda (for future parallelization)

### In Progress
- None

### To Do

#### New Algorithm Implementations
- **Graph-Based Decomposition**: Literature claims this is optimal approach
  - Status: Not implemented
  - Needs: Research paper review, implementation in `src/algorithms/`

- **DTD (Distance Transformation Decomposition)**:
  - Status: Not implemented
  - Mentioned in thesis theoretical part

- **Delta Method**: Standalone implementation for comparison
  - Status: Not implemented
  - Needed for comprehensive algorithm comparison

- **Generalized Delta Method**: Extended Delta approach
  - Status: Not implemented
  - Separate from Delta Method

#### Experiments & Benchmarking
- **Comprehensive benchmark**: Run all algorithms on `research_leafs_binary/` dataset
- **Literature comparison**: Compare results with extracted values from thesis figures/tables
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
  - Review literature for GA crossover methods suitable for rectangle decomposition
  - Identify 3-5 promising strategies to implement

- [x] **Implement crossover strategies**
  - [x] Strategy 1: Subset Greedy Crossover (Subset Crossover with Greedy Non-overlapping Extension)
  - [x] Strategy 2: Single-point crossover
  - [x] Strategy 3: Two-point crossover
  - [x] Strategy 4: Uniform crossover

- [x] **Experimental comparison**
  - Select best crossover strategy for final GA implementation
  - **Result**: Subset Greedy Crossover selected as default (best performance)

**Status**: ✅ COMPLETED - 4 crossover strategies implemented and integrated. Configuration system supports selecting any crossover method via `crossover_method` parameter.

---

### Phase 2: Missing Algorithm Implementations
**Goal**: Implement remaining algorithms for comprehensive comparison

- [ ] **Graph-Based Decomposition**
  - Research implementation details from literature
  - Implement algorithm (literature claims this is optimal)
  - Create experiment script `experiments/scripts/run_graph_based.py`

- [ ] **DTD (Distance Transformation Decomposition)**
  - Research DTD algorithm
  - Implement in `src/algorithms/dtd.py`
  - Create experiment script

- [ ] **Delta Method**
  - Implement standalone Delta Method
  - Create experiment script

- [ ] **Generalized Delta Method**
  - Implement Generalized Delta Method
  - Create experiment script

**Notes**: These algorithms are mentioned in thesis theoretical part and needed for comprehensive comparison.

---

### Phase 3: Benchmarking & Experiments
**Goal**: Run comprehensive experiments and compare all algorithms

- [ ] **Dataset preparation**
  - Verify `data/datasets/research_leafs_binary/` is complete
  - Document dataset characteristics (image sizes, complexity)
  - Create dataset metadata file

- [ ] **Run benchmark experiments**
  - [ ] GA with best crossover strategy
  - [ ] Quadtree
  - [ ] Graph-Based Decomposition
  - [ ] DTD
  - [ ] Delta Method
  - [ ] Generalized Delta Method
  - [ ] ILP (CBC and Gurobi)

- [ ] **Results analysis**
  - Compare rectangle counts with literature results (thesis figures/tables)
  - Analyze execution times
  - Generate visualizations
  - Create comparison tables

- [ ] **Documentation**
  - Document findings
  - Update thesis with experimental results

**Reference**: Extracted results from literature are in thesis figures/tables for comparison.

---

### Phase 4: Optimization (Conditional)
**Goal**: Parallelize GA if execution time is too slow on large images

- [ ] **Performance profiling**
  - Profile GA execution on large images from research dataset
  - Identify bottlenecks
  - Decide if CUDA parallelization is necessary

- [ ] **CUDA implementation** (if needed)
  - Activate `cuda_env` Conda environment
  - Install Numba CUDA dependencies
  - Parallelize GA fitness evaluation
  - Parallelize mutation/crossover operations
  - Benchmark CUDA vs CPU performance

- [ ] **Integration & testing**
  - Integrate CUDA-accelerated GA into experiment pipeline
  - Verify correctness (results should match CPU version)
  - Re-run large-scale experiments if needed

**Notes**: CUDA environment `cuda_env` already prepared in Conda. Only implement if GA is too slow.

---

## Key Files & Locations

### Algorithms
- `src/algorithms/genetic.py` - GA implementation ✅
- `src/algorithms/quadtree.py` - Quadtree implementation ✅
- `src/solvers/ilp_solver_cbc.py` - ILP with CBC ✅
- `src/solvers/ilp_solver_gurobi.py` - ILP with Gurobi ✅
- `src/algorithms/graph_based.py` - Graph-Based ❌
- `src/algorithms/dtd.py` - DTD ❌
- `src/algorithms/delta.py` - Delta Method ❌
- `src/algorithms/generalized_delta.py` - Generalized Delta ❌

### Experiment Scripts
- `experiments/scripts/run_ga.py` ✅
- `experiments/scripts/run_ilp_cbc.py` ✅
- `experiments/scripts/run_ilp_gurobi.py` ✅
- Other algorithm scripts ❌

### Datasets
- `data/datasets/research_leafs_binary/` - Main benchmark dataset ✅
- `data/datasets/icons/` - Icon dataset
- `data/datasets/synthetic/` - Synthetic test images

### Results
- `experiments/results/rectangles/` - Solution JSON files
- `experiments/results/visualizations/` - PNG visualizations
- `experiments/results/logs/` - CSV experiment logs

---

## Current Configuration

### GA Configuration (from CLAUDE.md)
```python
ExperimentConfig(
    algorithm="ga_quadtree",  # or "ga_rle", "ga_random"
    pop_size=20,
    generations=100,
    p_geometry=0.2,   # Geometry mutation
    p_merge=0.2,      # Merge mutation
    p_local=0.3,      # Local repartition mutation
    crossover_method="subset_greedy",  # or "single_point", "two_point", "uniform"
)
```

### Mutation Naming
- **GML**: Geometry + Merge + Local
- Set `p_*=0.0` to disable a mutation type

---

## Notes & Decisions

### Crossover Strategies (Phase 1) ✅
**Completed**: 2026-02-03

**Implemented Methods**:
1. **Subset Greedy Crossover** (`subset_greedy`) - DEFAULT
   - Full name: Subset Crossover with Greedy Non-overlapping Extension
   - Selects random subset from Parent 1, greedily adds non-overlapping rectangles from Parent 2
   - Best performance and speed

2. **Single-Point Crossover** (`single_point`)
   - Split parents at random point, merge first part of P1 with second part of P2

3. **Two-Point Crossover** (`two_point`)
   - Take middle section from one parent, ends from other parent

4. **Uniform Crossover** (`uniform`)
   - Each gene selected independently with probability p (default 0.5)

**Configuration**: Use `crossover_method` parameter in `run_ga()` or `ExperimentConfig`

**Decision**: Subset Greedy Crossover selected as default for production use based on best performance characteristics.

### Algorithm Implementation Decisions (Phase 2)
_To be filled during implementation_

### Performance Observations (Phase 3)
_To be filled during benchmarking_

### CUDA Implementation Notes (Phase 4)
_To be filled if CUDA parallelization is needed_

---

## Thesis Integration

### Comparison Targets
Results extracted from literature papers are in thesis figures/tables. These will be used as benchmark targets for algorithm comparison.

### Dataset
`research_leafs_binary` dataset corresponds to images used in referenced papers, enabling direct comparison of algorithm performance.

---

**End of Progress Tracker**