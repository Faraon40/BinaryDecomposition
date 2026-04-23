"""Algorithm registry — single source of truth for CLI."""

GA_ALGOS = frozenset({
    "ga_dm", "ga_gdm", "ga_random", "ga_qtd", "ga_lrf",
})

ALGORITHM_REGISTRY = {
    "dm": {
        "name": "Delta Method",
        "description": (
            "Greedy row/column split decomposition. "
            "Fast and deterministic."
        ),
        "group": "deterministic",
        "speed": "fast",
        "example": (
            "python -m src.cli run --algo dm "
            "--input img.npy --output out/"
        ),
    },
    "gdm": {
        "name": "Generalized Delta Method",
        "description": (
            "Bidirectional DM (row+col), picks best direction. "
            "Fast and deterministic."
        ),
        "group": "deterministic",
        "speed": "fast",
        "example": (
            "python -m src.cli run --algo gdm "
            "--input img.npy --output out/"
        ),
    },
    "quadtree": {
        "name": "Quadtree",
        "description": (
            "Hierarchical divide-and-conquer decomposition. "
            "Accepts --no-full-decomposition and --no-trim."
        ),
        "group": "deterministic",
        "speed": "fast",
        "example": (
            "python -m src.cli run --algo quadtree "
            "--input img.npy --output out/"
        ),
    },
    "largest_rect": {
        "name": "Largest Rectangle",
        "description": (
            "Greedy histogram-based decomposition. "
            "Accepts --coverage-threshold."
        ),
        "group": "deterministic",
        "speed": "fast",
        "example": (
            "python -m src.cli run --algo largest_rect "
            "--coverage-threshold 0.95 --input img.npy --output out/"
        ),
    },
    "graph_based": {
        "name": "Graph-Based (Optimal)",
        "description": (
            "Flow-network optimal decomposition. "
            "WARNING: very slow (~2.5 min/image)."
        ),
        "group": "deterministic",
        "speed": "very slow",
        "example": (
            "python -m src.cli run --algo graph_based "
            "--input img.npy --output out/"
        ),
    },
    "ga_dm": {
        "name": "GA — DM init",
        "description": (
            "Genetic algorithm with Delta Method initialization."
        ),
        "group": "genetic",
        "speed": "slow",
        "example": (
            "python -m src.cli run --algo ga_dm "
            "--ga-preset fast --input img.npy --output out/"
        ),
    },
    "ga_gdm": {
        "name": "GA — GDM init",
        "description": (
            "Genetic algorithm with GDM initialization. "
            "Best overall performance."
        ),
        "group": "genetic",
        "speed": "slow",
        "example": (
            "python -m src.cli run --algo ga_gdm "
            "--ga-preset research --input leafs/ --output out/"
        ),
    },
    "ga_random": {
        "name": "GA — Random init",
        "description": (
            "Genetic algorithm with random population initialization."
        ),
        "group": "genetic",
        "speed": "slow",
        "example": (
            "python -m src.cli run --algo ga_random "
            "--ga-preset default --input img.npy --output out/"
        ),
    },
    "ga_qtd": {
        "name": "GA — Quadtree init",
        "description": (
            "Genetic algorithm with Quadtree initialization."
        ),
        "group": "genetic",
        "speed": "slow",
        "example": (
            "python -m src.cli run --algo ga_qtd "
            "--input img.npy --output out/"
        ),
    },
    "ga_lrf": {
        "name": "GA — LargestRect init",
        "description": (
            "Genetic algorithm with Largest-Rectangle initialization."
        ),
        "group": "genetic",
        "speed": "slow",
        "example": (
            "python -m src.cli run --algo ga_lrf "
            "--input img.npy --output out/"
        ),
    },
}
