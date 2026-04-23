"""CLI commands: list and help."""

from src.cli.registry import ALGORITHM_REGISTRY, GA_ALGOS


def handle_list(args):
    """Print a table of all available algorithms."""
    det = {
        k: v for k, v in ALGORITHM_REGISTRY.items()
        if v["group"] == "deterministic"
    }
    ga = {
        k: v for k, v in ALGORITHM_REGISTRY.items()
        if v["group"] == "genetic"
    }

    print()
    print("Available algorithms")
    print("=" * 60)

    print("\nDeterministic:")
    for key, meta in det.items():
        speed = f"[{meta['speed']}]"
        print(f"  {key:<14}  {meta['name']:<28}  {speed}")

    print("\nGenetic Algorithm (GA):")
    for key, meta in ga.items():
        print(f"  {key:<14}  {meta['name']}")

    print()
    print("Run an algorithm:")
    print(
        "  python -m src.cli run --algo <key> "
        "--input <path> --output <dir>"
    )
    print("Show algorithm details:")
    print("  python -m src.cli help <algo>")
    print()


def handle_help(args):
    """Print detailed help for a specific algorithm."""
    algo = getattr(args, "algo", None)

    if algo is None:
        handle_list(args)
        return

    if algo not in ALGORITHM_REGISTRY:
        print(f"Unknown algorithm: '{algo}'")
        print(
            "Run 'python -m src.cli list' to see available algorithms."
        )
        return

    meta = ALGORITHM_REGISTRY[algo]
    is_ga = algo in GA_ALGOS

    print()
    print(f"Algorithm: {algo}  —  {meta['name']}")
    print("-" * 60)
    print(f"Group:       {meta['group']}")
    print(f"Speed:       {meta['speed']}")
    print(f"Description: {meta['description']}")
    print()
    print("Common flags:")
    print("  --input    <file|dir>   Input image or directory")
    print("  --output   <dir>        Output directory")
    print("  --no-viz                Skip visualization PNG")
    print("  --no-csv                Skip CSV logging")
    print("  --verbose               Show algorithm output")
    print("  --max-images N          Limit batch to N images")
    print(
        "  --dataset-name <name>   Override dataset label in output paths"
    )
    print(
        "  --invert / --no-invert  Binarization direction for PNG inputs"
    )

    if algo == "quadtree":
        print()
        print("Quadtree-specific flags:")
        print(
            "  --no-full-decomposition  "
            "Use adaptive min_size instead of subdividing to min_size=2"
        )
        print("  --no-trim                Skip GDM trim of mixed leaves")

    if algo == "largest_rect":
        print()
        print("Largest-rect-specific flags:")
        print(
            "  --coverage-threshold F   "
            "Stop when F fraction of pixels is covered (default: 0.95)"
        )

    if is_ga:
        print()
        print("GA parameters (preset + optional overrides):")
        print(
            "  --ga-preset     default | fast | research | <path.toml>"
        )
        print("  --ga-pop-size   N     Population size")
        print("  --ga-generations N   Maximum generations")
        print("  --ga-patience   N    Early-stopping patience")
        print("  --ga-seed       N    Random seed (random if omitted)")
        print("  --ga-elite-size N    Elite individuals preserved")
        print("  --ga-penalty    F    Invalid-pixel penalty multiplier")
        print(
            "  --ga-crossover  subset_greedy | single_point | "
            "two_point | uniform"
        )
        print()
        print("  Mutation probabilities:")
        print("    --ga-p-geometry F   Geometry mutation (G)")
        print("    --ga-p-merge    F   Merge mutation (M)")
        print("    --ga-p-local    F   Local repartition mutation (L)")
        print("    --ga-p-largest  F   Largest-rect mutation (R)")
        print("    --ga-p-delete   F   Delete mutation (D)")
        print("    --ga-p-split    F   Split mutation (S)")
        print("    --ga-p-shift    F   Shift mutation (H)")
        print("    --ga-p-repair   F   Coverage repair probability")

    print()
    print("Example:")
    print(f"  {meta['example']}")
    print()
