"""Build ExperimentConfig from parsed CLI arguments."""

from experiments.src.config import ExperimentConfig
from src.cli.registry import GA_ALGOS


def build_config(algo: str, args) -> ExperimentConfig:
    """Construct an ExperimentConfig from CLI arguments.

    For GA algorithms, merges --ga-preset with --ga-* overrides.
    For deterministic algorithms, uses algorithm-specific fields only.

    Parameters
    ----------
    algo : str
        Algorithm key (e.g. 'dm', 'ga_gdm').
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    ExperimentConfig
        Fully populated config ready for the runner.

    """
    seed = getattr(args, "ga_seed", None)

    if algo in GA_ALGOS:
        from src.cli.arg_groups.ga_args import resolve_ga_params
        ga_params = resolve_ga_params(args)
        return ExperimentConfig(
            name=f"cli_{algo}",
            seed=seed,
            algorithm=algo,
            **ga_params,
        )

    config = ExperimentConfig(
        name=f"cli_{algo}",
        seed=seed,
        algorithm=algo,
    )

    if algo == "quadtree":
        config.quadtree_full_decomposition = not getattr(
            args, "no_full_decomposition", False
        )
        config.quadtree_trim = not getattr(args, "no_trim", False)

    if algo == "largest_rect":
        threshold = getattr(args, "coverage_threshold", None)
        if threshold is not None:
            config.largest_rect_coverage = threshold

    return config
