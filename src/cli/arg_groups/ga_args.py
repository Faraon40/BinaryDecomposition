"""GA argument group and preset resolver for the CLI."""

from pathlib import Path

# Built-in presets as Python dicts — avoids TOML parsing for common cases
_BUILTIN_PRESETS = {
    "default": {
        "pop_size": 30,
        "generations": 100,
        "elite_size": 3,
        "penalty": 2.0,
        "patience": 5,
        "crossover_method": "subset_greedy",
        "p_geometry": 0.30,
        "p_merge": 0.10,
        "p_local": 0.50,
        "p_largest": 0.20,
        "p_delete": 0.20,
        "p_split": 0.20,
        "p_shift": 0.05,
        "p_repair": 0.50,
    },
    "fast": {
        "pop_size": 10,
        "generations": 30,
        "elite_size": 2,
        "penalty": 2.0,
        "patience": 3,
        "crossover_method": "subset_greedy",
        "p_geometry": 0.30,
        "p_merge": 0.10,
        "p_local": 0.50,
        "p_largest": 0.20,
        "p_delete": 0.20,
        "p_split": 0.20,
        "p_shift": 0.05,
        "p_repair": 0.50,
    },
    "research": {
        "pop_size": 50,
        "generations": 200,
        "elite_size": 3,
        "penalty": 2.0,
        "patience": 10,
        "crossover_method": "subset_greedy",
        "p_geometry": 0.30,
        "p_merge": 0.10,
        "p_local": 0.50,
        "p_largest": 0.20,
        "p_delete": 0.20,
        "p_split": 0.20,
        "p_shift": 0.05,
        "p_repair": 0.50,
    },
}

_CROSSOVER_CHOICES = [
    "subset_greedy", "single_point", "two_point", "uniform",
]


def add_ga_args(parser):
    """Add --ga-* argument group to an argparse parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to extend with GA-specific arguments.

    """
    g = parser.add_argument_group(
        "GA parameters",
        "Used only for ga_dm / ga_gdm / ga_random / ga_qtd / "
        "ga_lrf / ga_mixed algorithms.",
    )
    g.add_argument(
        "--ga-preset",
        metavar="NAME|PATH",
        default="default",
        help=(
            "Built-in preset (default|fast|research) or path to a "
            "custom .toml file. CLI flags override preset values. "
            "(default: default)"
        ),
    )
    g.add_argument(
        "--ga-pop-size",
        type=int,
        metavar="N",
        default=None,
        help="Population size (overrides preset).",
    )
    g.add_argument(
        "--ga-generations",
        type=int,
        metavar="N",
        default=None,
        help="Maximum number of generations (overrides preset).",
    )
    g.add_argument(
        "--ga-patience",
        type=int,
        metavar="N",
        default=None,
        help="Early-stopping patience in generations (overrides preset).",
    )
    g.add_argument(
        "--ga-seed",
        type=int,
        metavar="N",
        default=None,
        help="Random seed. Omit for a random seed per run.",
    )
    g.add_argument(
        "--ga-elite-size",
        type=int,
        metavar="N",
        default=None,
        help="Number of elite individuals preserved (overrides preset).",
    )
    g.add_argument(
        "--ga-penalty",
        type=float,
        metavar="F",
        default=None,
        help="Penalty multiplier for invalid solutions (overrides preset).",
    )
    g.add_argument(
        "--ga-crossover",
        choices=_CROSSOVER_CHOICES,
        default=None,
        metavar="METHOD",
        help=(
            f"Crossover method. Choices: "
            f"{', '.join(_CROSSOVER_CHOICES)}. (overrides preset)"
        ),
    )
    g.add_argument(
        "--ga-p-geometry",
        type=float,
        metavar="P",
        default=None,
        help="Probability of geometry mutation (G) (overrides preset).",
    )
    g.add_argument(
        "--ga-p-merge",
        type=float,
        metavar="P",
        default=None,
        help="Probability of merge mutation (M) (overrides preset).",
    )
    g.add_argument(
        "--ga-p-local",
        type=float,
        metavar="P",
        default=None,
        help=(
            "Probability of local repartition mutation (L) "
            "(overrides preset)."
        ),
    )
    g.add_argument(
        "--ga-p-largest",
        type=float,
        metavar="P",
        default=None,
        help=(
            "Probability of largest-rect mutation (R) "
            "(overrides preset)."
        ),
    )
    g.add_argument(
        "--ga-p-delete",
        type=float,
        metavar="P",
        default=None,
        help="Probability of delete mutation (D) (overrides preset).",
    )
    g.add_argument(
        "--ga-p-split",
        type=float,
        metavar="P",
        default=None,
        help="Probability of split mutation (S) (overrides preset).",
    )
    g.add_argument(
        "--ga-p-shift",
        type=float,
        metavar="P",
        default=None,
        help="Probability of shift mutation (H) (overrides preset).",
    )
    g.add_argument(
        "--ga-p-repair",
        type=float,
        metavar="P",
        default=None,
        help=(
            "Probability of coverage repair after mutation "
            "(overrides preset)."
        ),
    )


def resolve_ga_params(args) -> dict:
    """Merge preset values with CLI overrides into a GA params dict.

    Priority: ExperimentConfig defaults < preset < CLI --ga-* flags.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    dict
        GA parameter dict ready to unpack into ExperimentConfig.

    """
    preset_name = getattr(args, "ga_preset", "default") or "default"
    params = _load_preset(preset_name)

    _overrides = {
        "ga_pop_size": "pop_size",
        "ga_generations": "generations",
        "ga_elite_size": "elite_size",
        "ga_penalty": "penalty",
        "ga_patience": "patience",
        "ga_crossover": "crossover_method",
        "ga_p_geometry": "p_geometry",
        "ga_p_merge": "p_merge",
        "ga_p_local": "p_local",
        "ga_p_largest": "p_largest",
        "ga_p_delete": "p_delete",
        "ga_p_split": "p_split",
        "ga_p_shift": "p_shift",
        "ga_p_repair": "p_repair",
    }

    for cli_attr, field_name in _overrides.items():
        val = getattr(args, cli_attr, None)
        if val is not None:
            params[field_name] = val

    return params


def _load_preset(name: str) -> dict:
    """Load a preset by name or file path.

    Parameters
    ----------
    name : str
        Built-in preset name (default|fast|research) or path to
        a custom .toml file.

    Returns
    -------
    dict
        Copy of the preset parameter dict.

    Raises
    ------
    FileNotFoundError
        If a custom .toml path does not exist.
    ImportError
        If tomllib/tomli is not available for custom .toml parsing.

    """
    key = name[:-5] if name.endswith(".toml") else name
    if key in _BUILTIN_PRESETS:
        return dict(_BUILTIN_PRESETS[key])

    path = Path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"GA preset file not found: {name}\n"
            f"Built-in presets: default, fast, research"
        )

    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # pip install tomli
        except ImportError:
            raise ImportError(
                "Parsing custom .toml presets requires Python 3.11+ "
                "or 'pip install tomli'."
            )

    with open(path, "rb") as f:
        return tomllib.load(f)
