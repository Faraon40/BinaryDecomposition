"""Top-level argparse dispatcher for python -m src.cli."""

import argparse
import sys

from src.cli.arg_groups.ga_args import add_ga_args
from src.cli.commands.list_cmd import handle_help, handle_list
from src.cli.commands.preprocess_cmd import handle_preprocess
from src.cli.commands.run_cmd import handle_run
from src.cli.registry import ALGORITHM_REGISTRY

_ALGO_CHOICES = list(ALGORITHM_REGISTRY.keys())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description=(
            "CLI tool for binary image rectangle decomposition. "
            "Run 'python -m src.cli list' to see available algorithms."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version="bdecomp-cli 1.0",
    )

    sub = parser.add_subparsers(dest="command", metavar="command")

    # ---- list ----
    list_p = sub.add_parser(
        "list", help="Show all available algorithms.",
    )
    list_p.set_defaults(func=handle_list)

    # ---- help ----
    help_p = sub.add_parser(
        "help", help="Show detailed help for a specific algorithm.",
    )
    help_p.add_argument(
        "algo",
        nargs="?",
        choices=_ALGO_CHOICES,
        metavar="algo",
        help=f"Algorithm key. Choices: {', '.join(_ALGO_CHOICES)}",
    )
    help_p.set_defaults(func=handle_help)

    # ---- preprocess ----
    pre_p = sub.add_parser(
        "preprocess",
        help="Binarize raster images (PNG/JPG) to .npy arrays.",
    )
    pre_p.add_argument(
        "--input", "-i", required=True,
        metavar="PATH",
        help="Input image file or directory of images.",
    )
    pre_p.add_argument(
        "--output", "-o", required=True,
        metavar="DIR",
        help="Output directory. Creates npy/ and/or png/ subdirs.",
    )
    pre_p.add_argument(
        "--invert", dest="invert", action="store_true", default=True,
        help=(
            "Dark object on light background (default). "
            "Use --no-invert for light object on dark background."
        ),
    )
    pre_p.add_argument(
        "--no-invert", dest="invert", action="store_false",
    )
    pre_p.add_argument(
        "--format",
        choices=["npy", "png", "both"],
        default="both",
        help="Output format: npy, png, or both. (default: both)",
    )
    pre_p.set_defaults(func=handle_preprocess)

    # ---- run ----
    run_p = sub.add_parser(
        "run",
        help="Run a decomposition algorithm on images.",
    )
    run_p.add_argument(
        "--algo", "-a", required=True,
        choices=_ALGO_CHOICES,
        metavar="ALGO",
        help=(
            f"Algorithm to run. Choices: {', '.join(_ALGO_CHOICES)}. "
            f"Run 'help <algo>' for details."
        ),
    )
    run_p.add_argument(
        "--input", "-i", required=True,
        metavar="PATH",
        help="Input .npy file, raster image, or directory.",
    )
    run_p.add_argument(
        "--output", "-o", required=True,
        metavar="DIR",
        help="Output directory for results.",
    )
    run_p.add_argument(
        "--no-viz", action="store_true", default=False,
        help="Skip saving the visualization PNG.",
    )
    run_p.add_argument(
        "--no-csv", action="store_true", default=False,
        help="Skip CSV logging.",
    )
    run_p.add_argument(
        "--verbose", "-v", action="store_true", default=False,
        help="Show per-image algorithm output.",
    )
    run_p.add_argument(
        "--max-images",
        type=int,
        metavar="N",
        default=None,
        help="Process at most N images (useful for quick testing).",
    )
    run_p.add_argument(
        "--dataset-name",
        metavar="NAME",
        default=None,
        help=(
            "Override dataset label used in output paths. "
            "Defaults to input directory name."
        ),
    )
    run_p.add_argument(
        "--invert", dest="invert", action="store_true", default=True,
        help=(
            "Dark object on light background for PNG inputs (default). "
            "Use --no-invert for light object on dark background."
        ),
    )
    run_p.add_argument(
        "--no-invert", dest="invert", action="store_false",
    )

    # Quadtree-specific flags
    run_p.add_argument(
        "--no-full-decomposition", action="store_true", default=False,
        help="Quadtree: use adaptive min_size instead of subdividing to 2.",
    )
    run_p.add_argument(
        "--no-trim", action="store_true", default=False,
        help="Quadtree: skip GDM trim of mixed-content leaf rectangles.",
    )

    # Largest-rect-specific flag
    run_p.add_argument(
        "--coverage-threshold",
        type=float,
        metavar="F",
        default=None,
        help=(
            "Largest-rect: stop when F fraction of pixels is covered. "
            "(default: 0.95)"
        ),
    )

    add_ga_args(run_p)
    run_p.set_defaults(func=handle_run)

    return parser


def dispatch(argv=None):
    """Parse argv and dispatch to the appropriate command handler.

    Parameters
    ----------
    argv : list of str, optional
        Argument list. Defaults to sys.argv[1:].

    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)

    args.func(args)
