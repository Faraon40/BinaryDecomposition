"""CLI command: run — execute a decomposition algorithm."""

import sys

from src.cli.config_builder import build_config
from src.cli.input_handler import (
    derive_dataset_name,
    load_image,
    resolve_input,
)
from src.cli.output_manager import OutputManager
from src.cli.progress import Progress
from src.cli.registry import ALGORITHM_REGISTRY


def handle_run(args):
    """Execute an algorithm on one image or a directory.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments. Expected attributes:
          algo, input, output, invert, verbose,
          no_viz, no_csv, max_images, dataset_name.

    """
    algo = args.algo
    if algo not in ALGORITHM_REGISTRY:
        print(f"Error: unknown algorithm '{algo}'.")
        print(
            "Run 'python -m src.cli list' to see available algorithms."
        )
        sys.exit(1)

    invert = getattr(args, "invert", True)
    max_images = getattr(args, "max_images", None)

    try:
        specs = resolve_input(args.input, invert=invert,
                              max_images=max_images)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    dataset_name = getattr(args, "dataset_name", None) or \
        derive_dataset_name(args.input)

    output_mgr = OutputManager(args.output, algo, dataset_name)
    progress = Progress(total=len(specs), algo=algo)

    from experiments.src.runner import run_single_experiment_from_array

    errors = 0
    for spec in specs:
        image_name = spec.path.name
        try:
            img = load_image(spec)
            config = build_config(algo, args)
            if getattr(args, "verbose", False):
                print(f"  Processing {image_name}...")

            solution, metrics, gen_history = \
                run_single_experiment_from_array(img, config)

            output_mgr.save(
                image_name=image_name,
                img=img,
                solution=solution,
                config=config,
                metrics=metrics,
                generation_history=gen_history,
                no_viz=getattr(args, "no_viz", False),
                no_csv=getattr(args, "no_csv", False),
            )

            progress.update(
                image_name,
                metrics["rectangle_count"],
                metrics["execution_time_sec"],
            )

        except Exception as exc:
            errors += 1
            progress.error(image_name, str(exc))
            output_mgr.log_error(image_name, str(exc))

    progress.done(args.output)
    if errors:
        print(f"Warning: {errors} image(s) failed. See errors.log.")
        sys.exit(1)
