"""Entry point for python -m src.cli."""

from src.cli.dispatcher import dispatch


def main():
    """Entry point for the decomp console script."""
    dispatch()


if __name__ == "__main__":
    main()
