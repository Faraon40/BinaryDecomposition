"""Simple console progress printer for batch CLI runs."""

import sys


class Progress:
    """Print per-image progress and summary for batch runs.

    Parameters
    ----------
    total : int
        Total number of images to process.
    algo : str
        Algorithm name shown in the header line.

    """

    def __init__(self, total: int, algo: str):
        self.total = total
        self.algo = algo
        self._current = 0
        print(f"Running {algo} on {total} image(s)...")

    def update(self, image_name: str, rect_count: int, elapsed: float):
        """Print one progress line after processing an image.

        Parameters
        ----------
        image_name : str
            Filename of the processed image.
        rect_count : int
            Number of rectangles in the solution.
        elapsed : float
            Processing time in seconds.

        """
        self._current += 1
        idx = f"[{self._current}/{self.total}]"
        timing = f"{elapsed:.1f}s"
        rects = f"{rect_count} rects"
        print(f"  {idx} {image_name}  {rects}  ({timing})")
        sys.stdout.flush()

    def error(self, image_name: str, msg: str):
        """Print an error line for a failed image.

        Parameters
        ----------
        image_name : str
            Filename that failed.
        msg : str
            Error message.

        """
        self._current += 1
        idx = f"[{self._current}/{self.total}]"
        print(f"  {idx} {image_name}  ERROR: {msg}")
        sys.stdout.flush()

    def done(self, output_dir: str):
        """Print final summary line.

        Parameters
        ----------
        output_dir : str
            Directory where results were saved.

        """
        print(f"Done. Results saved to: {output_dir}")
