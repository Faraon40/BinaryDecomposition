"""Type definitions for genetic algorithm."""

from typing import List, Tuple

Rectangle = Tuple[int, int, int, int]  # (x, y, width, height)


class Chromosome:
    """Chromosome representing a solution as list of rectangles."""

    def __init__(self, rectangles: List[Rectangle]):
        """Initialize chromosome.

        Args:
            rectangles: List of rectangle tuples.

        """
        self.rectangles = rectangles
        self.fitness = None
