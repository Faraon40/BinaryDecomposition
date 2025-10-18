from typing import List, Tuple

Rectangle = Tuple[int, int, int, int]  # (x, y, width, height)


class Chromosome:
    """"""
    def __init__(self, rectangles: List[Rectangle]):
        """"""
        self.rectangles = rectangles
        self.fitness = None
