from typing import List, Optional

class AbstractQuantilePdfConstructor():
    def __init__(self, quantiles, locations):
        pass

    # All the math should happen here.
    # This is public so that the user can trigger a recalculation of the of
    # variables needed to reconstruct the original PDF.
    # Should either return functions or set variables that will receive
    # x values and return y values.
    def prepare_constructor(self):
        pass

    # This is the method that the user would most often be interacting with by
    # passing a grid (set of x values) and optionally a list of indexes for
    # for filtering. And receiving back a list of lists of floats.
    def construct_pdf(self, grid, row:Optional[List[int]]=None) -> List[List[float]]:
        pass
