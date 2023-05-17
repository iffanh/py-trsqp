import numpy as np
import casadi as ca
from typing import Tuple

class FilterSQP():
    def __init__(self, constants) -> None:
        self.filters = []
        
        self.constants = constants
    def add_to_filter(self, coordinate:Tuple[float, float]) -> bool:
        
        is_acceptable = False
        self.filters.sort(key=lambda x: x[1])
        if len(self.filters) == 0:
            self.filters.append(coordinate)
            is_acceptable = True
        
        else:
            g = self.constants['gamma_vartheta']
            cond = True
            for coord in self.filters:
                cond1 = coordinate[1] < (1-g)*coord[1]
                cond2 = coordinate[0] < coord[0] - g*coordinate[1]
                _cond = cond1 or cond2
                cond = cond and _cond
            if cond:
                is_acceptable = True
                    
            if is_acceptable:
                curr_filter = [coordinate]
                for coord in self.filters:
                    if coord[1] >= (1-g)*coordinate[1] and coord[0] >= coordinate[0] - g*coord[1]:
                        # curr_filter.append(coordinate)
                        pass
                    else:
                        curr_filter.append(coord)
                self.filters = curr_filter
        self.filters.sort(key=lambda x: x[1])
        
        return is_acceptable
        