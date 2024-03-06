import unittest

import numpy as np
import casadi as ca

import py_trsqp.utils.filter as filter

INPUT_SYMBOLS = ca.SX.sym('x', 2)

class FilterTest(unittest.TestCase):
       
    CONSTANTS = {}
    CONSTANTS["gamma_vartheta"] = 0.5

    the_filter = filter.FilterSQP(constants=CONSTANTS) 
            
    def test_filter(self):
        
        acc1 = self.the_filter.add_to_filter((2,10), True)
        acc3 = self.the_filter.add_to_filter((11,2), True)
        acc2 = self.the_filter.add_to_filter((4,3), True)
        
        self.assertTrue(acc1)
        self.assertTrue(acc3)
        self.assertTrue(acc2)
        
        acc4 = self.the_filter.add_to_filter((4,3), True)
        self.assertFalse(acc4)
        
        acc5 = self.the_filter.add_to_filter((2,2), True)
        self.assertTrue(acc5)
        
if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()