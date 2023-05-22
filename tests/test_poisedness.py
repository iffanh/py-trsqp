import unittest

import numpy as np
import casadi as ca

import py_trsqp.utils.lagrange_polynomial as lp

INPUT_SYMBOLS = ca.SX.sym('x', 2)

def cf(x):
    return x[0]**2 + x[1]**2 + x[0]*x[1]*1

YPROB = np.array([[0.97340063, 0.97307894, 0.97303661, 0.97295692, 0.97278178, 0.97340063],
                    [0.94716219, 0.94684049, 0.94652159, 0.94716219, 0.94696372, 0.94651879]])

FPROB = cf(YPROB)
RPROB = 3.22E-04

class PoisednessTest(unittest.TestCase):
    
    # with function value
    l = lp.LagrangePolynomials(input_symbols=INPUT_SYMBOLS, 
                                   pdegree=2)
    l.initialize(y=YPROB, f=FPROB, sort_type='function')    
            
    def test_poisedness(self):

        ## Tend to go crazy when done incorrectly
        p = self.l.poisedness(rad=RPROB, center=YPROB[:,0]).max_poisedness()
        
        self.assertGreater(100, p)
        
if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()