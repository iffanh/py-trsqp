import py_trsqp.trsqp as tq
import numpy as np
import unittest

def cf(x):
    return 10*(x[0]-1.0)**2 + 4*(x[1]-1.0)**2

def ineq(x):
    return x[0] - x[1] - 1

def eq(x):
    return x[0]**2 - x[1] - 1

class TRSQPTest(unittest.TestCase):
        
    def test_trsqp_penalty0(self):
        tr = tq.TrustRegionSQPFilter(x0=np.array([-1.5,-1.0]).T, 
                                    k=6,
                                    cf=cf, 
                                    eqcs=[], 
                                    ineqcs=[],
                                    opts={'solver': 'penalty'})
        tr.optimize(max_iter=100) 
        
        self.assertAlmostEqual(tr.iterates[-1]['y_curr'][0], 1.0)
        self.assertAlmostEqual(tr.iterates[-1]['y_curr'][1], 1.0)