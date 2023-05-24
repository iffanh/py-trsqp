import py_trsqp.trsqp as tq
import numpy as np
import unittest
from py_trsqp.utils.TR_exceptions import IncorrectInputException

def simple(x):
    return 10*(x[0]-1.0)**2 + 4*(x[1]-1.0)**2

def rosen(x:np.ndarray) -> np.ndarray: # Rosenbrock function: OF
    return 100*(x[1]-x[0]**2)**2+((x[0]-1)**2)

def eq1(x):
    return x[0] - x[1] - 1

CONSTANTS = {}
CONSTANTS["L_threshold"] = 1.005
CONSTANTS["eta_1"] = 0.0001
CONSTANTS["eta_2"] = 0.1
CONSTANTS["gamma_1"] = 0.8
CONSTANTS["gamma_2"] = 2.0
class TRSQPTest(unittest.TestCase):
    
    def test_trsqp_exceptions(self):
        # len(ub) != len(x0)
        with self.assertRaises(IncorrectInputException):
            tr = tq.TrustRegionSQPFilter(x0=[-2.5,-1.0], 
                                        k=6,
                                        cf=simple,
                                        ub=[3], 
                                        lb=[-1])
            
        # len(ub) != len(lb)
        with self.assertRaises(IncorrectInputException):
            tr = tq.TrustRegionSQPFilter(x0=[-2.5,-1.0], 
                                        k=6,
                                        cf=simple,
                                        ub=[3], 
                                        lb=[-1, 1])
            
        # type(ub) != type(lb)
        with self.assertRaises(IncorrectInputException):
            tr = tq.TrustRegionSQPFilter(x0=[-2.5,-1.0], 
                                        k=6,
                                        cf=simple,
                                        ub=3, 
                                        lb=[-1])
        
    def test_trsqp_simple(self):
        tr = tq.TrustRegionSQPFilter(x0=[-2.5,-1.0], 
                                    k=6,
                                    cf=simple,
                                    ub=3.0,
                                    lb=-3.0,
                                    eqcs=[], 
                                    ineqcs=[],
                                    opts={'solver': 'ipopt'}, 
                                    constants=CONSTANTS)
        tr.optimize(max_iter=100)
        
        self.assertAlmostEqual(tr.iterates[-1]['y_curr'][0], 1.0, places=4)
        self.assertAlmostEqual(tr.iterates[-1]['y_curr'][1], 1.0, places=4)
        
    def test_trsqp_simple1(self):
        tr = tq.TrustRegionSQPFilter(x0=[-2.5,-1.0], 
                                    k=6,
                                    cf=simple, 
                                    ub=3.0,
                                    lb=-3.0,
                                    eqcs=[eq1], 
                                    ineqcs=[],
                                    opts={'solver': 'ipopt'}, 
                                    constants=CONSTANTS)
        tr.optimize(max_iter=100)
        
        sol0 = 36/28 #analytical solution
        sol1 = 8/28
        
        self.assertAlmostEqual(tr.iterates[-1]['y_curr'][0], sol0, places=4)
        self.assertAlmostEqual(tr.iterates[-1]['y_curr'][1], sol1, places=4)
        
    def test_trsqp_rosen(self):
        tr = tq.TrustRegionSQPFilter(x0=[-2.5,-1.0], 
                                    k=6,
                                    cf=rosen, 
                                    ub=4.0,
                                    lb=-4.0,
                                    eqcs=[], 
                                    ineqcs=[],
                                    opts={'solver': 'ipopt'}, 
                                    constants=CONSTANTS)
        tr.optimize(max_iter=100)
        
        self.assertAlmostEqual(tr.iterates[-1]['y_curr'][0], 1.0, places=4)
        self.assertAlmostEqual(tr.iterates[-1]['y_curr'][1], 1.0, places=4)
              
if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()