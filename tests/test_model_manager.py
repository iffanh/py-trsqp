import unittest

import numpy as np
import casadi as ca

import py_trsqp.utils.model_manager as mm

INPUT_SYMBOLS = ca.SX.sym('x', 2)
Y = np.array([[0.0, 0.0],
            [-0.98, -0.96], 
            [-0.96, -0.98],
            [0.98, 0.96], 
            [0.96, 0.98], 
            [0.94, 0.94]]).T
        
def cf(x):
    return x[0]**2 + x[1]**2 + 1

class SetGeometryTest(unittest.TestCase):
    """
    Test class for SetGeometry 
    """
    def test_poisedness(self):
        # test 2 dimension
        L = 10.0
        sg = mm.SetGeometry(input_symbols=INPUT_SYMBOLS, Y=Y, rad=1.0, L=L)
        sg.improve_geometry()
        
        # test poisedness 
        p = np.max(sg.poisedness())
        self.assertGreaterEqual(L, p)

class CostFunctionModelTest(unittest.TestCase):
    
    def test_cf_model_1(self):
        # Before improving poisedness
        fY = cf(Y)
        cfm1 = mm.CostFunctionModel(input_symbols=INPUT_SYMBOLS, 
                                   Y=Y, 
                                   fY=fY)
        f1 = cfm1.model.interpolate([0.0, 0.0])
        self.assertNotEqual(f1, 1.0) ## Not good enough model
        
    def test_cf_model_2(self):
        # After improving poisedness
        L = 10.0
        sg = mm.SetGeometry(input_symbols=INPUT_SYMBOLS, Y=Y, rad=1.0, L=L)
        sg.improve_geometry()
        improved_model = sg.model
        _Y = improved_model.y
        _fY = cf(_Y)
        cfm2 = mm.CostFunctionModel(input_symbols=INPUT_SYMBOLS, 
                                   Y=_Y, 
                                   fY=_fY)
        f2 = cfm2.model.interpolate([0.0, 0.0])
        self.assertAlmostEqual(f2, 1.0, places=5) ## Good enough model
        

if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()