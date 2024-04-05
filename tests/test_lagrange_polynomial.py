import unittest

import numpy as np
import casadi as ca

import py_trsqp.utils.lagrange_polynomial as lp

INPUT_SYMBOLS = ca.SX.sym('x', 2)
Y = np.array([[0.0, 0.0],
            [-0.98, -0.96], 
            [-0.96, -0.98],
            [0.98, 0.96], 
            [0.96, 0.98], 
            [0.94, 0.94]]).T

def cf(x):
    return x[0]**2 + x[1]**2 + x[0]*x[1]*1

F = cf(Y)

YGOOD = np.array([[0.0, 0.0],
                [-0.967, 0.254], 
                [-0.96, -0.98],
                [0.98, 0.96], 
                [-0.199, 0.979], 
                [0.707, -0.707]]).T

FGOOD = cf(YGOOD)

YLARGE = np.random.random(size=(15, 45))

class LagrangePolynomialsTest(unittest.TestCase):
    
    # with function value
    l = lp.LagrangePolynomials(input_symbols=INPUT_SYMBOLS, 
                                   pdegree=2)
    l.initialize(y=Y, f=F, sort_type='function')
        
    def test_NP(self):
        self.assertEqual(self.l.N, Y.shape[0])
        self.assertEqual(self.l.P, Y.shape[1])
        
    def test_radius(self):
        self.assertGreaterEqual(np.sqrt(2), self.l.tr_radius)
     
    def test_indices_and_coefficients(self):
        # degree 1
        l = lp.LagrangePolynomials(input_symbols=INPUT_SYMBOLS, 
                                    pdegree=1)
        l.initialize(y=Y, f=F, sort_type='function')
    
        self.assertListEqual(l.multiindices, [(0,0), (0,1), (1,0)])
        self.assertListEqual(l.coefficients, [1.0, 1.0, 1.0])
    
        # degree 2
        l = lp.LagrangePolynomials(input_symbols=INPUT_SYMBOLS, 
                                    pdegree=2)
        l.initialize(y=Y, f=F, sort_type='function')
    
        self.assertListEqual(l.multiindices, [(0,0), (0,1), (1,0), (0,2), (1,1), (2,0)])
        self.assertListEqual(l.coefficients, [1.0, 1.0, 1.0, 2.0, 1, 2.0])
        
        # large dataset. This has to be computed fast!
        input_symbols_large = ca.SX.sym('x', YLARGE.shape[0])
        l = lp.LagrangePolynomials(input_symbols=input_symbols_large, 
                                    pdegree=2)
        l.initialize(y=YLARGE, f=None, sort_type='function')
    
    def test_polynomial_basis(self):
        
        basis = self.l.polynomial_basis
        for base, val in zip(basis, [1.0, 1.0, 1.0, 0.5, 1.0, 0.5]):
            self.assertEqual(base.feval((1.0,1.0)), val)
        
    def test_model_polynomial(self):
        l = lp.LagrangePolynomials(input_symbols=INPUT_SYMBOLS, 
                                   pdegree=2)
        l.initialize(y=YGOOD, f=FGOOD, sort_type='function')
        
        # model = l.model_polynomial_normalized
        for y, feval in zip(YGOOD.T, FGOOD):
            self.assertAlmostEqual(l.interpolate(y), feval)        
        
if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()