import unittest

import numpy as np
import casadi as ca

import py_trsqp.utils.model_improvement_without_feval as mi
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

class ModelImprovementTest(unittest.TestCase):
    l = lp.LagrangePolynomials(input_symbols=INPUT_SYMBOLS, 
                                   pdegree=2)
    l.initialize(y=Y, f=F, sort_type='function')
    m = mi.ModelImprovement(input_symbols=INPUT_SYMBOLS)
    
    rad = 1.0
    center = Y[:,0]
    L = 100.0 #poisedness threshold
    
    def test_improve_model(self):
    
        p_before = self.l.poisedness(rad=self.rad, center=self.center).max_poisedness()
    
        lafter = self.m.improve_model(lpolynomials=self.l, 
                             rad = self.rad, 
                             center= self.center,
                             L=self.L, 
                             max_iter=10)
        
        p_after = lafter.poisedness(rad=self.rad, center=self.center).max_poisedness()
        
        self.assertGreater(self.L, p_after)
        self.assertGreater(p_before, p_after)
        
        pass