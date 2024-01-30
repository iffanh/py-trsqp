import py_trsqp.trsqp as tq
import numpy as np
import unittest
from py_trsqp.utils.TR_exceptions import IncorrectInputException

def simple_but_many(x):
    return 10*(x[0]-1.0)**2 + 4*(x[1]-1.0)**2 + \
           10*(x[2]-1.0)**2 + 4*(x[3]-1.0)**2 + \
           10*(x[4]-1.0)**2 + 4*(x[5]-1.0)**2 + \
           10*(x[6]-1.0)**2 + 4*(x[7]-1.0)**2 + \
           10*(x[8]-1.0)**2 + 4*(x[9]-1.0)**2

def simple(x):
    return 10*(x[0]-1.0)**2 + 4*(x[1]-1.0)**2

def rosen(x:np.ndarray) -> np.ndarray: # Rosenbrock function: OF
    return 100*(x[1]-x[0]**2)**2+((x[0]-1)**2)


def disc(x:np.ndarray) -> np.ndarray:
    return 2 - (x[0]**2 + x[1]**2) 

def eq1(x):
    return x[0] - x[1] - 1

def ineq1(x):
    return x[0] + x[1]**2

def ineq2(x):
    return x[0]**2 + x[1]

def ineq3(x):
    return x[0]**2 + x[1]**2 - 1

def ackley(x:np.ndarray) -> np.ndarray: 
    return -20*np.exp(-0.2*np.sqrt(0.5*(x[0]**2 + x[1]**2))) - np.exp(0.5*(np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]))) + np.e + 20

def mccormick(x):
    return np.sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5*x[0] + 2.5*x[1] + 1

CONSTANTS = {}
CONSTANTS["L_threshold"] = 1.0
CONSTANTS["kappa_vartheta"] = 0.1
CONSTANTS["eta_1"] = 0.1
CONSTANTS["eta_2"] = 0.4
CONSTANTS["gamma_0"] = 0.3
CONSTANTS["gamma_1"] = 0.7
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
    
    def test_trsqp_simple_max_points(self):
        print("======== SIMPLE MAX POINTS 1 ========")
        tr = tq.TrustRegionSQPFilter(x0=[-2.5,-2.0], 
                                    k=6,
                                    cf=simple,
                                    ub=3.0,
                                    lb=-3.0,
                                    eqcs=[], 
                                    ineqcs=[],
                                    opts={'solver': 'ipopt', 'max_points': 2}, 
                                    constants=CONSTANTS)
        tr.optimize(max_iter=10)
        print(f"Total number of feval = {tr.iterates[-1]['total_number_of_function_calls']}")
        
        for t in tr.iterates:
            self.assertLessEqual(t['number_of_function_calls'], 3)
            
        print("======== SIMPLE MAX POINTS 2 ========")
        tr = tq.TrustRegionSQPFilter(x0=[-2.5,-2.0], 
                                    k=6,
                                    cf=simple,
                                    ub=3.0,
                                    lb=-3.0,
                                    eqcs=[], 
                                    ineqcs=[],
                                    opts={'solver': 'ipopt', 'max_points': 7}, 
                                    constants=CONSTANTS)
        tr.optimize(max_iter=10)
        print(f"Total number of feval = {tr.iterates[-1]['total_number_of_function_calls']}")
        
        for t in tr.iterates:
            self.assertLessEqual(t['number_of_function_calls'], 6)
        
    def test_trsqp_simple_bound(self):
        print("======== SIMPLE BUDGET with BOUND ========")
        tr = tq.TrustRegionSQPFilter(x0=[-2.5,-2.0], 
                                    k=6,
                                    cf=simple,
                                    ub=-1.9,
                                    lb=-5.0,
                                    eqcs=[], 
                                    ineqcs=[],
                                    opts={'solver': 'ipopt', 'budget': 100}, 
                                    constants=CONSTANTS)
        tr.optimize(max_iter=50)
        # print(f"Total number of feval = {tr.iterates[-1]['total_number_of_function_calls']}")
        self.assertLessEqual(tr.iterates[-1]['y_curr'][0], -1.9)
        self.assertLessEqual(tr.iterates[-1]['y_curr'][1], -1.9)
        
    def test_trsqp_simple_budget(self):
        print("======== SIMPLE BUDGET ========")
        tr = tq.TrustRegionSQPFilter(x0=[-2.5,-2.0], 
                                    k=6,
                                    cf=simple,
                                    ub=3.0,
                                    lb=-3.0,
                                    eqcs=[], 
                                    ineqcs=[],
                                    opts={'solver': 'ipopt', 'budget': 1}, 
                                    constants=CONSTANTS)
        tr.optimize(max_iter=50)
        print(f"Total number of feval = {tr.iterates[-1]['total_number_of_function_calls']}")
        
        self.assertEqual(tr.termination_status, 'Budget Exceeded')
            
    def test_trsqp_simple(self):
        print("======== SIMPLE ========")
        tr = tq.TrustRegionSQPFilter(x0=[-2.5,-2.0], 
                                    k=6,
                                    cf=simple,
                                    ub=3.0,
                                    lb=-3.0,
                                    eqcs=[], 
                                    ineqcs=[],
                                    opts={'solver': 'ipopt'}, 
                                    constants=CONSTANTS)
        tr.optimize(max_iter=50)
        print(f"Total number of feval = {tr.iterates[-1]['total_number_of_function_calls']}")
        
        self.assertAlmostEqual(tr.iterates[-1]['y_curr'][0], 1.0, places=4)
        self.assertAlmostEqual(tr.iterates[-1]['y_curr'][1], 1.0, places=4)
        
    def test_trsqp_simple1(self):
        print("======== SIMPLE w CONSTRAINT ========")
        
        tr = tq.TrustRegionSQPFilter(x0=[-2.5,-2.0], 
                                    k=6,
                                    cf=simple, 
                                    ub=3.0,
                                    lb=-3.0,
                                    eqcs=[eq1], 
                                    ineqcs=[],
                                    opts={'solver': 'ipopt'}, 
                                    constants=CONSTANTS)
        tr.optimize(max_iter=50)
        print(f"Total number of feval = {tr.iterates[-1]['total_number_of_function_calls']}")
        
        sol0 = 36/28 #analytical solution
        sol1 = 8/28
        
        self.assertAlmostEqual(tr.iterates[-1]['y_curr'][0], sol0, places=3)
        self.assertAlmostEqual(tr.iterates[-1]['y_curr'][1], sol1, places=3)
         
    def test_trsqp_rosen(self):
        print("======== ROSENBROCK ========")
        CONSTANTS = {}
        CONSTANTS["L_threshold"] = 1.000
        CONSTANTS["eta_1"] = 1E-8
        CONSTANTS["eta_2"] = 1E-7
        CONSTANTS["gamma_0"] = 0.5
        CONSTANTS["gamma_1"] = 0.7
        CONSTANTS["gamma_2"] = 1.5
        CONSTANTS["stopping_radius"] = 1E-12
        tr = tq.TrustRegionSQPFilter(x0=[-2.5,-2.0], 
                                    k=6,
                                    cf=rosen, 
                                    ub=6.0,
                                    lb=-6.0,
                                    eqcs=[], 
                                    ineqcs=[],
                                    opts={'solver': 'ipopt'}, 
                                    constants=CONSTANTS)
        tr.optimize(max_iter=1000)
        print(f"Total number of feval = {tr.iterates[-1]['total_number_of_function_calls']}")
        
        self.assertAlmostEqual(tr.iterates[-1]['y_curr'][0], 1.0, places=2)
        self.assertAlmostEqual(tr.iterates[-1]['y_curr'][1], 1.0, places=2)
        
    def test_trsqp_rosen_with_disc(self):
        print("======== ROSENBROCK w DISC ========")
        CONSTANTS = {}
        CONSTANTS["L_threshold"] = 1.000
        CONSTANTS["eta_1"] = 1E-8
        CONSTANTS["eta_2"] = 1E-7
        CONSTANTS["gamma_0"] = 0.6
        CONSTANTS["gamma_1"] = 0.8
        CONSTANTS["gamma_2"] = 1.5
        CONSTANTS["stopping_radius"] = 1E-12
        tr = tq.TrustRegionSQPFilter(x0=[-2.5,-2.0], 
                                    k=6,
                                    cf=rosen, 
                                    ub=6.0,
                                    lb=-6.0,
                                    eqcs=[], 
                                    ineqcs=[disc],
                                    opts={'solver': 'ipopt'}, 
                                    constants=CONSTANTS)
        tr.optimize(max_iter=1000)
        print(f"Total number of feval = {tr.iterates[-1]['total_number_of_function_calls']}")
        
        self.assertAlmostEqual(tr.iterates[-1]['y_curr'][0], 1.0, places=2)
        self.assertAlmostEqual(tr.iterates[-1]['y_curr'][1], 1.0, places=2)
        
    def test_trsqp_rosen_with_ineq(self):
        print("======== ROSENBROCK w INEQ ========")
        CONSTANTS = {}
        CONSTANTS["L_threshold"] = 1.000
        CONSTANTS["eta_1"] = 1E-8
        CONSTANTS["eta_2"] = 0.2
        CONSTANTS["gamma_0"] = 0.5
        CONSTANTS["gamma_1"] = 0.7
        CONSTANTS["gamma_2"] = 1.5
        CONSTANTS["init_radius"] = 0.5
        CONSTANTS["stopping_radius"] = 1E-10
        tr = tq.TrustRegionSQPFilter(x0=[0.0,0.0], #x0=[-2.,1.0], 
                                    k=6,
                                    cf=rosen, 
                                    ub=[0.5, 1000],
                                    lb=[-0.5, -1000],
                                    eqcs=[], 
                                    ineqcs=[ineq1, ineq2, ineq3],
                                    opts={'solver': 'ipopt'}, 
                                    constants=CONSTANTS)
        tr.optimize(max_iter=1000)
        print(f"Total number of feval = {tr.iterates[-1]['total_number_of_function_calls']}")

        self.assertAlmostEqual(tr.iterates[-1]['y_curr'][0], 0.5, places=2)
        self.assertAlmostEqual(tr.iterates[-1]['y_curr'][1], 0.5*np.sqrt(3), places=2)
        
    def test_trsqp_ackley(self):
        print("======== ACKLEY ========")
        CONSTANTS = {}
        CONSTANTS["L_threshold"] = 1.000
        CONSTANTS["eta_1"] = 0.1
        CONSTANTS["eta_2"] = 0.2
        CONSTANTS["gamma_0"] = 0.5
        CONSTANTS["gamma_1"] = 0.7
        CONSTANTS["gamma_2"] = 1.5
        CONSTANTS["stopping_radius"] = 1E-12
        tr = tq.TrustRegionSQPFilter(x0=[-2.5,2.0], 
                                    k=6,
                                    cf=ackley, 
                                    ub=5.0,
                                    lb=-5.0,
                                    eqcs=[], 
                                    ineqcs=[],
                                    opts={'solver': 'ipopt'}, 
                                    constants=CONSTANTS)
        tr.optimize(max_iter=1000)
        print(f"Total number of feval = {tr.iterates[-1]['total_number_of_function_calls']}")
        
        self.assertAlmostEqual(tr.iterates[-1]['y_curr'][0], 0.0, places=3)
        self.assertAlmostEqual(tr.iterates[-1]['y_curr'][1], 0.0, places=3)
        
    def test_trsqp_mccormick(self):
        print("======== MCCORMICK ========")
        CONSTANTS = {}
        CONSTANTS["L_threshold"] = 1.000
        CONSTANTS["eta_1"] = 0.001
        CONSTANTS["eta_2"] = 0.2
        CONSTANTS["gamma_0"] = 0.5
        CONSTANTS["gamma_1"] = 0.7
        CONSTANTS["gamma_2"] = 1.5
        CONSTANTS["init_radius"] = 1.0
        CONSTANTS["stopping_radius"] = 1E-12
        tr = tq.TrustRegionSQPFilter(x0=[0.0,0.0], #x0=[1.0,1.0], 
                                    k=6,
                                    cf=mccormick, 
                                    ub=[4, 4],
                                    lb=[-1.5, -3],
                                    eqcs=[], 
                                    ineqcs=[],
                                    opts={'solver': 'ipopt'}, 
                                    constants=CONSTANTS)
        tr.optimize(max_iter=1000)
        print(f"Total number of feval = {tr.iterates[-1]['total_number_of_function_calls']}")
        
        self.assertAlmostEqual(tr.iterates[-1]['y_curr'][0], -0.54719, places=3)
        self.assertAlmostEqual(tr.iterates[-1]['y_curr'][1], -1.54719, places=3)
        self.assertAlmostEqual(tr.iterates[-1]['fY'][0], -1.9133, places=3)
    
    # ## HEAVY TEST
    # def test_trsqp_simple_but_many(self):
    #     print("======== HEAVY: SIMPLE BUT MANY ========")
        
    #     tr = tq.TrustRegionSQPFilter(x0=[-2.0]*10,
    #                                 k = 11,
    #                                 cf=simple_but_many, 
    #                                 ub=3.0,
    #                                 lb=-3.0,
    #                                 eqcs=[], 
    #                                 ineqcs=[],
    #                                 opts={'solver': 'ipopt', 
    #                                       'max_points': 21}, 
    #                                 constants=CONSTANTS)
    #     tr.optimize(max_iter=500)
    #     print(f"Total number of feval = {tr.iterates[-1]['total_number_of_function_calls']}")
        
    #     sol = 1.000
    #     for i in range(10):
    #         self.assertAlmostEqual(tr.iterates[-1]['y_curr'][i], sol, places=2)
              
if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()