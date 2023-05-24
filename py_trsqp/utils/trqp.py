import numpy as np
import casadi as ca
from typing import Tuple

from .model_manager import ModelManager
from .TR_exceptions import TRQPIncompatible, EndOfAlgorithm

import warnings
warnings.filterwarnings("ignore")

class TRQP():
    def __init__(self, models:ModelManager, ub:list, lb:list, radius:float, solver:str="penalty") -> None:
        
        if solver == "ipopt":
            self.sol, self.radius, self.is_compatible = self.invoke_composite_step(models, ub, lb, radius)
        elif solver == "penalty":
            self.sol, self.radius, self.is_compatible = self.invoke_penalty_step(models, ub, lb, radius)
        
    def invoke_penalty_step(self, models:ModelManager, ub:float, lb:float, radius:float):
        from scipy.optimize import minimize, Bounds
        
        data = models.m_cf.model.y
        center = data[:,0]
         
        input_symbols = models.input_symbols

        # Cost function
        cf = models.m_cf.model.model_polynomial.symbol
        
        phi = cf + 0
        
        xk = center
        for k in range(100):
            mu = 1.0
            # Build ultimate objective function
            ## equality constraints
            for m in models.m_eqcs.models:
                phi = phi + mu*(m.model.model_polynomial.symbol**2)
                
            ## inequality constraints
            for m in models.m_ineqcs.models:
                phi = phi + mu*(ca.fmax(0, -m.model.model_polynomial.symbol))**2
                
            phif = ca.Function('phif', [input_symbols], [phi])
            
            # tr radius as input bound      
            ubx = np.min((center + radius, ub), axis=0) 
            lbx = np.max((center - radius, lb), axis=0)
            lbx[lbx > ubx] = ubx[lbx > ubx]
            
            prob = minimize(phif, xk, method="SLSQP", bounds=Bounds(lb=lbx, ub=ubx))
            
            if not prob.success:
                raise Exception(prob.message)
            
            xprev = xk
            xk = prob.x
            
            if np.abs(phif(xprev) - phif(xk)) < 1E-5:
                sol = xk
                break
            
            # update penalty
            mu = mu*1.2
            
            if k == 99:
                print("Not solved!!")
                raise Exception("Not solved")
            
        return ca.DM(sol), radius, True

    def invoke_penalty_restoration_step(self, models:ModelManager, ub:float, lb:float, radius:float):
        input_symbols = models.input_symbols
        data = models.m_cf.model.y
        center = data[:,0]
            
        # tr radius as input bound      
        ubx = np.min((center + radius, ub), axis=0) 
        lbx = np.max((center - radius, lb), axis=0)
        lbx[lbx > ubx] = ubx[lbx > ubx]
        
        nlp = {
            'x': input_symbols,
            'f': models.m_viol.symbol
        }
        
        opts = {'ipopt.print_level':0, 'print_time':0, 'ipopt.sb': 'yes'}
        
        solver = ca.nlpsol('TRQP_restoration', 'ipopt', nlp, opts)
        sol = solver(x0=center+(radius/100), ubx=ubx, lbx=lbx)
        if solver.stats()['success']:
            pass
        else:
            # raise EndOfAlgorithm(f"Impossible to compute restoration step. current iterate: {center}. 'best solution' = {sol['x']}")
            print(f"Next solution point might not be totally feasible")
            pass
            
        return sol, radius

    def invoke_composite_step(self, models:ModelManager, ub:list, lb:list, radius:float) -> Tuple[np.ndarray, float, bool]:
        ## construct TQRP problem (page 722, Chapter 15: Sequential Quadratic Programming)
        data = models.m_cf.model.y
        center = data[:,0]
         
        input_symbols = models.input_symbols

        # Cost function
        cf = models.m_cf.model.model_polynomial.symbol
        
        ubg = []
        lbg = []
        
        g = []
        
        # Equality constraints
        if len(models.m_eqcs.models) == 0:
            pass
        else:
            eqcs = ca.vertcat(*[m.model.model_polynomial.symbol for m in models.m_eqcs.models])
            jc_eqcs = ca.jacobian(eqcs, input_symbols)
            eqcs_c = ca.Function('c_E', [input_symbols], [eqcs]) # equality constraint at center
            jc_eqcs_c = ca.Function('A_E', [input_symbols], [jc_eqcs]) # jacobian of equality constraint at center
            
            g_eq = ca.simplify(eqcs_c(center) + ca.mtimes(jc_eqcs_c(center), input_symbols - center))
            g.append(g_eq)
            for _ in range(len(models.m_eqcs.models)):
                ubg.append(0.)
                lbg.append(0.)
      
        # Inequality constraints
        if len(models.m_ineqcs.models) == 0:
            pass
        else:
            ineqcs = ca.vertcat(*[m.model.model_polynomial.symbol for m in models.m_ineqcs.models])
            jc_ineqcs = ca.jacobian(ineqcs, input_symbols)
            ineqcs_c = ca.Function('c_I', [input_symbols], [ineqcs]) # inequality constraint at center
            jc_ineqcs_c = ca.Function('A_I', [input_symbols], [jc_ineqcs]) # jacobian of inequality constraint at center
        
            g_ineq = ca.simplify(ineqcs_c(center) + ca.mtimes(jc_ineqcs_c(center), input_symbols - center))
            g.append(g_ineq)
            for _ in range(len(models.m_ineqcs.models)):
                ubg.append(ca.inf)
                lbg.append(0.)
          
        # tr radius as input bound      
        ubx = np.min((center + radius, ub), axis=0) 
        lbx = np.max((center - radius, lb), axis=0)
        lbx[lbx > ubx] = ubx[lbx > ubx]
        
        # construct NLP problem
        nlp = {
            'x': input_symbols,
            'f': cf, 
            'g': ca.vertcat(*g)
        }

        # opts = {"error_on_fail": True, "verbose": True}
        opts = {'ipopt.print_level':0, 'print_time':0, 'ipopt.sb': 'yes'}
        
        # solve TRQP problem
        solver = ca.nlpsol('TRQP_composite', 'ipopt', nlp, opts)
        sol = solver(x0=center+(radius/1000), ubx=ubx, lbx=lbx, ubg=ubg, lbg=lbg)

        is_compatible = True
        try:
            if not solver.stats()['success']:
                print(f"fail with 1/1000 perturbation as initial point")
                sol = solver(x0=center+(radius/100), ubx=ubx, lbx=lbx, ubg=ubg, lbg=lbg)
                if not solver.stats()['success']:
                    raise TRQPIncompatible(f"TRQP is incompatible. Invoke restoration step")
        except TRQPIncompatible:
            sol, radius = self.invoke_restoration_step(models, ub, lb, radius)
            is_compatible = False

        return sol['x'], radius, is_compatible

    def invoke_restoration_step(self, models:ModelManager, ub:list, lb:list, radius:float):
        
        print(f"Invoke restoration step")
        
        input_symbols = models.input_symbols
        data = models.m_cf.model.y
        center = data[:,0]
            
        # tr radius as input bound      
        ubx = np.min((center + radius, ub), axis=0) 
        lbx = np.max((center - radius, lb), axis=0)
        lbx[lbx > ubx] = ubx[lbx > ubx]
        
            
        
        nlp = {
            'x': input_symbols,
            'f': models.m_viol.symbol
        }
        
        opts = {'ipopt.print_level':0, 'print_time':0, 'ipopt.sb': 'yes'}
        
        solver = ca.nlpsol('TRQP_restoration', 'ipopt', nlp, opts)
        sol = solver(x0=center+(radius/100), ubx=ubx, lbx=lbx)
        if solver.stats()['success']:
            pass
        else:
            # raise EndOfAlgorithm(f"Impossible to compute restoration step. current iterate: {center}. 'best solution' = {sol['x']}")
            print(f"Next solution point might not be totally feasible")
            pass
            
        return sol, radius