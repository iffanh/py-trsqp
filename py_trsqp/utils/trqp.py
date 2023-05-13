import numpy as np
import casadi as ca
from typing import Tuple

from .model_manager import ModelManager
from .TR_exceptions import TRQPIncompatible, EndOfAlgorithm

import warnings
warnings.filterwarnings("ignore")

class TRQP():
    def __init__(self, models:ModelManager, radius:float) -> None:
        self.sol, self.radius, self.is_compatible = self.invoke_composite_step(models, radius)

    def invoke_composite_step(self, models:ModelManager, radius:float) -> Tuple[np.ndarray, float, bool]:
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
        ubx = center + radius
        lbx = center - radius

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
                print(f"fail with 1/p perturbation as initial point")
                sol = solver(x0=center+(radius/100), ubx=ubx, lbx=lbx, ubg=ubg, lbg=lbg)
                if not solver.stats()['success']:
                    raise TRQPIncompatible(f"TRQP is incompatible. Invoke restoration step")
        except TRQPIncompatible:
            sol, radius = self.invoke_restoration_step(models, radius)
            is_compatible = False

        return sol['x'], radius, is_compatible

    def invoke_restoration_step(self, models:ModelManager, radius:float):
        
        print(f"Invoke restoration step")
        ubg = [radius]
        lbg = [0]
        
        input_symbols = models.input_symbols
        data = models.m_cf.model.y
        center = data[:,0]
            
        # tr radius as input bound      
        ubx = center + radius
        lbx = center - radius
        
        nlp = {
            'x': input_symbols,
            'f': models.m_viol.symbol
        }
        
        opts = {'ipopt.print_level':0, 'print_time':0, 'ipopt.sb': 'yes'}
        
        solver = ca.nlpsol('TRQP_restoration', 'ipopt', nlp, opts)
        sol = solver(x0=center+(radius/100), ubx=ubx, lbx=lbx, ubg=ubg, lbg=lbg)
        if solver.stats()['success']:
            pass
        else:
            # raise EndOfAlgorithm(f"Impossible to compute restoration step. current iterate: {center}. 'best solution' = {sol['x']}")
            print(f"Next solution point might not be totally feasible")
            pass
            
        return sol, radius