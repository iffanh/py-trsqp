"""
An attempt to make a benchmark studies using the well known CUTEst problem set.
To be compared with the simple Newton's method. 
"""

import numpy as np
import pycutest
from py_trsqp.trsqp import TrustRegionSQPFilter
from py_trsqp.test.benchmark_utils import (tprint, 
                                          save_json, 
                                          get_constants, 
                                          get_bounds, 
                                          get_constraints, 
                                          get_constraints_trsqp, 
                                          run_optimizer, 
                                          is_feasible_trqp, 
                                          write_summary_to_txt)

OUTPUT_FILE='output_linear.txt'
OUTPUT_JSON='output_linear.json'
CONSTANTS = get_constants()

def run(n_max=5, n_cons_max=10):
    # constraint
    tprint(f"", filename=OUTPUT_FILE, restart=True)
    status = dict()
    for problem_name in pycutest.find_problems(constraints="linear"):
        try:
            p = pycutest.import_problem(problem_name)
        except AttributeError:
            continue
        
        p = pycutest.import_problem(problem_name)
        
        c = p.cons(p.x0)
        n_cons = c.shape[0]
        
        if p.x0.shape[0] > n_max: 
            continue
        
        if n_cons > n_cons_max:
            continue
        
        status[problem_name] = dict()
        
        bounds = get_bounds(p)
        consts = get_constraints(p)
        status[problem_name]['scipy-slsqp'] = run_optimizer(p, consts=consts, bounds=bounds, method='SLSQP')
        status[problem_name]['scipy-trust'] = run_optimizer(p, consts=consts, bounds=bounds, method='trust-constr')
            
        ## Build Constraints for COBYLA
        consts = get_constraints(p, ineq_to_eq=True)    
        status[problem_name]['scipy-cobyla'] = run_optimizer(p, consts=consts, bounds=bounds, method='COBYLA')
        
        # Constraints for trsqp
        eqcs, ineqcs = get_constraints_trsqp(p)
        dim = p.x0.shape[0]
        nsample = int((dim+1)*(dim+2)/2)
        tr = TrustRegionSQPFilter(x0=p.x0, 
                                k=nsample,
                                cf=p.obj,
                                eqcs=eqcs,
                                ineqcs=ineqcs, 
                                ub=list(p.bu),
                                lb=list(p.bl), 
                                opts={'solver': "ipopt"}, 
                                constants=CONSTANTS)
        tr.optimize(max_iter=50)
        
        
        status[problem_name]["trsqp"] = dict()
        status[problem_name]["trsqp"]['x'] = dict()
        for i, _x in enumerate(tr.iterates[-1]['y_curr']):
            status[problem_name]["trsqp"]['x'][i] = _x
        status[problem_name]["trsqp"]['f'] = tr.iterates[-1]['fY'][0]
        status[problem_name]["trsqp"]['is_feasible'] = is_feasible_trqp(eqcs=eqcs, ineqcs=ineqcs, sol=tr.iterates[-1]['y_curr'])
        status[problem_name]["trsqp"]['nfev'] = tr.iterates[-1]['total_number_of_function_calls']
        
    write_summary_to_txt(filename=OUTPUT_FILE, status=status, title=f"Function evaluation", key='f')
    write_summary_to_txt(filename=OUTPUT_FILE, status=status, title=f"Number of function evaluation", key='nfev')
    save_json(status, filename=OUTPUT_JSON)
    
    return status
    
if __name__ == '__main__':
    run()