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
                                          run_optimizer, 
                                          write_summary_to_txt)

OUTPUT_FILE='output_unconstrained.txt'
OUTPUT_JSON='output_unconstrained.json'
CONSTANTS = get_constants()

def run(n_max=5):
    tprint(f"", filename=OUTPUT_FILE, restart=True)
    status = dict()
    for problem_name in pycutest.find_problems(constraints="unconstrained"):
        p = pycutest.import_problem(problem_name)

        if p.n > n_max: 
            continue

        x = p.x0
        
        status[problem_name] = dict()
        consts = get_constraints(p)
        bounds = get_bounds(p)
        # scipy optimize
        status[problem_name]['scipy-slsqp'] = run_optimizer(p, consts=consts, bounds=bounds, method='SLSQP')
        status[problem_name]['scipy-trust'] = run_optimizer(p, consts=consts, bounds=bounds, method='trust-constr')
            
        ## Build Constraints for COBYLA
        status[problem_name]['scipy-cobyla'] = run_optimizer(p, consts=consts, bounds=bounds, method='COBYLA')
        
        ## TR test
        dim = x.shape[0]
        # nsample = dim*2 + 1
        nsample = (dim+1)*(dim+2)/2
        
        tr = TrustRegionSQPFilter(x0=p.x0, 
                                k=int(nsample),
                                cf=p.obj,
                                ub=list(p.bu),
                                lb=list(p.bl), 
                                opts={'solver': "ipopt"}, 
                                constants=CONSTANTS)
        tr.optimize(max_iter=100)

        status[problem_name]["trsqp"] = dict()
        status[problem_name]["trsqp"]['x'] = dict()
        for i, _x in enumerate(tr.iterates[-1]['y_curr']):
            status[problem_name]["trsqp"]['x'][i] = _x
        status[problem_name]["trsqp"]['f'] = tr.iterates[-1]['fY'][0]
        status[problem_name]["trsqp"]['nfev'] = tr.iterates[-1]['total_number_of_function_calls']
        status[problem_name]["trsqp"]['is_feasible'] = True
        
    write_summary_to_txt(filename=OUTPUT_FILE, status=status, title=f"Function evaluation", key='f', _format=float)
    write_summary_to_txt(filename=OUTPUT_FILE, status=status, title=f"Number of function evaluation", key='nfev', _format=int)
    save_json(status, filename=OUTPUT_JSON)
    
    return status
    
if __name__ == '__main__':
    run()