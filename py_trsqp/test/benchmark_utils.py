import json
import scipy.optimize
import numpy as np
from typing import List

def tprint(txt:str, filename:str, restart=False):
    
    if restart: 
        with open(filename, 'w') as f:
            f.write(txt)
            f.write('\n')
    else:    
        with open(filename, 'a') as f:
            f.write(txt)
            f.write('\n')
            
def save_json(data, filename):
    with open(filename, 'w') as f:        
        json.dump(data, f, indent=4)
        
def get_constants():
    constants = dict()
    constants["gamma_0"] = 0.2
    constants["gamma_1"] = 0.5
    constants["gamma_2"] = 2.0 #1.5 Eq
    constants["eta_1"] = 0.001
    constants["eta_2"] = 0.1
    constants["mu"] = 0.01
    constants["gamma_vartheta"] = 1E-8 #1E-4 
    constants["kappa_vartheta"] = 1E-4
    constants["kappa_radius"] = 0.8
    constants["kappa_mu"] = 10
    constants["kappa_tmd"] = 0.01

    constants["init_radius"] = 1.0
    constants["stopping_radius"] = 1E-7
    constants["L_threshold"] = 1.000
    return constants

## Build Bounds
def get_bounds(p):
    bounds = []
    for l, u in zip(p.bl, p.bu):
        bounds.append((l, u))
        
    return bounds
        
## Build Constraints
def get_constraints(p, ineq_to_eq=False):
    eq_cons_flags = p.is_eq_cons
    if eq_cons_flags is None:
        return []
    
    consts = []
    if not ineq_to_eq:
        for i, flag in enumerate(eq_cons_flags):
            const = {}
            const['fun'] = lambda x, i=i: p.cons(x, index=i)

            if flag:
                const['type'] = "eq"
            else:
                const['type'] = "ineq"
            
            consts.append(const)    
    else:
        for i, flag in enumerate(eq_cons_flags):
            if flag:
                const = {}
                const['fun'] = lambda x, i=i: p.cons(x, index=i)
                const['type'] = "ineq"
                consts.append(const) 
                
                const = {}
                const['fun'] = lambda x, i=i: -p.cons(x, index=i)
                const['type'] = "ineq"
                consts.append(const) 
            else:
                const = {}
                const['fun'] = lambda x, i=i: p.cons(x, index=i)
                const['type'] = "ineq"
                consts.append(const)    
    
    return consts

def get_constraints_trsqp(p):
    eqcs = []
    ineqcs = []
    eq_cons_flags = p.is_eq_cons
    for i, flag in enumerate(eq_cons_flags):
        conf = lambda x, i=i: p.cons(x, index=i)
        if flag:
            eqcs.append(conf)
        else:
            ineqcs.append(conf)
    return eqcs, ineqcs

def run_optimizer(p, consts, bounds, method) -> dict:
    data = dict()
    try:
        res = scipy.optimize.minimize(fun=p.obj, method=method, x0=p.x0, constraints=consts, bounds=bounds)
        data['x'] = dict()  
        for i, _x in enumerate(res.x):
            data['x'][i] = _x
        data['f'] = res.fun
        data['nfev'] = res.nfev
        data['is_feasible'] = is_feasible(consts, res.x)
    except:
        data['x'] = None
        data['f'] = None
        data['nfev'] = None
        data['is_feasible'] = None
        
    return data

def is_feasible(consts:List[dict], sol:np.ndarray, tol=1E-8) -> bool:
    for const in consts:
        if const['type'] == "eq":
            if np.abs(const['fun'](sol)) > tol:
                return False
        elif const['type'] == "ineq":
            if const['fun'](sol) > tol:
                return False
    return True

def is_feasible_trqp(eqcs:List[callable], ineqcs:List[callable], sol, tol=1E-8) -> bool:
    for eqc in eqcs:
        if np.abs(eqc(sol)) > tol:
            return False 
    for ineqc in ineqcs:
        if ineqc(sol) > tol:
            return False
    return True

def write_summary_to_txt(filename:str, status:dict, title:str, key:str, _format=callable):
    tprint(f"{title}", filename=filename)
    
    for problem_name in status.keys():
        txt = f"| PROBLEM \t\t"
        for solver_type in status[problem_name].keys():
            txt = txt + f"| {solver_type} \t\t"
        txt = txt + f"|"
        
        break
        
    for problem_name in status.keys():
        txt = f"| {problem_name} \t\t"
        for solver_type in status[problem_name].keys():
            try:
                if _format == float:
                    txt = txt + f"| {_format(status[problem_name][solver_type][key]):5e} \t\t"
                else:
                    txt = txt + f"| {_format(status[problem_name][solver_type][key])} \t\t"
            except TypeError:
                txt = txt + f"| {status[problem_name][solver_type][key]} \t\t"
        txt = txt + f"|"
        tprint(txt, filename=filename)