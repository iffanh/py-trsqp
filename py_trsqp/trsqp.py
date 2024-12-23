import numpy as np
from numpy.linalg import LinAlgError
import casadi as ca
from typing import List, Tuple, Union
from multiprocessing import Pool
import copy
from .utils.TR_exceptions import IncorrectConstantsException, EndOfAlgorithm, RedundantPoint, IncorrectInputException, FailedSimulation, IllPoisedModel, PoisednessIsZeroException
from .utils.simulation_manager import SimulationManager
from .utils.model_manager import SetGeometry, ModelManager, CostFunctionModel, EqualityConstraintModels, InequalityConstraintModels, ViolationModel
from .utils.trqp import TRQP
from .utils.filter import FilterSQP
from .utils.model_improvement_without_feval import generate_uniform_sample_nsphere
from .utils.lagrange_polynomial import LagrangePolynomials

class TrustRegionSQPFilter():
    
    def __init__(self, x0:list, cf:callable, ub:Union[List[float], float]=np.inf, lb:Union[List[float], float]=-np.inf, eqcs:List[callable]=[], ineqcs:List[callable]=[], constants:dict=dict(), opts:dict={'solver': "ipopt", 'budget': 1000}) -> None:
        
        def _check_constants(constants:dict) -> dict:
            
            try:
                tmp = constants["gamma_0"]
            except:
                constants["gamma_0"] = 0.5
                
            try:
                tmp = constants["gamma_1"]
            except:
                constants["gamma_1"] = 0.7
                
            try:
                tmp = constants["gamma_2"]
            except:
                constants["gamma_2"] = 1.2 
            
            try:
                tmp = constants["eta_1"]
            except:
                constants["eta_1"] = 0.1
                
            try:
                tmp = constants["eta_2"]
            except:
                constants["eta_2"] = 0.4
                
            try:
                tmp = constants["mu"]
            except:
                constants["mu"] = 0.01
                
            try:
                tmp = constants["gamma_vartheta"]
            except:   
                constants["gamma_vartheta"] = 1E-8
                
            try:
                tmp = constants["kappa_vartheta"]
            except:
                constants["kappa_vartheta"] = 1E-2
                
            try:
                tmp = constants["kappa_radius"] 
            except:
                constants["kappa_radius"] = 0.8
                
            try:
                tmp = constants["kappa_mu"] 
            except:
                constants["kappa_mu"] = 10
                
            try:
                tmp = constants["kappa_tmd"]
            except:
                constants["kappa_tmd"] = 0.01

            try:
                tmp = constants["init_radius"]
            except:
                constants["init_radius"] = 1.
                
            try:
                tmp = constants["stopping_radius"] 
            except:
                constants["stopping_radius"] = 1E-7
                
            try:
                tmp = constants["max_radius"] 
            except:
                constants["max_radius"] = 3.0
                
            try:
                tmp = constants["L_threshold"] 
            except:
                constants["L_threshold"] = 100.0
            
            # if constants is not None:
            if constants["gamma_0"] <= 0.0:
                raise IncorrectConstantsException(f"gamma_0 has to be larger than 0. Got {constants['gamma_0']}")

            if constants["gamma_1"] <= constants["gamma_0"]:
                raise IncorrectConstantsException(f"gamma_1 must be strictly larger than gamma_0. Got gamma_1 = {constants['gamma_1']} and gamma_0 = {constants['gamma_0']}")

            if constants["gamma_1"] >= 1.0:
                raise IncorrectConstantsException(f"gamma_1 must be strictly less than 1. Got {constants['gamma_1']}")

            if constants["gamma_2"] < 1.0:
                raise IncorrectConstantsException(f"gamma_2 must be larger than or equal to 1. Got {constants['gamma_2']}")

            if constants["eta_1"] <= 0.0:
                raise IncorrectConstantsException(f"eta_1 must be strictly larger than 0. Got {constants['eta_1']}")

            if constants["eta_2"] < constants["eta_1"]:
                raise IncorrectConstantsException(f"eta_2 must be larger than or equal to eta_1. Got eta_1 = {constants['eta_1']} and eta_2 = {constants['eta_2']}")

            if constants["eta_2"] >= 1.0:
                raise IncorrectConstantsException(f"eta_2 must be strictly less than 1. Got {constants['eta_2']}")

            if constants["gamma_vartheta"] <= 0 or constants["gamma_vartheta"] >= 1:
                raise IncorrectConstantsException(f"gamma_vartheta must be between 0 and 1. Got {constants['gamma_vartheta']}") 

            if constants["kappa_vartheta"] <= 0 or constants["kappa_vartheta"] >= 1:
                raise IncorrectConstantsException(f"kappa_vartheta must be between 0 and 1. Got {constants['kappa_vartheta']}")

            if constants["kappa_radius"] <= 0 or constants["kappa_radius"] > 1:
                raise IncorrectConstantsException(f"kappa_radius must be between 0 and 1. Got {constants['kappa_radius']}")

            if constants["kappa_mu"] <= 0:
                raise IncorrectConstantsException(f"kappa_mu must be strictly larger than 0. Got {constants['kappa_mu']}")

            if constants["mu"] <= 0 or constants["mu"] >= 1:
                raise IncorrectConstantsException(f"mu must be between 0 and 1. Got {constants['mu']}")

            if constants["kappa_tmd"] <= 0 or constants["kappa_tmd"] > 1:
                raise IncorrectConstantsException(f"kappa_tmd must be between 0 and 1. Got {constants['kappa_tmd']}")

            if constants["init_radius"] <= 0:
                raise IncorrectConstantsException(f"Initial radius must be strictly positive. Got {constants['init_radius']}")

            return constants

        def _check_constraints(eqcs:List[callable], ineqcs:List[callable]) -> Tuple:
            n_eqcs = len(eqcs)
            n_ineqcs = len(ineqcs)
            
            return n_eqcs, n_ineqcs

        def _check_input(ub, lb, x0) -> Tuple[List, List]:
            # check typing
            if (type(ub) is list) and (type(lb) is list):
                if not (len(ub) == len(lb)):
                    raise IncorrectInputException(f"Upper bound and lower bound have to be the same size. Got len(ub) = {len(ub)} and len(lb) = {len(lb)}")
                    
                if not (len(ub) == len(x0)):
                    raise IncorrectInputException(f"Initial values and bounds have to be the same size. Got len(x0) = {len(x0)} and len(ub) = {len(ub)}")
                    
            elif (type(ub) is float) and (type(lb) is float):
                ub = [ub for i in range(len(x0))]
                lb = [lb for i in range(len(x0))]
            
            else:
                # should not come here
                raise IncorrectInputException(f"Type of ub and lb must be either list or float. Got type(ub) = {type(ub)} and type(lb) = {type(lb)}")
            
            return ub, lb
        
        def _check_opts(opts):
            
            _opts = {}
            try: 
                _opts['solver'] = opts['solver']
            except: 
                _opts['solver'] = 'ipopt'
            
            try:
                _opts['budget'] = opts['budget']
            except: 
                _opts['budget'] = 1000
                
            try:
                _opts['max_points'] = opts['max_points']
                
                max_points = int((len(x0)+1)*(len(x0)+2)/2)
                if opts['max_points'] > max_points:
                    print(f"Too many allowed points. Reduced from {opts['max_points']} to {max_points}")
                    _opts['max_points'] = max_points
                    
                max_points = int((len(x0)+1))
                if opts['max_points'] < max_points:
                    print(f"Too few allowed points. Increased from {opts['max_points']} to {max_points}")
                    _opts['max_points'] = max_points
                    
            except: 
                _opts['max_points'] = int((len(x0)+1)*(len(x0)+2)/2)
                
            return _opts

        
        self.opts = _check_opts(opts)
        self.constants = _check_constants(constants=constants)
        self.n_eqcs, self.n_ineqcs = _check_constraints(eqcs=eqcs, ineqcs=ineqcs)
        self.ub, self.lb = _check_input(ub=ub, lb=lb, x0=x0)
        
        ## Transform input and bounds
        self.n_points = 1
        x0 = np.array(x0)
        # self.xn = x0*1
        self.xn = (np.array(self.ub) + np.array(self.lb))/2
        self.zero_flags = x0 == 0.0
        
        x0 = self.norm(x0)
        self.ubt = self.norm(np.array(self.ub)) # t = tilde, normalized bound
        self.lbt = self.norm(np.array(self.lb)) # t = tilde, normalized bound
        
        # TODO:radius needs to be updated IF it exceeds the bound.    
        rad = constants['init_radius']*1            
        # self.dataset = x0[:, np.newaxis] + rad*generate_uniform_sample_nsphere(k=x0.shape[0]+1, d=x0.shape[0], L=self.constants['L_threshold'])         
        self.dataset = x0[:, np.newaxis] + rad*generate_uniform_sample_nsphere(k=2, d=x0.shape[0], L=self.constants['L_threshold'])
        
        ## Transform functions
        cf = self.transform_functions(cf)
        
        eqcs = [self.transform_functions(eqc) for eqc in eqcs]
        ineqcs = [self.transform_functions(ineqc) for ineqc in ineqcs]
        
        self.sm = SimulationManager(cf, eqcs, ineqcs) # Later this will be refactored for reservoir simulation
        self.input_symbols = ca.SX.sym('x', x0.shape[0])
    
    def transform_functions(self, f:callable):
        return lambda x: f(self.denorm(x))
    
    # def norm(self, x:np.ndarray):
        
    #     xt = []
    #     for i, xi in enumerate(x):
    #         xci = self.xn[i]
    #         xti = (xi - xci)/self.R[i]
    #         xt.append(xti)
            
    #     return np.array(xt)
    
    # def denorm(self, xt:np.ndarray):
        
    #     x = []
    #     for i, xti in enumerate(xt):
    #         xci = self.xn[i]
    #         xi = xti*self.R[i] + xci
    #         x.append(xi)
            
    #     return np.array(x)
    
    def norm(self, x:np.ndarray):
        
        xt = []
        for i, xi in enumerate(x):
            xci = self.xn[i]
            if xi > xci:
                xti = (xi - xci)/(self.ub[i] - xci)
            else:
                xti = (xi - xci)/(xci - self.lb[i])
            xt.append(xti)
            
        return np.array(xt)
    
    def denorm(self, xt:np.ndarray):
        
        x = []
        for i, xti in enumerate(xt):
            xci = self.xn[i]
            if xti > 0:
                xi = xti*(self.ub[i] - xci) + xci
            else:
                xi = xti*(xci - self.lb[i]) + xci
            x.append(xi)
            
        return np.array(x)
    
    def __str__(self) -> str:
        return f"TrustRegionSQPFilter(n_eqcs={self.n_eqcs}, n_ineqcs={self.n_ineqcs})"
    
    def run_simulations(self, Y:np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        
        # run cost function and build the corresponding model
        fY_cf = []
        for i in range(Y.shape[1]):
            fY = self.sm.cf.func(Y[:,i])
            if np.isnan(fY) or np.isinf(fY):
                raise FailedSimulation(f"Failed at x={Y[:,i]}")
            fY_cf.append(fY)
        fY_cf = np.array(fY_cf)
        
        # do the same with equality constraints
        fYs_eq = []
        for eqc in self.sm.eqcs.eqcs:
            
            fYs = []
            for i in range(Y.shape[1]):
                fY = eqc.func(Y[:,i])
                if np.isnan(fY) or np.isinf(fY):
                    raise FailedSimulation(f"Failed at x={Y[:,i]}")
                fYs.append(fY)
            
            fYs = np.array(fYs)
            fYs_eq.append(fYs)

        # do the same with inequality constraints
        fYs_ineq = []
        for ineqc in self.sm.ineqcs.ineqcs:
            fYs = []
            for i in range(Y.shape[1]):
                fY = ineqc.func(Y[:,i])
                if np.isnan(fY) or np.isinf(fY):
                    raise FailedSimulation(f"Failed at x={Y[:,i]}")
                fYs.append(fY)
            
            fYs = np.array(fYs)
            fYs_ineq.append(fYs)
        
        return fY_cf, fYs_eq, fYs_ineq
    
    def calculate_violation(self, Y:np.ndarray, fYs_eq:List[np.ndarray], fYs_ineq:List[np.ndarray]):
        
        TOL = 1E-7
        
        # # create violation function Eq 15.5.3            
        violations = []
        violations_eq = []
        violations_ineq = []
        
        self.v_eq_lists = []
        self.v_ineq_lists = []
        for j in range(Y.shape[1]):
            
            v = 0.0 # getting the maximum constraint from both equality and inequality constraints
            v_eq = 0.0 # getting the maximum value from the equality constraints
            v_eq_list = [] # saving all the violation value for equality constraints
            for i in range(self.sm.eqcs.n_eqcs):
                tmp = fYs_eq[i][j]
                
                if np.abs(tmp) < TOL:
                    tmp = 0.
                
                v_eq_list.append(tmp)
                v = ca.fmax(v, ca.fabs(tmp))
                v_eq = ca.fmax(v_eq, ca.fabs(tmp))
                
            v_ineq = 0.0 # getting the maximum value from the inequality constraints
            v_ineq_list = [] # saving all the violation value for inequality constraints
            for i in range(self.sm.ineqcs.n_ineqcs):
                tmp = fYs_ineq[i][j]
                
                if np.abs(tmp) < TOL:
                    tmp = 0.
                
                v_ineq_list.append(tmp)
                v = ca.fmax(v, ca.fmax(0.0, -tmp))
                v_ineq = ca.fmax(v_ineq, ca.fmax(0.0, -tmp))
            
            violations.append(v)
            violations_eq.append(v_eq)
            violations_ineq.append(v_ineq)    
            
            self.v_eq_lists.append(v_eq_list)
            self.v_ineq_lists.append(v_ineq_list)
            
        violations = np.array(violations)
        
        return violations, violations_eq, violations_ineq
    
    def reorder_samples(self, Y, fY_cf, fYs_eq, fYs_ineq, v, v_eq, v_ineq):
        
        ## Here we reorder such that the center is the best point
        indices = list(range(v.shape[0]))
        fY_cf_list = [-fy for fy in list(fY_cf)]
        
        v_eq_list = [v if (np.abs(v) > 1E-8) else 0 for v in v_eq]
        v_ineq_list = [v if (v > 1E-8) else 0 for v in v_ineq]
        
        ## TODO: Ordering can be sensitive to very small error 
        triples = list(zip([-v for v in v_eq_list],
                           [-v for v in v_ineq_list], 
                           fY_cf_list, 
                           indices))
        
        
        triples.sort(key=lambda x:(x[0], x[1], x[2]), reverse=True)
        sorted_index = [ind[3] for ind in triples]
        
        # Y = Y[:, sorted_index]
        # fY_cf = fY_cf[sorted_index]
        
        # fYs_eq = [f[sorted_index] for f in fYs_eq]
        # fYs_ineq = [f[sorted_index] for f in fYs_ineq]
        
        # v[:] = [v[i] for i in sorted_index]
        # v_eq[:] = [v_eq[i] for i in sorted_index]
        # v_ineq[:] = [v_ineq[i] for i in sorted_index]
        
        ## any points that violate the bounds should not be picked as the "best" points
        ## this could happen due to the violation in bound from resampling of points
        
        and_check = []
        for j in range(Y.shape[1]):
            lbt_check = all(self.lbt - 1E-6 <= Y[:,j])
            ubt_check = all(self.ubt + 1E-6 >= Y[:,j])
            check = np.logical_and(lbt_check, ubt_check)
            and_check.append(check)        
                
        good_sorted_index = []
        bad_sorted_index = []
        for index in sorted_index:
            
            if and_check[index]: #if lb <= y <= ub
                good_sorted_index.append(index)
            else: # otherwise
                bad_sorted_index.append(index)
                
        new_sorted_index = good_sorted_index + bad_sorted_index
            
        Y = Y[:, new_sorted_index]
        fY_cf = fY_cf[new_sorted_index]
        
        fYs_eq = [f[new_sorted_index] for f in fYs_eq]
        fYs_ineq = [f[new_sorted_index] for f in fYs_ineq]
        
        v[:] = [v[i] for i in new_sorted_index]
        v_eq[:] = [v_eq[i] for i in new_sorted_index]
        v_ineq[:] = [v_ineq[i] for i in new_sorted_index]
        
        return Y, fY_cf, fYs_eq, fYs_ineq, v, v_eq, v_ineq
    
    def main_run(self, Y:np.ndarray, reorder:bool=True):

        try:
            m_cf = CostFunctionModel(input_symbols=self.input_symbols, 
                                    Y=Y, 
                                    fY=None)
        except LinAlgError:
            raise IllPoisedModel(f"SVD did not converge. Rebuilding from the last point.")
        
        try:
            fY_cf, fYs_eq, fYs_ineq = self.run_simulations(Y)
            v, v_eq, v_ineq = self.calculate_violation(Y=Y, fYs_eq=fYs_eq, fYs_ineq=fYs_ineq)
            if reorder:
                Y, fY_cf, fYs_eq, fYs_ineq, v, v_eq, v_ineq = self.reorder_samples(Y, fY_cf, fYs_eq, fYs_ineq, v, v_eq, v_ineq)
            fail_flag = False
        except FailedSimulation as e:
            fY_cf = None
            fYs_eq = [None]
            fYs_ineq = [None]
            v = np.array([None])
            v_eq = np.array([None])
            v_ineq = np.array([None])
            self.v_eq_lists = [None]
            self.v_ineq_lists = [None]
            fail_flag = True
        
        self.violations = v
        self.violations_eq = v_eq
        self.violations_ineq = v_ineq
        
        m_cf = CostFunctionModel(input_symbols=self.input_symbols, 
                                    Y=Y, 
                                    fY=fY_cf)

        m_eqcs = EqualityConstraintModels(input_symbols=self.input_symbols, 
                                    Y=Y, 
                                    fYs=fYs_eq,
                                    cfm=m_cf)

        m_ineqcs = InequalityConstraintModels(input_symbols=self.input_symbols, 
                                            Y=Y, 
                                            fYs=fYs_ineq,
                                            cfm=m_cf)
        
        m_viol = ViolationModel(input_symbols=self.input_symbols, m_cf = m_cf, m_eqcs=m_eqcs, m_ineqcs=m_ineqcs, Y=Y, fail_flag=fail_flag)
        
              
        return ModelManager(input_symbols=self.input_symbols, m_cf=m_cf, m_eqcs=m_eqcs, m_ineqcs=m_ineqcs, m_viol=m_viol)

    def run_single_simulation(self, y:np.ndarray) -> Tuple[float, float]:
        
        TOL = 1E-7
        
        fy = self.sm.cf.func(y)
        if np.isnan(fy) or np.isinf(fy):
            raise FailedSimulation(f"Failed at x={y}")
            
        v_eq = 0
        for eqc in self.sm.eqcs.eqcs:
            fY = eqc.func(y)
            if np.abs(fY) < TOL:
                fY = 0.0
                 
            if np.isnan(fy) or np.isinf(fy):
                raise FailedSimulation(f"Failed at x={y}")
            if ca.fabs(fY) > v_eq:
                v_eq = ca.fabs(fY) 
        
        v_ineq = 0 ## v_ineq >= 0
        for eqc in self.sm.ineqcs.ineqcs:
            fY = eqc.func(y)
            if np.abs(fY) < TOL:
                fY = 0.0
                
            if np.isnan(fy) or np.isinf(fy):
                raise FailedSimulation(f"Failed at x={y}")
            if -fY > v_ineq:
                v_ineq = -fY
        
        v = ca.fmax(0, ca.fmax(v_eq, v_ineq))
        return fy, v

    def solve_TRQP(self, models:ModelManager, radius:float) -> Tuple[np.ndarray, float, bool]:
        solver = self.opts["solver"]
        trqp_mod = TRQP(models=models, ub=self.ubt, lb=self.lbt, radius=radius, solver=solver)
        sol = trqp_mod.sol.full()[:,0]
        return sol, trqp_mod.radius, trqp_mod.is_compatible
    
    def change_point(self, models:ModelManager, Y:np.ndarray, y_next:np.ndarray, fy_next, v_next, radius:float, it_code:int) -> np.ndarray:
        if fy_next is not None:
            # indices = list(range(self.violations.shape[0]))
            worst_f = models.m_cf.model.f.argsort()
            worst_v = self.violations.argsort()
           
            new_Y = np.concatenate([y_next[:, np.newaxis], Y], axis=1)
            worst_f = np.concatenate([[fy_next], models.m_cf.model.f])
            worst_v = np.concatenate([[v_next], self.violations])
            
            # if it_code in [4, 5]:
            if it_code in []:
                # we are not ordering by violation first, following the algorithm.
                pass
            else:
                indices = list(range(self.violations.shape[0] + 1))
                tuples = list(zip(worst_v, worst_f, indices))
                tuples.sort(key=lambda x:(x[0], x[1]), reverse=False)
                
                indices_1 = [ind[2] for ind in tuples]
                
                new_Y = new_Y[:, indices_1]
            
        else:
            indices = list(range(self.violations.shape[0]))
            
            worst_f = models.m_cf.model.f.argsort()
            worst_v = self.violations.argsort()
            
            tuples = list(zip(worst_v, worst_f, indices))
            tuples.sort(key=lambda x:(x[0], x[1]), reverse=False)
            indices_1 = [ind[2] for ind in tuples]

            new_Y = models.m_cf.model.y*1
            if it_code in [7]:
                new_Y = np.concatenate([y_next[:, np.newaxis], new_Y], axis=1)
                
            else:
                
                new_Y = new_Y[:, indices_1]
                new_Y = np.concatenate([new_Y, y_next[:, np.newaxis]], axis=1)

        and_check = []
        for j in range(new_Y.shape[1]):
            lbt_check = all(self.lbt - 1E-6 <= new_Y[:,j])
            ubt_check = all(self.ubt + 1E-6 >= new_Y[:,j])
            check = np.logical_and(lbt_check, ubt_check)
            and_check.append(check)        
                
        good_sorted_index = []
        bad_sorted_index = []
        # for index in indices_1:
        for index in range(new_Y.shape[1]):
            if and_check[index]: #if lb <= y <= ub
                good_sorted_index.append(index)
            else: # otherwise
                bad_sorted_index.append(index)
                
        new_sorted_index = good_sorted_index + bad_sorted_index
        new_Y = new_Y[:, new_sorted_index]

        niter = 1

        max_points = self.opts['max_points']

        while max_points < new_Y.shape[1] and niter < new_Y.shape[1]: 
        # too many points, remove the point that contributes to the worst model
            # Change point with largest poisedness
            input_symbols = models.m_cf.model.input_symbols
            _model = LagrangePolynomials(input_symbols=input_symbols, pdegree=2)
            _model.initialize(y=new_Y, f=None, tr_radius=None)
            
            poisedness = _model.poisedness(rad=radius, center=Y[:,0])
  
            # index to replace -> poisedness.index            
            if poisedness.index > 0:
                new_Y = np.delete(new_Y, poisedness.index, axis=1)
            else:
                new_Y = np.delete(new_Y, new_Y.shape[1]-1, axis=1)
                pass
            
            niter = niter + 1

        return new_Y
    
    def _update_radius(self, nd:int, nx:int, factor:float, radius:float):
    
        
        # if nx < 2*nd + 1: 
        if nx < self.opts['max_points']:
            # we don't punish bad prediction because we don't have sufficient points for linear model
            return radius
        else:
            
            return factor*radius
        
    def step_model_improvement(self, Y, radius, need_rebuild, need_model_improvement):
        
        ### STEP 2: Model improvement
        ## prepare/check the interpolation points
        if need_rebuild:
            
            center = Y[:, [0]]
            # new_y = center + (Y-center)*radius 
            # new_y = center + radius*generate_uniform_sample_nsphere(k=2*Y.shape[0]+1, d=Y.shape[0], L=self.constants['L_threshold'])
            new_y = center + radius*generate_uniform_sample_nsphere(k=2, d=Y.shape[0], L=self.constants['L_threshold'])
            need_rebuild = False
                    
        elif need_model_improvement:
            # TODO: introduce criticality step!!!
            
            if Y.shape[1] <= Y.shape[0] + 1:
                # model = LagrangePolynomials(input_symbols=self.input_symbols, pdegree=1)
                # model.initialize(y=Y, tr_radius=radius)    
                center = Y[:, [0]]
                new_y = center + radius*generate_uniform_sample_nsphere(k=Y.shape[1], d=Y.shape[0], L=self.constants['L_threshold'])
            else: 
                model = LagrangePolynomials(input_symbols=self.input_symbols, pdegree=2)
                model.initialize(y=Y, tr_radius=radius)
            
                try: 
                    poisedness = model.poisedness(rad=radius, center=Y[:,0])
                    if poisedness.max_poisedness() > self.constants['L_threshold']:
                        sg = SetGeometry(input_symbols=self.input_symbols, Y=Y, rad=radius, L=self.constants['L_threshold'])
                        sg.improve_geometry()     
                        improved_model = sg.model
                        new_y = improved_model.y
                        
                    else:
                        new_y = Y*1
                except PoisednessIsZeroException:
                    center = Y[:, [0]]
                    new_y = center + radius*generate_uniform_sample_nsphere(k=Y.shape[1], d=Y.shape[0], L=self.constants['L_threshold'])
                                    
        else:
            new_y = Y*1
        
        return new_y
    
    def step_organize_and_inform(self, k, radius, it_code):
        
        if k == 0:
            # first iteration, best point IS the center of the trust-region
            _fy = self.models.m_cf.model.f
            _v = self.violations
                    
            self.filter_SQP.add_to_filter((_fy[0], _v[0]))
            Y = self.models.m_cf.model.y*1
            y_curr = Y[:,0]
            f_curr = self.models.m_cf.model.f[0]
            v_curr = self.violations[0]
            
            best_point = dict()
            best_point['y'] = self.denorm(y_curr)
            best_point['f'] = f_curr
            best_point['v'] = v_curr
            
        else:
            # next iterations, best point might not be the center of the trust-region.
            
            Y = self.models.m_cf.model.y*1
            y_test = Y[:,0]
            f_test = self.models.m_cf.model.f[0]
            v_test = self.violations[0]
            
            # check bound
            dist = 0 # distance to bound for the previous point
            p = self.iterates[k-1]['best_point']['y']
            for i, (lb, ub) in enumerate(zip(self.lbt, self.ubt)):
                if p[i] < lb:
                    dist += (p[i] - lb)**2
                elif p[i] > ub:
                    dist += (p[i] - ub)**2
                    
            dist_test = 0 # distance to bound for the test point
            for i, (lb, ub) in enumerate(zip(self.lbt, self.ubt)):
                if y_test[i] < lb:
                    dist_test += (y_test[i] - lb)**2
                elif y_test[i] > ub:
                    dist_test += (y_test[i] - ub)**2
            
            # bound: If the bound violation for the test point is larger, then we pick the previous point. 
            # otherwise continue with other test
            if (dist >= dist_test):
                # check violation
                if v_test is not None:
                    if self.iterates[k-1]['best_point']['v'] > v_test:
                        best_point = dict()
                        best_point['y'] = self.denorm(y_test)
                        best_point['f'] = f_test
                        best_point['v'] = v_test
                    else:
                        _diff = np.abs(self.iterates[k-1]['best_point']['v'] - v_test)
                        if self.iterates[k-1]['best_point']['f'] > f_test and _diff < 1E-5:
                            best_point = dict()
                            best_point['y'] = self.denorm(y_test)
                            best_point['f'] = f_test
                            best_point['v'] = v_test
                        else:
                            best_point = self.iterates[k-1]['best_point']
                else:
                    best_point = self.iterates[k-1]['best_point']
            else:
                best_point = self.iterates[k-1]['best_point']
        
        iterates = dict()
        iterates['iteration_no'] = k
        # iterates['Y'] = Y
        # iterates['fY'] = self.models.m_cf.model.f
        # iterates['v'] = self.violations
        # iterates['all_violations'] = {'equality': self.v_eq_lists, 'inequality': self.v_ineq_lists}
        # iterates['y_curr'] = self.denorm(Y[:,0])
        iterates['best_point'] = best_point 
        iterates['filters'] = self.filter_SQP.filters
        iterates['radius'] = radius
        iterates['models'] = self.models
        iterates['total_number_of_function_calls'] = self.sm.cf.number_of_function_calls
        if k > 0:
            iterates['number_of_function_calls'] = iterates['total_number_of_function_calls'] - self.iterates[k-1]['total_number_of_function_calls'] 
        else:
            iterates['number_of_function_calls'] = iterates['total_number_of_function_calls']*1
        
        neval = iterates['number_of_function_calls']
        iterates["it_code"] = it_code
        
        #Inform user
        if (best_point['f'] is not None) and (self.models.m_cf.model.f[0] is not None):
            # print(f"It. {k}: Best point, x= {self.denorm(y_curr)}, f= {f_curr:.5e}, v= {v_curr:.5e}, r= {radius:.2e}, g= {np.linalg.norm(self.models.m_cf.model.gradient(y_curr)):.2e}, it_code= {it_code}, nevals= {neval}, n_points= {Y.shape[1]}")
            # print(f"It. {k}: Best point, x= {best_point['y']}, f= {best_point['f']:.5e}, v= {best_point['v']:.5e}, r= {radius:.2e}, g= {np.linalg.norm(self.models.m_cf.model.gradient(best_point['y'])):.2e}, it_code= {it_code}, nevals= {neval}, n_points= {Y.shape[1]}")
            print(f"It. {k}: f= {best_point['f']:.5e}, v= {best_point['v']:.5e}, r= {radius:.2e}, it_code= {it_code}, nevals= {neval}, n_points= {Y.shape[1]}")
            return iterates, True
        
        else: # failed simulation
            print(f"It. {k}: Failed. r= {radius:.2e}, prev. it_code= {it_code}, nevals= {neval}, ")
            return iterates, False
        
    def step_solve_subproblem(self, iterates):
        Y = self.models.m_cf.model.y*1
        
        try:
            y_next, radius, self.is_trqp_compatible = self.solve_TRQP(models=self.models, radius=iterates['radius'])
            for i in range(self.models.m_cf.model.y.shape[1]):
                if np.linalg.norm(y_next - self.models.m_cf.model.y[:,i]) == 0.0:
                    raise RedundantPoint(y_next)
                
            exit_code = 'Normal step'
            it_code = 0
        except EndOfAlgorithm:
            print(f"Impossible to solve restoration step")
            exit_code = 'Restoration step'
            # need_model_improvement = True
            it_code = 9
            # self.iterates.append(iterates)
            # break
        except RedundantPoint as e:
            print(f"Point already exist : {e}. Try to reduce poisedness threshold")
            exit_code = 'Redundant point'
            # need_model_improvement = True
            self.is_trqp_compatible = False
            it_code = 8
            y_next = np.mean(Y, axis=1) + radius*np.random.rand(Y.shape[0])
            # Add random values to avoid the same value
        
        self.iterates.append(iterates)
        
        return y_next, radius, exit_code, it_code
    
    def step_updates(self, iterates, y_next, radius):
        
        # Y = iterates['Y']
        Y = self.models.m_cf.model.y*1
        
        # if self.is_trqp_compatible:
        try:
            fy_next, v_next = self.run_single_simulation(y_next)
            is_acceptable_in_the_filter = self.filter_SQP.add_to_filter((fy_next, v_next), to_add=False)
            is_next_point_good = True
        except FailedSimulation as e:
            is_acceptable_in_the_filter = False
            is_next_point_good = False
            fy_next = None
            v_next = None
            pass
        
        if self.is_trqp_compatible:
            
            if is_next_point_good:
            
                # print(f"y_next, fy_next, v_next, is_acceptable_in_the_filter = {self.denorm(y_next)}, {fy_next}, {v_next}, {is_acceptable_in_the_filter}")     
                if is_acceptable_in_the_filter:
                    
                    # y_curr = self.denorm(Y[:,0])
                    
                    v_curr = self.models.m_viol.feval_normalized(Y[:,0]).full()[0][0]
                    
                    # mfy_curr = self.models.m_cf.model.model_polynomial_normalized.feval(Y[:,0])
                    # mfy_next = self.models.m_cf.model.model_polynomial_normalized.feval(y_next)
                    mfy_curr = self.models.m_cf.model.interpolate(Y[:,0])
                    mfy_next = self.models.m_cf.model.interpolate(y_next)
                    fy_curr = self.models.m_cf.model.f[0]
                    
                    rho = (fy_curr - fy_next)/(mfy_curr - mfy_next)
                    if mfy_curr - mfy_next >= self.constants['kappa_vartheta']*(v_curr**2) and rho < self.constants['eta_1']:
                        radius = self._update_radius(Y.shape[0], Y.shape[1], self.constants['gamma_1'], radius)
                        need_rebuild = False
                        need_model_improvement = True
                        it_code = 1
                        
                        Y = self.change_point(self.models, Y, y_next, fy_next, v_next, radius, it_code)
                        
                    elif mfy_curr - mfy_next < self.constants['kappa_vartheta']*(v_curr**2):
                        self.filter_SQP.add_to_filter((fy_next, v_next))
                        
                        if rho >= self.constants['eta_2']:
                            radius = self._update_radius(Y.shape[0], Y.shape[1], self.constants['gamma_2'], radius)
                            need_rebuild = False
                            need_model_improvement = False
                            it_code = 2
                        elif rho >= self.constants['eta_1'] and rho < self.constants['eta_2']:
                            radius = self._update_radius(Y.shape[0], Y.shape[1], self.constants['gamma_1'], radius)
                            need_rebuild = False
                            need_model_improvement = True
                            it_code = 3
                        else:
                            radius = self._update_radius(Y.shape[0], Y.shape[1], self.constants['gamma_0'], radius)
                            need_rebuild = False
                            need_model_improvement = True
                            it_code = 8
                        
                        Y = self.change_point(self.models, Y, y_next, fy_next, v_next, radius, it_code)
                        
                    else:
                        if rho >= self.constants['eta_2']:
                            radius = self._update_radius(Y.shape[0], Y.shape[1], self.constants['gamma_2'], radius)
                            need_rebuild = False
                            need_model_improvement = False
                            it_code = 4
                        elif rho >= self.constants['eta_1'] and rho < self.constants['eta_2']:
                            radius = self._update_radius(Y.shape[0], Y.shape[1], self.constants['gamma_1'], radius)
                            need_rebuild = False
                            need_model_improvement = True
                            it_code = 5
                        else:
                            radius = self._update_radius(Y.shape[0], Y.shape[1], self.constants['gamma_0'], radius)
                            need_rebuild = False
                            need_model_improvement = True
                            it_code = 9
                        
                        Y = self.change_point(self.models, Y, y_next, fy_next, v_next, radius, it_code)
                        
                else:
                    radius = self._update_radius(Y.shape[0], Y.shape[1], self.constants['gamma_0'], radius)
                    it_code = 6
                    need_rebuild = False
                    need_model_improvement = True
                    # need_model_improvement = False
                    Y = self.change_point(self.models, Y, y_next, fy_next, v_next, radius, it_code)
            else:
                # radius = self._update_radius(Y.shape[0], Y.shape[1], self.constants['gamma_0'], radius)
                radius = radius*self.constants['gamma_0']
                it_code = 13
                Y = self.change_point(self.models, Y, y_next, fy_next, v_next, radius, it_code)
                need_rebuild = True
                need_model_improvement = False
                    
        else:
            
            fy_curr = self.models.m_cf.model.f[0]
            v_curr = self.violations[0]
            
            _ = self.filter_SQP.add_to_filter((fy_curr, v_curr))
            
            try:
                if (v_curr - v_next)**2 > 1E-6:        
                    # Enlarge radius when TRQP is not compatible -> will diverge if there is no feasible solutions
                    radius = self._update_radius(Y.shape[0], Y.shape[1], self.constants['gamma_2'], radius)
                else:
                    radius = self._update_radius(Y.shape[0], Y.shape[1], self.constants['gamma_0'], radius)
            except:
                radius = self._update_radius(Y.shape[0], Y.shape[1], self.constants['gamma_2'], radius)
            
            it_code = 7
            Y = self.change_point(self.models, Y, y_next, None, None, radius, it_code)
            need_rebuild = False
            need_model_improvement = False
        
        return Y, radius, need_rebuild, need_model_improvement, it_code
    
    def optimize(self, max_iter=15):
        
        need_model_improvement = False
        need_rebuild = False
        it_code = -1
        neval = 0
        
        # initialize filter
        self.filter_SQP = FilterSQP(constants=self.constants)
        radius = self.constants['init_radius']
        Y = self.dataset*1

        self.iterates = []
        
        try: 
            for k in range(max_iter):
                if radius < self.constants["stopping_radius"]:
                    print(f"Radius too small.")
                    exit_code = 'Minimum radius'
                    break
                
                # STEP 1: Model improvement algorithm
                new_y = self.step_model_improvement(Y, radius, need_rebuild, need_model_improvement)

                # STEP 2: Run simulation and build models
                try:
                    self.models = self.main_run(Y=new_y, reorder=True)
                except IllPoisedModel:
                    need_rebuild = False
                    need_model_improvement = True
                    continue
                except FailedSimulation:
                    need_rebuild = True
                    need_model_improvement = False
                    continue
                
                # STEP 2.5: Some organization and inform user 
                iterates, sim_success = self.step_organize_and_inform(k, radius, it_code)
                if not sim_success:
                    it_code = 10
                    radius = self.constants['gamma_0']*radius
                    need_model_improvement = True
                    need_rebuild = True
                    self.iterates.append(iterates)
                    continue
                
                # STEP 3: Solving the subproblem
                y_next, radius, exit_code, it_code = self.step_solve_subproblem(iterates)
                
                
                # STEP 4: Updates
                Y, radius, need_rebuild, need_model_improvement, it_code = self.step_updates(iterates, y_next, radius)    
                
                if k == max_iter - 1:
                    exit_code = 'Maximum iteration'

                # check whether the upcoming simulation exceeds the budget
                if self.opts['budget'] < self.sm.cf.number_of_function_calls:
                    exit_code = 'Budget Exceeded'
                    break
        
            try:
                tmp = exit_code
            except:
                exit_code = "Unknown"
        except KeyboardInterrupt:
            print("User interrupted")
            exit_code = 'User interruption'
            it_code = 666
            
        self.termination_status = exit_code
        print(f"Termination code : {exit_code}")