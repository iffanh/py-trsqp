import numpy as np
import casadi as ca
from typing import List, Tuple
from multiprocessing import Pool

from .utils.TR_exceptions import IncorrectConstantsException, EndOfAlgorithm, RedundantPoint, PoisednessIsZeroException, SolutionFound
from .utils.simulation_manager import SimulationManager
from .utils.model_manager import SetGeometry, ModelManager, CostFunctionModel, EqualityConstraintModels, InequalityConstraintModels, ViolationModel
from .utils.trqp import TRQP
from .utils.filter import FilterSQP
from .utils.model_improvement_without_feval import generate_uniform_sample_nsphere
# from .utils.lagrange_polynomial import LagrangePolynomials

class TrustRegionSQPFilter():
    
    def __init__(self, x0:np.ndarray, k:int, cf:callable, eqcs:List[callable], ineqcs:List[callable], constants:dict=dict()) -> None:
        
        def _check_constants(constants:dict) -> dict:
            
            try:
                tmp = constants["gamma_0"]
            except:
                constants["gamma_0"] = 0.2
                
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

        self.constants = _check_constants(constants=constants)
        self.n_eqcs, self.n_ineqcs = _check_constraints(eqcs=eqcs, ineqcs=ineqcs)
        self.sm = SimulationManager(cf, eqcs, ineqcs) # Later this will be refactored for reservoir simulation

        self.dataset = x0[:, np.newaxis] + constants['init_radius']*generate_uniform_sample_nsphere(k=k, d=x0.shape[0])
        self.input_symbols = ca.SX.sym('x', x0.shape[0])

        pass

    def __str__(self) -> str:
        return f"TrustRegionSQPFilter(n_eqcs={self.n_eqcs}, n_ineqcs={self.n_ineqcs})"
    
    def run_simulations(self, Y:np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        
        # run cost function and build the corresponding model
        fY_cf = []
        for i in range(Y.shape[1]):
            fY = self.sm.cf.func(Y[:,i])
            fY_cf.append(fY)
        fY_cf = np.array(fY_cf)
        
        # do the same with equality constraints
        fYs_eq = []
        for eqc in self.sm.eqcs.eqcs:
            
            fYs = []
            for i in range(Y.shape[1]):
                fY = eqc.func(Y[:,i])
                fYs.append(fY)
            
            fYs = np.array(fYs)
            fYs_eq.append(fYs)

        # do the same with inequality constraints
        fYs_ineq = []
        for ineqc in self.sm.ineqcs.ineqcs:
            
            fYs = []
            for i in range(Y.shape[1]):
                fY = ineqc.func(Y[:,i])
                fYs.append(fY)
            
            fYs = np.array(fYs)
            fYs_ineq.append(fYs)
        
        return fY_cf, fYs_eq, fYs_ineq
    
    def calculate_violation(self, Y:np.ndarray, fYs_eq:List[np.ndarray], fYs_ineq:List[np.ndarray]):
        
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
                
                v_eq_list.append(tmp)
                v = ca.fmax(v, ca.fabs(tmp))
                v_eq = ca.fmax(v_eq, ca.fabs(tmp))
                
            v_ineq = 0.0 # getting the maximum value from the inequality constraints
            v_ineq_list = [] # saving all the violation value for inequality constraints
            for i in range(self.sm.ineqcs.n_ineqcs):
                tmp = fYs_ineq[i][j]
                
                v_ineq_list.append(tmp)
                v = ca.fmax(v, ca.fmax(0.0, -fYs_ineq[i][j]))
                v_ineq = ca.fmax(v_ineq, ca.fmax(0.0, -fYs_ineq[i][j]))
            
            violations.append(v)
            violations_eq.append(v_eq)
            violations_ineq.append(v_ineq)    
            
            self.v_eq_lists.append(v_eq_list)
            self.v_ineq_lists.append(v_ineq_list)
            
        violations = np.array(violations)
        
        return violations, violations_eq, violations_ineq
    
    def reorder_samples(self, Y, fY_cf, fYs_eq, fYs_ineq):
        
        ## Here we reorder such that the center is the best point
        indices = list(range(self.violations.shape[0]))
        fY_cf_list = [-fy for fy in list(fY_cf)]
        
        triples = list(zip([-v for v in self.violations_eq],
                           [-v for v in self.violations_ineq], 
                           fY_cf_list, 
                           indices))
        
        triples.sort(key=lambda x:(x[0], x[1], x[2]), reverse=True)
        sorted_index = [ind[3] for ind in triples]
        
        Y = Y[:, sorted_index]
        fY_cf = fY_cf[sorted_index]
        
        fYs_eq = [f[sorted_index] for f in fYs_eq]
        fYs_ineq = [f[sorted_index] for f in fYs_ineq]
        
        return Y, fY_cf, fYs_eq, fYs_ineq
    
    def main_run(self, Y:np.ndarray):

        fY_cf, fYs_eq, fYs_ineq = self.run_simulations(Y)
        
        v, v_eq, v_ineq = self.calculate_violation(Y=Y, fYs_eq=fYs_eq, fYs_ineq=fYs_ineq)
        self.violations = v
        self.violations_eq = v_eq
        self.violations_ineq = v_ineq
                
        Y, fY_cf, fYs_eq, fYs_ineq = self.reorder_samples(Y, fY_cf, fYs_eq, fYs_ineq)
        
        m_cf = CostFunctionModel(input_symbols=self.input_symbols, 
                                 Y=Y, 
                                 fY=fY_cf)

        m_eqcs = EqualityConstraintModels(input_symbols=self.input_symbols, 
                                    Y=Y, 
                                    fYs=fYs_eq)

        m_ineqcs = InequalityConstraintModels(input_symbols=self.input_symbols, 
                                              Y=Y, 
                                              fYs=fYs_ineq)
        
        m_viol = ViolationModel(input_symbols=self.input_symbols, m_eqcs=m_eqcs, m_ineqcs=m_ineqcs, Y=Y)

        return ModelManager(input_symbols=self.input_symbols, m_cf=m_cf, m_eqcs=m_eqcs, m_ineqcs=m_ineqcs, m_viol=m_viol)

    def run_single_simulation(self, y:np.ndarray) -> Tuple[float, float]:
        fy = self.sm.cf.func(y)
        
        v_eq = 0
        for eqc in self.sm.eqcs.eqcs:
            fY = eqc.func(y) 
            if ca.fabs(fY) > v_eq:
                v_eq = ca.fabs(fY) 
        
        v_ineq = 0
        for eqc in self.sm.ineqcs.ineqcs:
            fY = eqc.func(y)
            if -fY > v_ineq:
                v_ineq = -fY
        
        v = ca.fmax(0, ca.fmax(v_eq, -v_ineq))
        return fy, v

    def solve_TRQP(self, models:ModelManager, radius:float) -> Tuple[np.ndarray, float, bool]:
        trqp_mod = TRQP(models, radius)
        sol = trqp_mod.sol.full()[:,0]
        is_trqp_compatible = trqp_mod.is_compatible
        radius = trqp_mod.radius
        return sol, radius, is_trqp_compatible
    
    def change_point(self, models:ModelManager, Y:np.ndarray, y_next:np.ndarray, radius:float, replace_type:str) -> np.ndarray:
        
        if replace_type == 'improve_model':
            # Change point with largest poisedness
            poisedness = models.m_cf.model.poisedness(rad=radius, center=Y[:,0])
            # index to replace -> poisedness.index
            new_Y = Y*1
            new_Y[:, poisedness.index] = y_next
            
        elif replace_type == 'worst_point': 
            indices = list(range(self.violations.shape[0]))
            worst_f = models.m_cf.model.f.argsort()
            worst_v = models.m_viol.violations.argsort()
            
            tuples = list(zip(worst_v, worst_f, indices))
            tuples.sort(key=lambda x:(x[0], x[1]), reverse=False)
            worst_index = [ind[2] for ind in tuples][-1]
        
            new_Y = models.m_cf.model.y*1
            new_Y[:, worst_index] = y_next

        return new_Y

    def optimize(self, max_iter=15):
        
        need_model_improvement = False
        it_code = -1
        
        # initialize filter
        self.filter_SQP = FilterSQP(constants=self.constants)
        radius = self.constants['init_radius']
        Y = self.dataset*1

        self.iterates = []
        for k in range(max_iter):

            if radius < self.constants["stopping_radius"]:
                print(f"Radius too small.")
                term_status = 'Minimum radius'
                break
            
            self.models = self.main_run(Y=Y)
            Y = self.models.m_cf.model.y*1
            y_curr = Y[:,0]
            f_curr = self.models.m_cf.model.f[0]
            v_curr = self.violations[0]
            print(f"Iteration : {k}: Best current point, x: {y_curr}. f: {f_curr}, v: {v_curr}, it_code: {it_code}")
            
            iterates = dict()
            iterates['iteration_no'] = k
            iterates['Y'] = Y
            iterates['fY'] = self.models.m_cf.model.f
            iterates['v'] = self.violations
            iterates['all_violations'] = {'equality': self.v_eq_lists, 'inequality': self.v_ineq_lists}
            iterates['y_curr'] = Y[:,0]
            iterates['filters'] = self.filter_SQP.filters
            iterates['radius'] = radius
            iterates['models'] = self.models
            iterates['total_number_of_function_calls'] = self.sm.cf.number_of_function_calls
            if k > 0:
                iterates['number_of_function_calls'] = iterates['total_number_of_function_calls'] - self.iterates[k-1]['total_number_of_function_calls'] 
            else:
                iterates['number_of_function_calls'] = iterates['total_number_of_function_calls']*1
            
            if need_model_improvement:
                poisedness = self.models.m_cf.model.poisedness(rad=radius, center=Y[:,0])
                if poisedness.max_poisedness() > self.constants['L_threshold']:
                    sg = SetGeometry(input_symbols=self.input_symbols, Y=Y, rad=radius, L=self.constants['L_threshold'])
                    sg.improve_geometry()        
                    improved_model = sg.model
                    self.models = self.main_run(Y=improved_model.y)
                    Y = improved_model.y
            
            poisedness = self.models.m_cf.model.poisedness(rad=radius, center=Y[:,0])
            if k == 0:
                
                _fy = self.models.m_cf.model.f
                _v = self.violations
                
                for ii in range(_v.shape[0]):
                    _ = self.filter_SQP.add_to_filter((_fy[ii], _v[ii]))
            
            try:
                y_next, radius, self.is_trqp_compatible = self.solve_TRQP(models=self.models, radius=radius)
                for i in range(self.models.m_cf.model.y.shape[1]):
                    if np.linalg.norm(y_next - self.models.m_cf.model.y[:,i]) == 0.0:
                        raise RedundantPoint(y_next)
            except EndOfAlgorithm:
                print(f"Impossible to solve restoration step. Current iterate = {Y[:,0]}")
                term_status = 'Restoration step'
                it_code = 9
                self.iterates.append(iterates)
                break
            except RedundantPoint as e:
                print(f"Point already exist : {e}. Try to reduce poisedness threshold")
                term_status = 'Redundant point'
                need_model_improvement = True
                it_code = 8
                self.iterates.append(iterates)
                continue
                # break
            
            self.iterates.append(iterates)
                
            if self.is_trqp_compatible:
                fy_next, v_next = self.run_single_simulation(y_next)
                is_acceptable_in_the_filter = self.filter_SQP.add_to_filter((fy_next, v_next))

                if is_acceptable_in_the_filter:
                    v_curr = self.models.m_viol.feval(y_curr).full()[0][0]
                    
                    mfy_curr = self.models.m_cf.model.model_polynomial.feval(y_curr)
                    mfy_next = self.models.m_cf.model.model_polynomial.feval(y_next)
                    fy_curr = self.models.m_cf.model.f[0]
                    
                    rho = (fy_curr - fy_next)/(mfy_curr - mfy_next)
                    if mfy_curr - mfy_next >= self.constants['kappa_vartheta']*(v_curr**2): 
                        if rho < self.constants['eta_1']:
                            radius = self.constants['gamma_1']*radius
                            Y = Y*1
                            Y[:, 1:] = Y[:, [0]] + (Y[:, 1:] - Y[:, [0]])*self.constants['gamma_0']
                            ### TODO: ad hoc to allow shrinking of the points (not only the radius) to "improve" the interpoaltion in the neighborhood
                            ### This messed up the precision of the center point somehow
                            need_model_improvement = True
                            it_code = 1
                        else:
                            if rho >= self.constants['eta_2']:
                                radius = radius*self.constants['gamma_2']
                                Y = self.change_point(self.models, Y, y_next, radius, 'worst_point')
                                need_model_improvement = False
                                it_code = 2
                            else:
                                radius = radius*self.constants['gamma_1']
                                Y = self.change_point(self.models, Y, y_next, radius, 'worst_point')
                                need_model_improvement = True
                                it_code = 3
                    else:
                        self.filter_SQP.add_to_filter((fy_next, v_next))
                            
                        if rho >= self.constants['eta_2']:
                            radius = radius*self.constants['gamma_2']
                            Y = self.change_point(self.models, Y, y_next, radius, 'worst_point')
                            need_model_improvement = False
                            it_code = 4
                        else:
                            radius = radius*self.constants['gamma_1']
                            Y = self.change_point(self.models, Y, y_next, radius, 'worst_point')
                            need_model_improvement = True
                            it_code = 5
                    pass
                
                else:
                    radius = self.constants['gamma_0']*radius
                    need_model_improvement = True
                    Y = Y*1
                    Y[:, 1:] = Y[:, [0]] + (Y[:, 1:] - Y[:, [0]])*self.constants['gamma_0'] # DEBUG
                    ### TODO: ad hoc to allow shrinking of the points (not only the radius) to "improve" the interpoaltion in the neighborhood
                    ### This messed up the precision of the center point somehow
                    it_code = 6
                    
                    
                    
                    
                    
            
            else:
                fy_curr = self.models.m_cf.model.f[0]
                v_curr = self.models.m_viol.feval(y_curr).full()[0][0]
                _ = self.filter_SQP.add_to_filter((fy_curr, v_curr))
                
                Y = self.change_point(self.models, Y, y_next, radius, 'worst_point')
                need_model_improvement = True
                it_code = 7
        
            if k == max_iter - 1:
                term_status = 'Maximum iteration'
            
        self.termination_status = term_status