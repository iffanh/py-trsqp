import numpy as np
import casadi as ca
from typing import List
from .lagrange_polynomial import LagrangePolynomials
from .model_improvement_without_feval import ModelImprovement

class SetGeometry():
    def __init__(self, input_symbols, Y:np.ndarray, rad:float=None, L:float=1.5) -> None:
        
        self.input_symbols = input_symbols
        self.Y = Y*1
        self.rad = rad
        self.L = L
        self.model = LagrangePolynomials(input_symbols=input_symbols, pdegree=2)
        self.model.initialize(y=Y, f=None, tr_radius=rad)
        
    def poisedness(self) -> float:
        
        if self.rad is None: 
            self.rad = self.model.sample_set.ball.rad*1
            p = self.model.poisedness(rad=self.rad, center=self.Y[:,0]).poisedness
        else:
            p = self.model.poisedness(rad=self.rad, center=self.Y[:,0]).poisedness
            
        return p
    
    def improve_geometry(self): 
        
        mi = ModelImprovement(input_symbols=self.input_symbols)
        self.model = mi.improve_model(lpolynomials=self.model, 
                                        rad=self.rad, 
                                        center=self.Y[:,0],
                                        L=self.L, 
                                        max_iter=self.Y.shape[1])
        

class CostFunctionModel():
    def __init__(self, input_symbols, Y:np.ndarray, fY:np.ndarray) -> None:
        
        if Y.shape[1] <= Y.shape[0]:
            self.model = LagrangePolynomials(input_symbols=input_symbols, pdegree=1)
            self.model.initialize(y=Y, f=fY, tr_radius=None)    
        else: 
            self.model = LagrangePolynomials(input_symbols=input_symbols, pdegree=2)
            self.model.initialize(y=Y, f=fY, tr_radius=None)

    def __str__(self) -> str:
        return f"CostFunctionModel(model = {self.model.model_polynomial.symbol})"

class EqualityConstraintModels():
    def __init__(self, input_symbols, Y:np.ndarray, fYs:List[np.ndarray], cfm:CostFunctionModel) -> None:
        # to save computation, lagrange polynomial bases are obtained from CostFunctionModel()
        
        self.models = [EqualityConstraintModel(input_symbols, Y, fY, i, cfm) for i, fY in enumerate(fYs)]
        self.n = len(self.models)
        self.cfm = cfm

    def __str__(self) -> str:
        return f"EqualityConstraintModels(n = {self.n})"

class EqualityConstraintModel():
    def __init__(self, input_symbols, Y:np.ndarray, fY:np.ndarray, index:int, cfm:CostFunctionModel) -> None:
        if Y.shape[1] <= Y.shape[0]:
            self.model = LagrangePolynomials(input_symbols=input_symbols, pdegree=1)
            # self.model.initialize(y=Y, f=fY, tr_radius=None) 
            self.model.initialize(y=Y, f=fY, tr_radius=None, lpolynomials=cfm.model.lagrange_polynomials_normalized)   
        else: 
            self.model = LagrangePolynomials(input_symbols=input_symbols, pdegree=2)
            # self.model.initialize(y=Y, f=fY, tr_radius=None)
            self.model.initialize(y=Y, f=fY, tr_radius=None, lpolynomials=cfm.model.lagrange_polynomials_normalized)   
        self.index = index

    def __str__(self) -> str:
        return f"EqualityConstraintModel(index={self.index}, model = {self.model.model_polynomial.symbol})"

class InequalityConstraintModels():
    def __init__(self, input_symbols, Y:List[np.ndarray], fYs:List[np.ndarray], cfm:CostFunctionModel) -> None:
        
        # self.models = [InequalityConstraintModel(input_symbols, Y, fY, i, 2) if i < (L - 2*nx) else InequalityConstraintModel(input_symbols, Y, fY, i, 1) for i, fY in enumerate(fYs)]
        self.models = [InequalityConstraintModel(input_symbols, Y, fY, i, cfm) for i, fY in enumerate(fYs)]
        self.n = len(self.models)
        self.cfm = cfm

    def __str__(self) -> str:
        return f"InequalityConstraintModels(n = {self.n})"

class InequalityConstraintModel():
    def __init__(self, input_symbols, Y:np.ndarray, fY:np.ndarray, index:int, cfm:CostFunctionModel) -> None:
        
        if Y.shape[1] <= Y.shape[0]:
            self.model = LagrangePolynomials(input_symbols=input_symbols, pdegree=1)
            # self.model.initialize(y=Y, f=fY, tr_radius=None)
            self.model.initialize(y=Y, f=fY, tr_radius=None, lpolynomials=cfm.model.lagrange_polynomials_normalized)   
        else: 
            self.model = LagrangePolynomials(input_symbols=input_symbols, pdegree=2)
            # self.model.initialize(y=Y, f=fY, tr_radius=None)
            self.model.initialize(y=Y, f=fY, tr_radius=None, lpolynomials=cfm.model.lagrange_polynomials_normalized)    
        self.index = index

    def __str__(self) -> str:
        return f"InequalityConstraintModel(index={self.index}, model = {self.model.model_polynomial.symbol})"

class ViolationModel():
    def __init__(self, input_symbols, m_cf:CostFunctionModel, m_eqcs:EqualityConstraintModels, m_ineqcs:InequalityConstraintModels, Y:np.ndarray, fail_flag:bool=False) -> None:
        
    
        if fail_flag:
            self.violations = np.array([None])
        else:
            # create violation function Eq 15.5.3
            # v = 0.0
            # for m in m_eqcs.models:
            #     v = ca.fmax(v, ca.fabs(m.model.model_polynomial.symbol))
        
            # for m in m_ineqcs.models:
            #     v = ca.fmax(v, ca.fmax(0, -m.model.model_polynomial.symbol)) # TODO:ADHOC
                
            # self.symbol = v
            # self.feval = ca.Function('Violation', [input_symbols], [self.symbol])
            
            # self.violations = []
            # for i in range(Y.shape[1]):
            #     self.violations.append(self.feval(Y[:,i]).full()[0][0])
                
            # self.violations = np.array(self.violations)

            # NORMALIZED
            v = 0.0
            for m in m_eqcs.models:
                v = ca.fmax(v, ca.fabs(m.model.model_polynomial_normalized.symbol))
        
            for m in m_ineqcs.models:
                v = ca.fmax(v, ca.fmax(0, -m.model.model_polynomial_normalized.symbol)) # TODO:ADHOC
                
            self.symbol_normalized = v
            self.feval = ca.Function('Violation', [input_symbols], [self.symbol_normalized])
            
            center = m_cf.model.center
            radius = m_cf.model.tr_radius
            
            self.feval_normalized = lambda x: self.feval((x - center)/radius)
            
            self.violations_normalized = []
            for i in range(Y.shape[1]):
                self.violations_normalized.append(self.feval(Y[:,i]).full()[0][0])
                
            self.violations_normalized = np.array(self.violations_normalized)
            
class ModelManager():
    """
    Responsible for managing ALL the polynomial models
    """

    def __init__(self, input_symbols, m_cf:CostFunctionModel, m_eqcs:EqualityConstraintModels, m_ineqcs:InequalityConstraintModels, m_viol:ViolationModel) -> None:
        self.m_cf = m_cf
        self.m_eqcs = m_eqcs
        self.m_ineqcs = m_ineqcs
        self.input_symbols = input_symbols
        self.m_viol = m_viol

    def __str__(self) -> str:
        return f"ModelManager(input_symbols = {self.input_symbols}, n_eqcs = {self.m_eqcs.n}, n_ineqcs = {self.m_ineqcs.n})"