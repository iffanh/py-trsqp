""" 
    This script should be able to find the polynomials, 
    given the interpolation set Y = {y0, y1, ..., yp} in R^n
    
    The polynomials takes the form:
     - Basis : 
        m(x) = \sum_{i=0}^p \phi_i(x) \alpha_i
     - Lagrange :
        m(x) = \sum_{i=0}^p f(y^i) l_i(x)
"""

import numpy as np
import itertools
import scipy
import scipy.special
import functools
from scipy.optimize import minimize, NonlinearConstraint, Bounds
from typing import Any, Tuple, List, Optional
import casadi as ca
from casadi import Function, SX, DM, mtimes, vertcat, horzcat

from .generate_poised_sets import SampleSets
from .TR_exceptions import PoisednessIsZeroException

from numpy import linalg as la
import numpy as np
import copy

from multiprocessing import Pool

def lagrange(poly, j, input_symbols):
    return LagrangePolynomial(poly, Function(f'lambda_{j}', [input_symbols], [poly]))
        
def _product(*args, repeat=1, limit=3):
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool if np.sum(x) + y < limit]
    for prod in result:
        yield tuple(prod)
       
@functools.lru_cache() 
def product(*args, repeat=1, limit=3):
    m = [i for i in _product(*args, repeat=repeat, limit=limit)]
    return m

def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False
    
class PolynomialBase:
    def __init__(self, symbol, feval:callable) -> None:
        
        self.symbol = ca.simplify(symbol)
        self.feval = feval
        
class LagrangePolynomial:
    def __init__(self, symbol, feval:callable) -> None:
        
        self.symbol = ca.simplify(symbol)
        self.feval = feval
        self.max_sol = None
        
    def _func_to_minimize(self, x:np.ndarray):
        return -np.abs(self.feval(x))
    
    def cons_f(self, x:np.ndarray, center:np.ndarray) -> float:
        return np.linalg.norm(x - center)
    
    def _define_nonlinear_constraint(self, rad:float, center:np.ndarray) -> NonlinearConstraint:
        return NonlinearConstraint(functools.partial(self.cons_f, center), 0, rad**2)
    
    def _define_nonlinear_bounds(self, rad:float, center:np.ndarray) -> Bounds:
        return Bounds(lb=center-rad, ub=center+rad)
        
    def _find_max_given_boundary(self, x0:np.ndarray, rad:float, center:np.ndarray) -> Tuple[Any, float]:
        
        self.success = False
        iteration = 1
        while not self.success and iteration < 2:
            nlinear_constraint = self._define_nonlinear_constraint(rad, center)
            nlinear_bound = self._define_nonlinear_bounds(rad, center)
            self.max_sol = minimize(self._func_to_minimize, x0, method='SLSQP', bounds=nlinear_bound, constraints=[nlinear_constraint])
            # self.max_sol = minimize(self._func_to_minimize, x0, method='SLSQP', constraints=[nlinear_constraint])
            # print(f"self.max_sol.message = {self.max_sol.message}, sol = {self.max_sol.x}")
            if not self.max_sol.success:
                x0 = x0 + np.random.rand()*rad/100
            else:
                self.success = True
            
            iteration = iteration + 1    

        self.min_lambda = self.feval(self.max_sol.x)
        
        return self.max_sol.x, self.min_lambda
        
class ModelPolynomial:
    def __init__(self, symbol, feval:callable) -> None:
        
        self.symbol = ca.simplify(symbol)
        self.feval = feval
        
class Poisedness:
    def __init__(self, index, max_points, poisedness) -> None:
        ''' Class for containing information regarding poisedness
        index = index of the best point
        '''
        self.index = index # Index with the maximum poisedness
        self.max_points = max_points
        self.poisedness = poisedness

        self._validate_class()
        
    def _validate_class(self):
        assert np.nanmax(self.poisedness) == self.poisedness[self.index]
    
    def max_poisedness(self) -> float:
        return np.nanmax(self.poisedness)
    
    def point_to_max_poisedness(self) -> np.ndarray:
        return self.max_points[self.index]
    

class LagrangePolynomials:
    def __init__(self, input_symbols, pdegree:int = 2):
        
        ### Must exists
        self.pdegree = pdegree
        self.input_symbols = input_symbols
        
    def initialize(self, y:np.ndarray, f:Optional[np.ndarray] = None, sort_type:str='function', interpolation_type:str = 'frobenius', lpolynomials:List[LagrangePolynomial] = None, tr_radius:float = None):
        """ This class should be able to generate lagrange polynomials given the samples

        Args:
            y (np.ndarray): an N x (P+1) array in which each column j in P represent an interpolation point. 
                            N is the dimension of the vectors.
            f (np.ndarray): an N x 1 array such that f = M \alpha
            pdegree (int) : polynomial degree for the approximation
            
        Methods:
            .interpolate : Given a vector x it will interpolate using the polynomials constructed 
                           by this class
                           
        Attributes: 
            .lagrange_polynomials : All the lagrange polynomials, lambda_i(x), wrapped in LagrangePolynomial class
            .model_polynomial : Ultimate polynomial model for the given interpolation set, wrpaped in ModelPolynomial class
            .polynomial_basis : List of the polynomial basis, wrapped in PolynomialBase class
            .get_poisedness : Calculate (minimum) poisedness of the given set
            
        """
        
        ### TODO: DEBUG
        # shift to center for the input symbol s = (x - c)
        
        self.original_symbols = copy.copy(self.input_symbols)
        input_symbols = copy.copy(self.input_symbols)
        for i in range(y.shape[0]):
            input_symbols[i] = self.input_symbols[i] - y[i,0]
        
        self.sample_set = SampleSets(y=y, sort_type=sort_type, f=f)
        
        self.y = self.sample_set.y
        self.center = self.y[:,[0]]
        self.f = f
        if (self.f is not None):
            if (self.y.shape[1] != self.f.shape[0]):
                raise Exception("y and f do not have the same shape!")
        else:
            self.f = [None]*self.y.shape[0]
        
        if tr_radius is None:
            self.tr_radius = self.sample_set.ball.rad
        else:
            self.tr_radius = tr_radius
        
        self.N = y.shape[0]
        self.P = y.shape[1]
        
        self.multiindices = self._get_multiindices(self.N, self.pdegree)
        self.coefficients = self._get_coefficients(self.multiindices)
    
        # Possible improvement is to let user decide which basis they want, e.g. for underdetermined problems (Ch. 5)
        self.polynomial_basis, _ = self._get_polynomial_basis(self.multiindices, self.coefficients, input_symbols)
        
        if lpolynomials is None:
            if interpolation_type == 'minimum':
                # self.lagrange_polynomials = self._build_lagrange_polynomials(self.polynomial_basis, self.y, self.input_symbols)
                self.lagrange_polynomials_normalized = self._build_lagrange_polynomials(self.polynomial_basis, (self.y - self.center)/self.tr_radius, self.input_symbols)
            elif interpolation_type == 'frobenius':
                # frobenius norm can only be used when the number of data points are between n+1 <= p <= 1/2 (n+1)(n+2)
                
                if self.N + 1 <= self.P and self.P < 0.5*(self.N+1)*(self.N+2):
                    # self.lagrange_polynomials = self._build_lagrange_polynomials_frobenius(self.y, is_gen=False)
                    self.lagrange_polynomials_normalized = self._build_lagrange_polynomials_frobenius((self.y - self.center)/self.tr_radius, is_gen=False)
                else:
                    # raise Exception(f"Number of points must be between {self.N + 1} and {int(0.5*(self.N+1)*(self.N+2))}")
                    # print(f"Number of points should be between {self.N + 1} and {int(0.5*(self.N+1)*(self.N+2))}. Currently {self.P}")
                    # self.lagrange_polynomials = self._build_lagrange_polynomials(self.polynomial_basis, self.y, self.input_symbols)
                    self.lagrange_polynomials_normalized = self._build_lagrange_polynomials(self.polynomial_basis, (self.y - self.center)/self.tr_radius, self.input_symbols)

            else:
                raise Exception(f"Interpolation type of {interpolation_type} is not known. Try 'minimum' of 'frobenius'")
        else: # When lagrange polynomials is constructed manually
            # self.lagrange_polynomials = lpolynomials
            self.lagrange_polynomials_normalized = lpolynomials
            # self.lagrange_polynomials_normalized = self._build_lagrange_polynomials(self.polynomial_basis, (self.y - self.y[:,[0]])/self.tr_radius, self.input_symbols)
        
        # self.model_polynomial = self._build_model_polynomial(self.lagrange_polynomials, self.f, self.input_symbols)
        self.model_polynomial_normalized = self._build_model_polynomial(self.lagrange_polynomials_normalized, self.f, self.input_symbols)
        if self.model_polynomial_normalized is None:
            # self.gradient, self.Hessian = None, None
            self.gradient_normalized, self.Hessian_normalized = None, None
        else: 
            # self.gradient, self.Hessian = self._get_coefficients_from_expression(self.model_polynomial.symbol, self.input_symbols, self.pdegree)
            self.gradient_normalized, self.Hessian_normalized = self._get_coefficients_from_expression(self.model_polynomial_normalized.symbol, self.input_symbols, self.pdegree)    
        
        self.index_of_largest_lagrangian_norm = None
        
    
    def interpolate(self, x:np.ndarray) -> float:
        """Interpolate using the interpolation model given the input x. 
           Maps R^n -> R, where n is the dimension of input data

        Args:
            x (np.ndarray): Input data, R^n

        Returns:
            float: Interpolation value
        """
        # return self.model_polynomial.feval(x)
        return self.model_polynomial_normalized.feval((x - self.y[:,0])/self.tr_radius)
        
    def poisedness(self, rad=None, center=None) -> Poisedness:
        """Calculate the minimum poisedness given the set of interpolation points.'
           The poisedness is calculated as: 
           
                Lambda = max_{0 <= i <= p} max_{x \in B} |lambda_i(x)|

        Returns:
            int: index of lagrange polynomial with the largest poisedness
            float: Minimum poisedness of the given interpolation set
        """
        
        return self._get_poisedness(self.lagrange_polynomials_normalized, 1, np.zeros(center.shape))
        
    def _get_poisedness(self, lagrange_polynomials:List[LagrangePolynomial], rad:float, center:np.ndarray) -> Poisedness:
    
        Lambda = 0.0
        index = -1
        
        Lambdas = []
        max_sols = []
        
        i = 0
        for lp in lagrange_polynomials:
            max_sol, feval = lp._find_max_given_boundary(x0=center, rad=rad, center=center)        
            max_sols.append(max_sol)
            
            Lambdas.append(np.abs(feval))
            
            if np.abs(feval) > Lambda:
                Lambda = np.abs(feval)
                index = i
            
            i = i+1
        
        if float(Lambda) == 0:
            raise PoisednessIsZeroException(f"Poisedness (Lambda) is 0. Something is wrong.")
        
        return Poisedness(index, max_sols, Lambdas)
    
    def _get_coefficients_from_expression(self, expression, input_symbols, degree:int=2) -> Tuple[np.ndarray, np.ndarray]:
        """ The function responsible for approximating the gradient and Hessian

        Args:
            expression (_type_): _description_
            input_symbols (_type_): _description_
            degree (int, optional): _description_. Defaults to 2.

        Raises:
            Exception: _description_

        Returns:
            Tuple[np.ndarray, np.ndarray]: _description_
        """
        if degree == 2:

            gradient = ca.jacobian(expression, vertcat(input_symbols))
            Hessian = ca.jacobian(gradient, vertcat(input_symbols))
            Hessian = DM(Hessian).full()
            try:
                Hessian = nearestPD(Hessian)
            except np.linalg.LinAlgError:
                pass
                
            return (Function('gradient', [input_symbols], [gradient]), Hessian)
                
        elif degree == 1:
            
            gradient = ca.jacobian(expression, vertcat(input_symbols))
            return (Function('gradient', [input_symbols], [gradient]), None)
        
        else:
            raise Exception("Only pdegree of 1 or 2 are supported")

    def _is_close_to_zero(self, var:float, tol:float=10E-5):
        if np.abs(var) < tol:
            return 0.0
        else:
            return float(var)
    
    def _build_model_polynomial(self, lagrange_polynomials:List[LagrangePolynomial], f:np.ndarray, input_symbols:list) -> ModelPolynomial:
        """ Generate model polynomial, m(x), given the lagrange polynomials and the interpolation set value

            m(x) = sum(lambda_i(x)*f(y_i))

        Args:
            lagrange_polynomials (List[LagrangePolynomial]): List of lagrange polynomials
            f (np.ndarray): interpolation set value
            input_symbols (list): _description_

        Returns:
            ModelPolynomial: Model polynomial
        """

        if f[0] is None:
            return None
        else:            
            polynomial_sum = 0
            i = 0
            for poly in list(lagrange_polynomials):
                polynomial_sum += poly.symbol*f[i]
                i += 1
            
            return ModelPolynomial(polynomial_sum, Function(f'm_f', [input_symbols], [polynomial_sum]))
    
    def _build_lagrange_polynomials(self, basis:List[PolynomialBase], data_points:np.ndarray, input_symbols:list) -> List[LagrangePolynomial]:
        """ Responsible for generating the lagrange polynomials using Cramer's rule: 
            lambda(x) = det(M(Y_i, \phi))/det(M(Y, \phi))

            where Y_i = Y \ y_i \cup x

        Args:
            basis (List[PolynomialBase]): List of polynomial base
            data_points (np.ndarray): matrix of interpolation set input Y {y1, y2, ..., yp}
            input_symbols (list): list of sympy symbol for the input X = {x1, x2, ..., xn}

        Returns:
            List[LagrangePolynomial]: Lagrange polynomials, {lambda_0, lambda_1, ..., lambda_p}
        """

        # Create the full matrix
        matrix = []
        for i in range(data_points.shape[1]):
            inp = data_points[:,i]
            entry = [float(d.feval(inp)) for d in basis]
            matrix.append(entry)
            
        basis_matrix = np.array(matrix)
        basis_vector = vertcat(*[d.symbol for d in basis])
        
        lagrange_matrix = mtimes(np.matmul(basis_matrix,(np.linalg.pinv(np.matmul(basis_matrix.T,basis_matrix)))), basis_vector) # Eq (4.7)
        
        lpolynomials = []
        for i in range(lagrange_matrix.shape[0]):
            lpolynomial = lagrange_matrix[i, :][0]
            f_lpolynomial = Function(f'lambda_{i}', [input_symbols], [lpolynomial])
            
            lpolynomials.append(LagrangePolynomial(lpolynomial, f_lpolynomial))

        return lpolynomials
    
    def _build_lagrange_polynomials_frobenius(self, data_points:np.ndarray, is_gen:bool=False) -> List[LagrangePolynomial]:
        """ Responsible for generating the lagrange polynomials in Frobenius norm sense. Read chapter 5 of Conn's book.

        Args:
            basis (List[PolynomialBase]): List of polynomial base
            data_points (np.ndarray): matrix of interpolation set input Y {y1, y2, ..., yp}

        Returns:
            List[LagrangePolynomial]: Lagrange polynomials, {lambda_0, lambda_1, ..., lambda_p}
        """

        basis = self.polynomial_basis
        
        # Create the full matrix)
        matrix = []
        for i in range(data_points.shape[1]):
            inp = data_points[:,i]
            entry = [float(d.feval(inp)) for d in basis]
            matrix.append(entry)
            
        basis_matrix = np.array(matrix)
        basis_vector = vertcat(*[d.symbol for d in basis])
        
        ## Build lagrange polynomials based on Frobenius norm
        # Divide between linear and quadratic terms
        n = data_points.shape[0]
        basis_matrix_linear = basis_matrix[:, :n+1]
        basis_matrix_quadratic = basis_matrix[:, n+1:]
        
        basis_vector_linear = basis_vector[:n+1, :]
        basis_vector_quadratic = basis_vector[n+1:, :]

        # Build matrix F (eq 5.7)
        matrix_A = mtimes(basis_matrix_quadratic, basis_matrix_quadratic.T)
        matrix_F = vertcat(horzcat(matrix_A, basis_matrix_linear), horzcat(basis_matrix_linear.T, 0*SX.eye(basis_matrix_linear.shape[1])))
        try:
            matrix_F_inv = ca.inv(matrix_F)
        except:
            raise Exception(f"Matrix F is non-singular.")
        
        # Construct (Eq 5.9) F x L = B
        matrix_B = vertcat(mtimes(basis_matrix_quadratic,basis_vector_quadratic), basis_vector_linear)
        solution_matrix = mtimes(matrix_F_inv,matrix_B)
    
        polys = [solution_matrix[i, :][0] for i in range(matrix_A.shape[0])]
        # lpolynomials = map(functools.partial(lagrange, input_symbols=self.input_symbols), polys, range(matrix_A.shape[0])) #faster than for loop
        lpolynomials = map(functools.partial(lagrange, input_symbols=self.original_symbols), polys, range(matrix_A.shape[0])) #faster than for loop
        
        if is_gen:
            return lpolynomials
        else:
            return list(lpolynomials)
    
    def _get_polynomial_basis(self, exponents:list, coefficients:list, input_symbols) -> Tuple[List[PolynomialBase], list]:
        """Responsible for generating all the polynomial basis, also returning input_symbols

        Args:
            exponents (list): combination of all possible exponents
            coefficients (list): combination of all possible coefficients

        Returns:
            Tuple[List[PolynomialBase], list]: List of polynomial bases and input symbols
        """

        basis = []
        for (exps, coef) in zip(exponents, coefficients):
            b = 1
            for symb, exp in zip(input_symbols.nz, exps):
                b *= symb**exp
            b = b/coef

            # (phi(x) sympy symbol, phi(x) evaluation), function evaluation called using a list x, phi(x)
            # basis.append(PolynomialBase(b, Function('phi_b', [self.input_symbols], [b])))
            basis.append(PolynomialBase(b, Function('phi_b', [self.original_symbols], [b])))

        return basis, self.input_symbols

    def _get_coefficients(self, multiindices:list) -> list:
        """Generate coefficients for every possible polynomial basis

        Args:
            multiindices (list): list of indices that will be used as the reciprocal in the coefficient calculation

        Returns:
            list: all possible combination of the coefficients
        """
        coefficients = [np.prod(scipy.special.factorial(ind)) for ind in multiindices]
        return coefficients
        
    def _get_multiindices(self, n:int, d:int) -> list:
        """ This function is responsible to get the multiindices to build the polynomial basis

        Args:
            n (int): dimension of the data 
            d (int): degree of polynomials
            p (int): number of data points

        Returns:
            list: all possible combination of the multiindices, Combination(n + d, d)
        """
        range_tuples = range(d + 1)
        multiindices = product(range_tuples, repeat=n, limit=d+1)
        # multiindices = [i for i in multiindices]
        
        # Make sure that the order of the polynomial basis is based on the level of expansion
        multiindices.sort(key= lambda x: np.sum(x))
        
        return multiindices 
         
    def _product(self, *argument):
        return itertools.product(*argument)
    
    def _check_lagrange_polynomials(self):
        """ Sanity check on the lagrange polynomial condition. Testing the kroenecker delta condition.
        """
        
        ## TODO: CHECK IF THE DATASET UPDATE AND POLYNOMIALS ARE ONE TO ONE
        for i, polynomial in enumerate(self.lagrange_polynomials):
            for j in range(self.y.shape[1]):
                eval = polynomial.feval(self.y[:, j])
                
                if i == j:
                    if np.abs(eval - 1) <= 10E-5:
                        pass
                    else:
                        poisedness = self.poisedness().max_poisedness()
                        raise Exception(f"i, j = {i}, {j} but l_i(y_j) = {eval}. Set poisedness = {poisedness}")
                else:
                    if np.abs(eval) <= 10E-5:
                        pass
                    else:
                        poisedness = self.poisedness().max_poisedness()
                        raise Exception(f"i, j = {i}, {j} but l_i(y_j) = {eval}. Set poisedness = {poisedness}")    