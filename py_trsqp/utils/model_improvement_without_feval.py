from .lagrange_polynomial import LagrangePolynomials, LagrangePolynomial
import numpy as np
import casadi as ca
from typing import Tuple
import functools
from casadi import Function
import time

def generate_spanning_set(k, dim):

    print(f"generate positive spanning set k = {k}, dim = {dim}")
    arr = np.zeros((dim, dim))
    for i in range(dim):
        arr[:,i] = -1/dim
        arr[i,i] = 1

    C = np.linalg.cholesky(arr).T
    vn = 0 
    for i in range(dim):
        v = C[:,i]
        vn -= v
        
    Y = np.concatenate([C.T, [vn]]).T #dim + 1
    
    if k > dim + 2 and k <= 2*dim + 1:
        for i in range(k - dim - 2):
            _v = -Y[:,[i]]
            Y = np.concatenate([Y, _v], axis=1)
    elif k <= dim + 1:
        Y = Y[:, :k-1]
    else:
        for i in range(1, dim):
            _v = -Y[:,[i]]
            Y = np.concatenate([Y, _v], axis=1) # 2dim + 1
        
    Y = np.concatenate((np.zeros((dim, 1)), Y), axis=1)
    
    return Y

def _generate_uniform_sample_nsphere(k:int, d:int):
    RNG = np.random.default_rng(seed=12345)
    samples = RNG.normal(loc=0.0, scale=1.0, size=(d, k))
    for i in range(samples.shape[1]):
        sample = samples[:,i]
        d = np.linalg.norm(sample)
        sample = sample/d
        samples[:,i] = sample
    return samples

@functools.lru_cache() 
def generate_uniform_sample_nsphere(k:int, d:int, L:float=1.0):
    
    if k <= 2*d + 1:
        return generate_spanning_set(k, d)
    
    print(f"Generating {k}-uniform points on the {d}-sphere ...")
    input_symbols = ca.SX.sym('x', d)
    samples = _generate_uniform_sample_nsphere(k-1, d)

    lpolynomials = LagrangePolynomials(input_symbols=input_symbols, pdegree=2)
    lpolynomials.initialize(y=samples, tr_radius=1.0)
    
    # Gets too slow for k > 10    
    # TODO: How to accelerate this
    # if k < 10:
    for i in range(k):
        # start = time.time()
        poisedness = lpolynomials.poisedness(rad=1.0, center=np.array([0.0]*d))
        curr_lambda = poisedness.max_poisedness()

        if curr_lambda < L:
            # enough
            new_y = lpolynomials.y*1
            break
        
        pindex = poisedness.index
        new_point = poisedness.point_to_max_poisedness()
        
        # copy values
        new_y = lpolynomials.y*1
                 
        # replace value
        new_y[:, pindex] = new_point
        lpoly = lpolynomials.lagrange_polynomials[pindex]
        
        ## Algorithm 6.1
        if lpoly.feval(new_point) == 0:
            raise Exception("Problem here")
        
        new_lpoly = lpoly.symbol/lpoly.feval(new_point) #(6.9)
        
        # update lagrange polynomial
        new_lpolynomials = []
        for j, _lpoly in enumerate(lpolynomials.lagrange_polynomials):
            if j == pindex:
                function = Function(f'lambda_{i}', [input_symbols], [new_lpoly])                         
                new_lpolynomials.append(LagrangePolynomial(new_lpoly, function))
                # continue 
            
            else:
            
                _feval = Function(f'lambda_{i}', [input_symbols], [_lpoly.symbol])
                _new_lpoly = _lpoly.symbol - _feval(new_point)*new_lpoly #(6.10)
                
                function = Function(f'lambda_{i}', [input_symbols], [_new_lpoly]) 
                new_lpolynomials.append(LagrangePolynomial(_new_lpoly, function))
        
        # create polynomials
        lpolynomials = LagrangePolynomials(input_symbols=input_symbols, pdegree=2)
        lpolynomials.initialize(y=new_y, f=None, tr_radius=1.0, lpolynomials=new_lpolynomials)
        
    new_y = np.concatenate((np.zeros((d, 1)), new_y), axis=1)

    return new_y

class ModelImprovement:
    """ Class that responsible for improving the lagrange polynomial models based on the poisedness of set Y. 
    
    """
    def __init__(self, input_symbols) -> None:
        self.input_symbols = input_symbols
        pass
    
    def improve_model(self, lpolynomials:LagrangePolynomials, rad:float, center:np.ndarray, L:float=1.0, max_iter:int=5, sort_type='function') -> Tuple[LagrangePolynomials, dict]:
        """ The function responsible for improving the poisedness of set Y in lagrange polynomial. 
        It follows from Algorithm 6.3 in Conn's book.

        Args:
            lpolynomials (LagrangePolynomials): LagrangePolynomials object to be improved
            func (callable): function call to evaluate the new points
            L (float, optional): Maximum poisedness in the new LagrangePolynomial object. Defaults to 100.0.
            max_iter (int, optional): Number of loops to get to the improved poisedness. Defaults to 5.

        Returns:
            LagrangePolynomials: New LagrangePolynomial object with improved poisedness
        """
        from casadi import Function
        
        for k in range(max_iter):
            # Algorithm 6.3
            poisedness = lpolynomials.poisedness(rad=rad, center=center)
            Lambda = poisedness.max_poisedness()

            ## TODO: Any ideas on how to circumvent the replacement of the best point?
            pindex = poisedness.index
            
            if pindex == 0:
            # if False:
            
                if np.abs(L - Lambda) <= 1E-5: # it's fine if it's very close
                # if False:
                    best_polynomial = lpolynomials
                    curr_Lambda = Lambda*1
                    break
                else:
                    tr_radius = lpolynomials.tr_radius*1
                    new_y = lpolynomials.y*1
                    
                    surface_points = generate_uniform_sample_nsphere(k=new_y.shape[1], d=new_y.shape[0], L=L)
                    new_y = center[:, np.newaxis] + surface_points*tr_radius
                    
                    lpolynomials = LagrangePolynomials(input_symbols=self.input_symbols, pdegree=1)
                    lpolynomials.initialize(y=new_y, f=None, sort_type=sort_type, tr_radius=tr_radius)   

                    best_polynomial = lpolynomials
                    curr_Lambda = Lambda*1
                    break
            else:
                if k == 0:
                    best_polynomial = lpolynomials
                    curr_Lambda = Lambda*1

                # main loop
                if Lambda > L:
                    
                    new_point = poisedness.point_to_max_poisedness()
                    new_point = new_point*lpolynomials.tr_radius + center
                    
                    # copy values
                    new_y = lpolynomials.y*1
                    tr_radius = lpolynomials.tr_radius*1
            
                    # replace value
                    new_y[:, pindex] = new_point
                    lpoly = lpolynomials.lagrange_polynomials_normalized[pindex]
                    
                    ## Algorithm 6.1
                    if lpoly.feval(new_point) == 0:
                        raise Exception("Problem here")
                    
                    new_lpoly = lpoly.symbol/lpoly.feval(new_point) #(6.9)
                    
                    # update lagrange polynomial
                    new_lpolynomials = []
                    for j, _lpoly in enumerate(lpolynomials.lagrange_polynomials_normalized):
                        
                        if j == pindex:
                            function = Function(f'lambda_{j}', [self.input_symbols], [new_lpoly])                         
                            new_lpolynomials.append(LagrangePolynomial(new_lpoly, function))
                        
                        else:
                        
                            _feval = Function(f'lambda_{j}', [self.input_symbols], [_lpoly.symbol])
                            _new_lpoly = _lpoly.symbol - _feval(new_point)*new_lpoly #(6.10)
                            
                            function = Function(f'lambda_{j}', [self.input_symbols], [_new_lpoly]) 
                            new_lpolynomials.append(LagrangePolynomial(_new_lpoly, function))
                    
                    # create polynomials
                    lpolynomials = LagrangePolynomials(input_symbols=self.input_symbols, pdegree=2)
                    lpolynomials.initialize(y=new_y, f=None, tr_radius=tr_radius, lpolynomials=new_lpolynomials)
                    
                    poisedness = lpolynomials.poisedness(rad=rad, center=center)
                    Lambda = poisedness.max_poisedness()
                    
                    # save polynomial with the smallest poisedness
                    if Lambda < curr_Lambda:

                        curr_Lambda = Lambda*1

                        if curr_Lambda < L:
                            return lpolynomials
            
            if k == max_iter-1:
                print(f"Could not construct polynomials with poisedness < {L} after {max_iter} iterations.") 
                
        return best_polynomial