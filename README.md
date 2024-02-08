# py-trsqp: Python Implementation of Sequential Quadratic Programming Trust-Region Filter Algorithm 

This repository contains a trust-region method for black-box optimization problems with the handling of output constraints. The algorithm used is the SQP-filter with a few modifications, providing an efficient approach for solving optimization problems where the objective function and constraints are not explicitly known but can be evaluated through a black-box function.

The SQP-filter algorithm employed in this repository is a variant of the Sequential Quadratic Programming (SQP) method. It incorporates a filter mechanism to handle constraints, allowing for feasible and improved solutions.

To solve the TRQP subproblem, the repository utilizes IPOPT, a powerful optimization solver. IPOPT (Interior Point OPTimizer) is known for its effectiveness in solving large-scale nonlinear optimization problems, including constrained optimization.

The optimization takes the following form

$$ \min_x f(x) $$

$$ c_i \geq 0, \quad i \in \mathcal{I} $$

$$ c_i = 0, \quad i \in \mathcal{E} $$

## Installation

Make sure you have a compatible Python environment set up before installing the dependencies.

First, install the dependencies: 

```shell
pip install numpy
pip install scipy
pip install casadi
pip install matplotlib
```

To install the `py-trsqp` package, run the following command:

```shell
pip install git+https://github.com/iffanh/py-trsqp.git
```

## Examples

Here is a simple usage of the package.
```python 
import py_trsqp.trsqp as tq
import numpy as np

def cf(x):
    return 10*(x[0]-1.0)**2 + 4*(x[1]-1.0)**2

def ineq(x):
    return x[0] - x[1] - 1

tr = tq.TrustRegionSQPFilter(x0=np.array([-1.5,-1.0]).T,    # initial points
                             cf=cf,                         # objective function
                             lb=[-5,-5],                    # lower bound
                             ub=[5,5],                      # upper bound
                             eqcs=[],                       # equality constraints
                             ineqcs=[ineq])                 # inequality constraints

tr.optimize(max_iter=20)
```



The `examples` folder in this repository contains notebooks that demonstrate the usage of the this package. These notebooks showcase various optimization problems and illustrate how to apply the trust-region method with output constraints to solve them. To explore the examples, navigate to the `examples` folder and run the notebooks using a Jupyter environment.


## Options
The options for the algorithm can be set using the `constants` as a dictionary. The available options are:

| Option   | Description | Default |
| -------- | ----------- | ----- |
| max_points |  Maximum number of interpolation points used in an iteration. <br> The value must be between $n+1$ and $\frac{1}{2}(n+1)(n+2)$   | $\frac{1}{2}(n+1)(n+2)$ |
| budget |  Maximum number of (different) function calls   | $1000$ |

## Hyperparameters

The hyperparameters for the algorithm is defined by the `constants` as a dictionary. The hyperparameters are set by changing the dictionary keys:

| Hyperparameter   | Symbol | Description | Value |
| ---------------- | ------ | ----------- | ----- |
| gamma_0          | $\gamma_0$ | radius multiplier factor when $\rho < \eta_1$ | $0 < \gamma_0 < \gamma_1$ |
| gamma_1          | $\gamma_1$ | radius multiplier factor when $\eta_1 \leq \rho < \eta_2$ | $\gamma_0 < \gamma_1 < 1$ |
| gamma_2          | $\gamma_2$ | radius multiplier factor when $\eta_2 \leq \rho$ | $1 < \gamma_2$ |
| eta_1            | $\eta_1$   | ratio 1 threshold | $0 < \eta_1 < \eta_2$ |
| eta_1            | $\eta_1$   | ratio 1 threshold | $\eta_1 < \eta_2 < 1$ |


The decision variables will be transformed into a normalized space. Note that the initial radius and the stopping radius is applied to this normalized space.

## Paper
The following paper(s) use this package:
1. Hannanu, M., Silva, T. L., Camponogara, E., and M. Hovd. "Well Control Optimization with Output Constraint Handling by Means of a Derivative-Free Trust Region Algorithm." Paper presented at the ADIPEC, Abu Dhabi, UAE, October 2023. https://doi.org/10.2118/216962-MS

## Reference

If you're interested in learning more about trust-region methods, you can refer to the following reference:

- Conn, A. R., Gould, N. I. M., & Toint, Ph. L. (2000). Trust-Region Methods. SIAM. [https://doi.org/10.1137/1.9780898719857](https://doi.org/10.1137/1.9780898719857)

For more information about IPOPT, the optimization solver used in this repository, you can visit the official IPOPT documentation:

- IPOPT: Interior Point OPTimizer. Retrieved from [https://coin-or.github.io/Ipopt/](https://coin-or.github.io/Ipopt/)
