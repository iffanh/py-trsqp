# py-trsqp: Derivative-Free Trust-Region Filter (DFTR-Filter) for Output-Constrained Optimization

This repository implements DFTR-Filter, a derivative-free model-based trust-region method with a filter mechanism for black-box optimization problems with output constraints. It combines the interpolation-based derivative-free trust-region theory of Conn, Scheinberg & Vicente (2009) with the SQP-filter trust-region theory of Conn, Gould & Toint (2000) and the filter mechanism of Fletcher & Leyffer (2002); see `dftrf_v6.pdf` in this repository for the full algorithm statement and convergence proof (Algorithm 5). Objective and constraints only need to be evaluable through a black-box function -- no derivatives are required. Each iteration builds a quadratic (or linear) interpolation model, takes a step via a QCQP subproblem with a Cauchy-domination safeguard, and judges the step against a filter, falling back to a dedicated restoration phase whenever the linearised constraints are locally infeasible.

Model building (interpolation, Lagrange polynomials, poisedness) is implemented in plain NumPy/SciPy for speed and numerical robustness; CasADi/IPOPT is used only to solve the constrained QCQP and restoration subproblems.

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
from py_trsqp import DFTRFilter

def cf(x):
    return 10*(x[0]-1.0)**2 + 4*(x[1]-1.0)**2

def ineq(x):
    return x[0] - x[1] - 1

opt = DFTRFilter(x0=[-1.5, -1.0],   # initial point
                  cf=cf,            # objective function
                  lb=[-5, -5],      # lower bound (must be finite)
                  ub=[5, 5],        # upper bound (must be finite)
                  eqcs=[],          # equality constraints
                  ineqcs=[ineq])    # inequality constraints

opt.optimize(max_iter=20)
print(opt.best.x, opt.best.f, opt.termination_status)
```

> **Note:** the public API changed in v0.2.0 (the derivative-free trust-region
> filter method, "DFTR-Filter", replacing the earlier
> `py_trsqp.trsqp.TrustRegionSQPFilter` heuristic). The `examples/` notebooks
> still show the old interface and need updating to `py_trsqp.DFTRFilter`
> before they will run again.



The `examples` folder in this repository contains notebooks that demonstrate the usage of the this package. These notebooks showcase various optimization problems and illustrate how to apply the trust-region method with output constraints to solve them. To explore the examples, navigate to the `examples` folder and run the notebooks using a Jupyter environment.


## Options
The options for the algorithm can be set using the `constants` dictionary passed to `DFTRFilter(...)`, or the `budget`/`max_iter` arguments to `.optimize()`. See `py_trsqp.algorithm.Constants` for the full, authoritative list and defaults. Notable ones:

| Option   | Description | Default |
| -------- | ----------- | ----- |
| max_points |  Maximum number of interpolation points used in an iteration. <br> The value must be between $n+1$ and $\frac{1}{2}(n+1)(n+2)$   | $\min(2n+1, \frac{1}{2}(n+1)(n+2))$ |
| budget (arg to `.optimize()`) |  Maximum number of (different) function calls   | $10 \times \text{max\_iter} \times (n+1)$ |

## Hyperparameters

The hyperparameters are set via the `constants` dictionary, matching the paper's notation:

| Hyperparameter   | Symbol | Description | Default |
| ---------------- | ------ | ----------- | ----- |
| gamma0          | $\gamma_0$ | radius shrink factor on a rejected/model-improving step (Rule M) | $0.5$ |
| gamma1          | $\gamma_1$ | radius growth factor on an accepted step | $2.0$ |
| eta2            | $\eta_2$   | ratio-test acceptance threshold for f-iterations | $0.75$ |
| kappa_theta, s_theta | $\kappa_\theta, s_\theta$ | switching condition (f- vs h-iteration) | $10^{-3}, 1.1$ |
| gamma_vartheta  | $\gamma_\vartheta$ | filter margin | $10^{-4}$ |
| mu, omega, lambda_th, epsilon_c | $\mu, \omega, \Lambda_{th}, \epsilon_c$ | criticality step (Crit) | $100, 0.5, 10, 10^{-3}$ |
| tau             | $\tau$ | Cauchy-step backtracking factor | $0.5$ |
| init_radius, max_radius, min_radius | $\Delta_0, \Delta_{max}, \Delta_{min}$ | trust-region radius bounds | $1.0, 2.0, 10^{-8}$ |
| epsilon_tol     | $\epsilon_{tol}$ | approximate-KKT stopping tolerance | $10^{-6}$ |

Decision variables are transformed into a normalized space (each coordinate mapped affinely so `[lb, ub]` becomes `[-1, 1]`); the initial/stopping radius are in that normalized space. Bounds must therefore be finite.


The decision variables will be transformed into a normalized space. Note that the initial radius and the stopping radius is applied to this normalized space.

## Paper
The following paper(s) use this package:
1. Hannanu, M., Silva, T. L., Camponogara, E., and M. Hovd. "Well Control Optimization with Output Constraint Handling by Means of a Derivative-Free Trust Region Algorithm." Paper presented at the ADIPEC, Abu Dhabi, UAE, October 2023. https://doi.org/10.2118/216962-MS
2. Hannanu, M.I., Camponogara, E., Silva, T.L. et al. A modified derivative-free SQP-filter trust-region method for uncertainty handling: application in gas-lift optimization. Optim Eng (2024). https://doi.org/10.1007/s11081-024-09909-0 
3. Hannanu, M., Silva, T. L., Camponogara, E., and M. Hovd. "Derivative-Free Well Control Optimization under Uncertainty." Paper presented at the ECMOR, Oslo, Norway, September 2024. https://doi.org/10.3997/2214-4609.202437013 

## Reference

If you're interested in learning more about trust-region methods, you can refer to the following reference:

- Conn, A. R., Gould, N. I. M., & Toint, Ph. L. (2000). Trust-Region Methods. SIAM. [https://doi.org/10.1137/1.9780898719857](https://doi.org/10.1137/1.9780898719857)

For more information about IPOPT, the optimization solver used in this repository, you can visit the official IPOPT documentation:

- IPOPT: Interior Point OPTimizer. Retrieved from [https://coin-or.github.io/Ipopt/](https://coin-or.github.io/Ipopt/)
