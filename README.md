# Trust-Region-Method

This repository contains a trust-region method for black-box optimization problems with the handling of output constraints. The algorithm used is the SQP-filter with a few modifications, providing an efficient approach for solving optimization problems where the objective function and constraints are not explicitly known but can be evaluated through a black-box function.

The SQP-filter algorithm employed in this repository is a variant of the Sequential Quadratic Programming (SQP) method. It incorporates a filter mechanism to handle constraints, allowing for feasible and improved solutions.

To solve the TRQP subproblem, the repository utilizes IPOPT, a powerful optimization solver. IPOPT (Interior Point OPTimizer) is known for its effectiveness in solving large-scale nonlinear optimization problems, including constrained optimization.

## Installation

Make sure you have a compatible Python environment set up before installing the dependencies.

First, install the dependencies: 

```shell
pip install numpy
pip install scipy
pip install casadi
pip install matplotlib
```

To install the Trust-Region-Method package, run the following command:

```shell
pip install git+https://github.com/iffanh/py-trsqp.git
```

## Examples

The `examples` folder in this repository contains notebooks that demonstrate the usage of the Trust-Region-Method package. These notebooks showcase various optimization problems and illustrate how to apply the trust-region method with output constraints to solve them. To explore the examples, navigate to the `examples` folder and run the notebooks using a Jupyter environment.

## Reference

If you're interested in learning more about trust-region methods, you can refer to the following reference:

- Conn, A. R., Gould, N. I. M., & Toint, Ph. L. (2000). Trust-Region Methods. SIAM. [https://doi.org/10.1137/1.9780898719857](https://doi.org/10.1137/1.9780898719857)

For more information about IPOPT, the optimization solver used in this repository, you can visit the official IPOPT documentation:

- IPOPT: Interior Point OPTimizer. Retrieved from [https://coin-or.github.io/Ipopt/](https://coin-or.github.io/Ipopt/)