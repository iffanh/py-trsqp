""" File to run when running benchmark studies 
    run with python3 -m py_trsqp.test.benchmark
"""

import py_trsqp.test.benchmark_unconstrained as pu
import py_trsqp.test.benchmark_linear as pl


if __name__ == '__main__':
    pu.run(n_max=10)
    pl.run(n_max=10, n_cons_max=50)