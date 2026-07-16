#!/bin/bash
# Source this to set up the CUTEst/pycutest environment for benchmarks/cutest_benchmark.py
# and tests/test_cutest_benchmark.py. See DEVELOPMENT.md, "CUTEst benchmark setup", for how
# these directories are built. Paths below match that guide's no-root/apt (micromamba) setup;
# adjust them if you installed CUTEst elsewhere (e.g. via apt + /usr/local).

export PATH=~/opt/cutest-toolchain/bin:$PATH
export ARCHDEFS=~/opt/cutest-src/ARCHDefs/
export SIFDECODE=~/opt/cutest-src/SIFDecode/
export CUTEST=~/opt/cutest-src/CUTEst/
export MASTSIF=~/opt/cutest-src/mastsif/
export MYARCH=pc64.lnx.gfo
export PYCUTEST_CACHE=~/opt/pycutest_cache

# Only needed if your Python has no matching system compiler/dev headers
# (see DEVELOPMENT.md); omit these three on a normal apt-based machine.
export CC=~/opt/cutest-toolchain/bin/gcc
export CXX=~/opt/cutest-toolchain/bin/g++
export CPATH=~/opt/py312-dev/include/python3.12
