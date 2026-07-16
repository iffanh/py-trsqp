# Development setup

## Creating the virtual environment (WSL / Ubuntu without `python3.12-venv`)

On this machine the system Python 3.12 ships **without `ensurepip`**, and the
`python3.12-venv` apt package is not installable (no install candidate, and no
sudo). Plain `python3 -m venv .venv` therefore fails with:

```
ensurepip is not available ... you need to install the python3.12-venv package
```

Work around it by creating the venv **without pip**, then bootstrapping pip
into it with `get-pip.py`. No apt, no root required (needs internet, since
`get-pip.py` downloads pip from PyPI):

```bash
cd ~/playground/py-trsqp

# 1. remove any old venv
rm -rf .venv

# 2. create a venv WITHOUT pip (sidesteps the missing ensurepip)
python3 -m venv --without-pip .venv

# 3. activate
source .venv/bin/activate

# 4. bootstrap pip into the venv
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python /tmp/get-pip.py

# 5. install dependencies
pip install -r requirements.txt
```

## Running the tests

```bash
source .venv/bin/activate
python -m pytest tests/ -q
```

## Jupyter kernel for the example notebooks

Deleting/recreating `.venv` breaks any previously registered kernel. Re-register
it (the notebooks under `examples/` use it):

```bash
source .venv/bin/activate
pip install ipykernel
python -m ipykernel install --user --name py-trsqp \
    --display-name "Python (py-trsqp .venv)"
```

The kernelspec points at `.venv/bin/python` by absolute path, so it keeps
working across venv recreations **as long as the repo stays at
`~/playground/py-trsqp`**. In VSCode you can instead just pick `.venv/bin/python`
directly via the notebook kernel picker / "Python: Select Interpreter" (needs
the WSL + Jupyter extensions, with VSCode connected to WSL).

## Notes

- `--without-pip` works because the only thing `python3.12-venv` provides that
  we need is `ensurepip`; `get-pip.py` replaces that step.
- If you ever have `python3-full` / `python3.12-venv` available (e.g. another
  machine), plain `python3 -m venv .venv` works normally and you can skip the
  bootstrap.
- Running the notebooks headless (e.g. to verify them) additionally needs
  `pip install nbconvert ipykernel`; these are dev-only and intentionally not
  in `requirements.txt`.

## CUTEst benchmark setup

`benchmarks/cutest_benchmark.py` and `tests/test_cutest_benchmark.py` exercise
DFTR-Filter against the [CUTEst](https://github.com/ralna/CUTEst) test problem
collection via [pycutest](https://github.com/jfowkes/pycutest). This needs a
compiled CUTEst/SIFDecode toolchain plus the MASTSIF problem repository, none
of which are pulled in by `requirements.txt` (they aren't Python packages, and
the pytest test skips itself when they're absent, so nothing else depends on
this). If the required env vars aren't set, both scripts skip/no-op cleanly.

### With root + apt (normal case)

```bash
sudo apt install gfortran cmake make
mkdir ~/cutest && cd ~/cutest
git clone --depth 1 https://github.com/ralna/ARCHDefs ./archdefs
git clone --depth 1 https://github.com/ralna/SIFDecode ./sifdecode
git clone --depth 1 https://github.com/ralna/CUTEst ./cutest
git clone --depth 1 https://bitbucket.org/optrove/sif ./mastsif

export ARCHDEFS=~/cutest/archdefs/
export SIFDECODE=~/cutest/sifdecode/
export CUTEST=~/cutest/cutest/
export MASTSIF=~/cutest/mastsif/
export MYARCH=pc64.lnx.gfo

bash -c "$(curl -fsSL https://raw.githubusercontent.com/jfowkes/pycutest/master/.install_cutest.sh)"

pip install pycutest
```

### Without root/apt (this sandbox's setup)

No `gfortran`, `cmake`, `make`, or `python3-dev` are installed here, and apt
needs root. Everything below is installed **outside the repo**, in `~/opt`,
and is fully reversible by deleting those directories - it never touches
system packages.

1. **Build toolchain via micromamba** (a standalone, no-root package manager):
   ```bash
   curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest -o micromamba.tar.bz2
   python3 -c "import tarfile; tarfile.open('micromamba.tar.bz2','r:bz2').extract('bin/micromamba','.')"
   ./bin/micromamba create -y -p ~/opt/cutest-toolchain -c conda-forge gfortran cmake make gcc gxx
   ```

2. **Clone ARCHDefs/SIFDecode/CUTEst/MASTSIF** as above, but under `~/opt/cutest-src/`.

3. **Patch `ARCHDefs/system.lnx`**: it hardcodes `MAKE="/usr/bin/make"`, which
   doesn't exist on a system with no build-essential. Point it at the conda
   one instead: `MAKE="~/opt/cutest-toolchain/bin/make"` (absolute path).
   Also `mkdir -p $SIFDECODE/versions $CUTEST/versions` - the install script
   assumes these directories already exist, but a fresh clone doesn't have them.

4. **Run the CI-flavoured install script** (non-interactive, no sudo) with
   `PATH`, `ARCHDEFS`, `SIFDECODE`, `CUTEST`, `MASTSIF`, `MYARCH` set as above:
   ```bash
   bash ~/opt/install_cutest.sh   # downloaded from the .install_cutest.sh URL above
   ```

5. **Install pycutest, and get Python.h from conda-forge** (the system Python
   here has no dev headers, and there's no apt access to add them):
   ```bash
   micromamba create -y -p ~/opt/py312-dev -c conda-forge python=3.12
   source .venv/bin/activate && pip install pycutest
   ```

6. **Every run** needs these environment variables (put them in a sourceable
   `scripts/cutest_env.sh` for convenience):
   ```bash
   export PATH=~/opt/cutest-toolchain/bin:$PATH
   export ARCHDEFS=~/opt/cutest-src/ARCHDefs/
   export SIFDECODE=~/opt/cutest-src/SIFDecode/
   export CUTEST=~/opt/cutest-src/CUTEst/
   export MASTSIF=~/opt/cutest-src/mastsif/
   export MYARCH=pc64.lnx.gfo
   export PYCUTEST_CACHE=~/opt/pycutest_cache
   export CC=~/opt/cutest-toolchain/bin/gcc
   export CXX=~/opt/cutest-toolchain/bin/g++
   export CPATH=~/opt/py312-dev/include/python3.12
   ```
   (`CC`/`CXX`/`CPATH` are only needed because this Python has no matching
   system compiler/headers; on a normal machine with `python3-dev` installed,
   plain `pip install pycutest` needs none of this.)

### Running the benchmark

```bash
source scripts/cutest_env.sh   # or export the vars above manually
python -m pytest tests/test_cutest_benchmark.py -v      # fast regression smoke test (~7 problems)
python benchmarks/cutest_benchmark.py --n-max 20 --count 40 --csv results.csv   # broader sweep + metrics
```

DFTR-Filter is derivative-free and builds quadratic interpolation models, so
per-iteration cost grows like O(n^2); `--n-max` keeps the sweep to problems the
method is actually designed for. `benchmarks/cutest_benchmark.py` runs each
problem in its own subprocess with a wall-clock `--timeout`, so one
pathological problem can't hang the whole sweep. Each CUTEst problem is
compared against a gradient-based SciPy reference (SLSQP/L-BFGS-B, using
CUTEst's exact derivatives - DFTR-Filter itself never sees them).

Two further scripts compare DFTR-Filter against other derivative-free (not
gradient-based) solvers on the same problems, rather than a single
gradient-based reference:

```bash
# DFTR-Filter vs. Nelder-Mead/COBYLA on a given list of CUTEst problems
python benchmarks/compare_dfo_baselines.py --problems AIRCRFTA BEALENE ...

# DFTR-Filter vs. the best of 5 alternatives (COBYLA, COBYQA, Py-BOBYQA,
# CMA-ES, Nelder-Mead) across a CUTEst sample; requires
# `pip install Py-BOBYQA cma` in addition to scipy
python benchmarks/multi_solver_compare.py --n-max 20 --count 40 --csv results.csv
```
