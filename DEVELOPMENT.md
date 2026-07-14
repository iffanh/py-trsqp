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
