# PBJ: Power spectrum & Bispectrum Joint analysis

*Authors*: Chiara Moretti, Andrea Oddo

*Contributors*: Maria Tsedrik, Kevin Pardede, Emilio Bellini, Elena
 Sarpa, Pedro Carrilho, Matilde Barberi Squarotti, Sujeong Lee, Jacopo
 Salvalaggio, Cecilia Oliveri

**Full documentation under construction [here](https://chiaramoretti.gitlab.io/pbj/)**

## Getting started

Step 1: clone the repository:
```
git clone https://gitlab.com/chiaramoretti/pbj.git
cd pbj
```

Step 2: create an environment, recommended option is to use `mamba`.
The following commands also work with `conda` (just substitute to `mamba`):
```
mamba env create -f environment.yml
```

Activate the environment:
```
mamba activate pbjenv
```

Step 3 (only on first usage):
From the `pbj/` folder, run
```
pip install .
```
to pip install the package. You are now ready to `import pbjcosmo`!


### Alternative environment creation

We also provide a `requirements.txt` file to create an environment
with `pyenv` and a `poetry.lock` file to create an environment with
poetry.  For `pyenv`, run the following from the `pbj/` folder:
```
# create environment in folder `<envname>` in the current directory
# you can specify your own path and name for the environment
python -m venv <envname>

# activate the environment. If you used a custom path or name for the
# environment, make sure to substitute `<envname>` with the correct path
source <envname>/bin/activate

# install all packages as listed in the `requirements.txt` file
pip install -r requirements.txt

# pip install pbjcosmo
pip install .
```

## Theoretical prediction

See `example_theory.py` in the `examples/` folder for an example on
how to get a theoretical prediction for the multipoles of the power
spectrum and bispectrum. There are some comments explaining the
various steps.

## Running a chain

The `examples/` folder also contains all you need to run a chain. The
example focuses on fixed cosmology, which can be run on a laptop in
only a few minutes, but `param_inference_powerspec.yaml`, `example_priors.yaml`
and `run_inference_powerspec.py` contain comments on how to modify the files to
run with varying cosmology. The datavectors and covariance used for
this are synthetic (noiseless) datavectors and a Gaussian covariance
stored in `examples/data/`