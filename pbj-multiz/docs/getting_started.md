# Getting started

## Installation

## Running the code


???+ abstract "TL;DR"

    You need only two files to run the code: a `param.yaml` parameterfile
    and a `run.py` script, plus a `prior.yaml` if you want to perform
    Bayesian inference

The main object is the `Pbj` object, that can be initialised by
passing a dictionary. To simplify usage, the dictionary can be read
from a `param.yaml` file, that contains all information needed for
initialisation and to run the inference pipeline. See bayesian
inference / parameter file for more info, or have a look at the
[examples](../examples) folder.

Once you have your parameterfile, initialising a `Pbj` object only
requires a few lines of code:

```py title="run.py"

import pbjcosmo 
import pbjcosmo.tools.param_handler as parhandler
import matplotlib.pyplot as plt
import numpy as np

# Load the parameter file, initialise a PBJ object and run init_theory
init_config = parhandler.read_file("param.yaml")
theoryobject = pbjcosmo.Pbj(init_config)
```

From here you can obtain theoretical predictions, or call the
`initialise_full()` method before calling the `run_sampler` method if
you want to perform Bayesian inference. Have a look at
[examples/example_theory.py](../examples/example_theory.py) and
[examples/run_inference.py](../examples/run_inference.py) for the
corresponding scripts.
