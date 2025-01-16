# Likelihood

The code implements several `likelihood models` to deal with different
setup specifications.

The likelihood function is always Gaussian, but has options to correct
for noise in the covariance matrix when it is computed from a finite
set of mocks (Sellentin, Hartlap corrections).

The code can automatically identify which model to use, based on the
setup passed in the `likelihood` section of the parameter file.  If
you add a new likelihood function, you can tel the code to use it by
specifying its name in the `model` key of the `likelihood` section.
