# Inputs

The input files containing observables measurements, covariance and
optionally the mixing matrix for window convolution must follow
specific formats. See Parameter File to see how to include them in the
`param.yaml`.

## Observable measurements

Must be in a txt file, single file for each observable and with
measurements in a single row. If you want to simultaneously fit
several measurements (for example, from a set of simulations that
share the same cosmology and population properties for the
galaxies/halos), you can pass a single file (per observable) with
different measurements on different rows.

The code automatically computes k-values based on the `grid` section
in the `param.yaml`. For example:

Note that measurements MUST follow the requested binning scheme, which
can be checked by accessing the `self.kPE` attribute (for power
spectrum measurements) and `self.kB1, self.kB2, self.kB3` attributes
(for bispectrum measurements).

## Covariance

Must be a txt file containing the full covariance matrix for all
observables to be fit. The code automatically takes care of cutting
the different blocks based on the selected scale cuts. Notice that the
elements of the covariance must have the same k-grid as the
measurements (specified in the `grid` section of the `param.yaml).

## Mixing matrix