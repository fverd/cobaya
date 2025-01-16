# Samplers

PBJ implements different options for samplers to explore the posterior
distribution of parameters. The sampler can be selected by specifying
the `sampler` key in the `likelihood` section of the `param.yaml`:



Possible options are:

- `emcee`: the affine invariant sampler implemented in the emcee package
- `metropolis`: the Metropolis-Hastings sampler implemented in the emcee package
- `nautilus`: nested sampler from the nautilus package
- `pocomc`: sampler implemented in the pocomc package
- `utranest`: nested sampler implemented in the ultranest package

Once the sampler is specified in the `param.yaml`, the code
automatically selects the sampler and the analysis can be run by
calling the `run_sampler` method, which requires as input the scale
cuts for the different observables. Additionally, each sampler has
different options, for example:



See the API for additional details.