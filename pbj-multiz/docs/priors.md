# Setting priors

To specify priors for the parameters you need to create a `yaml` file that has to be specified in the parameter file:

```yaml
prior: prior.yaml
```

The yaml contains all informations on the parameters that are to be
sampled in the Bayesian analysis, parameters that are to be
analytically marginalised over and parameters that have to be fixed to
a specific value.


## Parameter names

The parameters MUST be specified using the following names:

- nuisance parameters: `b1`, `b2`, `bG2`, `bG3` for the biases, `c0`,
  `c2`, `c4`, `ck4` for the counterterms, `aP`, `e0k2`, `e2k2` for the
  shot noise parameters
- systematics: `f_out` for the fraction of interlopers, `sigma_z` for
  the redshift error
- cosmological parameters: `Och2`, `Obh2`, `As`, `ns`, `h`, `w0`,
  `wa`, `Mnu`
- beyond $\Lambda$CDM: `gamma` for the growth index parameterisation,
  `Omrc` for nDGP modified gravity, `xi`, `A` for Dark Scattering
  parameters

Additionally, derived parameters can also be specified (see below).


## Sampled parameters

To specify parameters to sample over, ad the following to the `prior.yaml` file:

```yaml title="prior.yaml"
b1:
  type: varied
  init_value: 2.71 # Initial value, used by emcee to initialise walkers
  prior: uniform # Prior distribution
  prior_params:
    min: 0.9
    max: 3.5
  latex: b_1 # String used when plotting
```

Possible options for the prior distribution are:

- uniform prior: specify `min` and `max` of the distribution in the
  `prior_params` section;

- Gaussian prior: specify `mean` and `sigma` of the distribution in
  the `prior_params` section;

- log-uniform prior: specify `min` and `max` of the distribution in
  the `prior_params` section;

- log-normal prior: specify `mean` and `sigma` of the distribution in
  the `prior_params` section;


## Derived parameters

Derived parameters can be specified by setting `type: derived` and
adding a function that describes the relation between the derived
parameters and the sampled parameters.

For example, we can set a local Lagrangian relation on `bG2` by adding
to the `prior.yaml`:

```yaml title="prior.yaml"

```

Another possibility is to change the sampled parameters, for example
to sample $\log A_s$ instead of $A_s$ directly we can add to the
`prior.yaml`:

```yaml title="prior.yaml"
logAs:
  type: varied
  init_value: 
  prior: uniform
  prior_params:
    min: 
    max: 
  latex: \log A_s

As:
  type: derived
  init_value: 
  prior: uniform
  prior_params:
    min: -4.
    max: 4.
  latex: b_2
```


## Fixed parameters

Fixed parameters can be specified by setting `type: fixed` and adding
a `value` for the parameter in the `params.yaml`. For example:

## Analytically marginalised parameters

Parameters that enter linearly in the model can be analytically
marginalised over instead of performing direct sampling, which speeds
up the convergence time of chains. This is possible for the following
nuisance parameters: `bG3`, `c0`, `c2`, `c4`, `ck4`, `aP`, `e0k2`,
`e2k2`. To activate this option, set `do_AM: True` to the
`param.yaml`.

Analytically marginalised parameters can be set by adding
`type: marg`. Additionally, one must specify the `mean` and `sigma`
for the parameter, as follows:

```yaml title="prior.yaml"
bG3:
  type: marg
  prior: gaussian # MUST be gaussian
  prior_params:
    mean: 0.
    sigma: 3.
  latex: b_{\Gamma_3}
```

!!! warning

    The analytically marginalised parameters MUST have a Gaussian
    prior, the code will raise an error otherwise.

At the moment, the only possibility when doing analytical
marginalisation is to marginalise over ALL linear parameters, or set
them to zero (no fixed value different from zero, and no sampling of
some parameters and marginalisation over the others).

## Jeffreys priors

There is an option to run with Jeffreys priors on the linear
parameter, the only modifications needed with respect to the
analytically marginalised case are:

- change the `type` to `jeffreys` in the `prior.yaml`
- set `do_jeffreys_priors` to True in the `param.yaml`

```yaml title="prior.yaml"
bG3:
  type: jeffreys
  prior: gaussian # MUST be gaussian
  prior_params:
    mean: 0.
    sigma: 3.
  latex: b_{\Gamma_3}
```

As for analytically marginalised parameters, if you activate this
option the code will apply Jeffreys priors to all possible
parameters. At the moment, the only alternative is to set one or more
of these parameters to zero.