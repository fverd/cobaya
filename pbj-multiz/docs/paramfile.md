# Parameter file

The parameter file must be in `yaml` format, and contains information
on different aspects of the analysis you can perform with PBJ. A
minimal parameter file must include two sections:

- input cosmology section: specifies values for the cosmological parameters to
  be used to initialise the `Pbj` class

- theory section: contains selection for the code to use to compute the linear
  power spectrum and infrared resummation routine

Optional:

- AP section: contains information on quantities for Alcock-Paczynski distortions

This will allow you to initialise a `Pbj` object from which you can
retrieve theoretial predictions for the observables. An example can be
found in [examples/param_theory.yaml](../examples/param_theory.yaml)

To perform Bayesian inference on the parameters, the code requires additional sections:

- binning section: contains information on the binning scheme, used to
  compute the k-grid. This must follow the specifications of the
  measurements one is trying to fit

- data files section: contains paths to the datafiles for the observables to fit

- covariance section: contains informations and path con the covariance matrix

- likelihood section: contains information on the observables to fit,
  type of likelihood, priors and sampler

- window section: contains information for window convolution and path
  to the mixing matrix

- output section: contains information on paths and names for the output files

- additional informations (see below)

## Binning

Contains information on the binning scheme in the following form:

```yaml
binning:
  type: Effective # options: Average, Effective
  grid:
    dk: 1 # Bin size
    cf: 1 # Center of first bin
    nbinsP: 128 # Number of bins in the power spectrum measurements
    nbinsB: 29 # Must always be included, even if not fitting bispectrum (set to 1)
  SNLeaking: False # only used if binning['type'] == Average
```

Everything is specified in terms of the fundamental frequency, so that
`dk: 1` is a bin-size corresponding to one fundamental frequency, `cf:
1` means the first bin is centered on the fundamental frequency.  The
binning `type` can be `Average` (full bin average, more expensive to
compute), `Effective` (k values correspond to k_effective, faster to
compute). The `SNLeaking` entry allows to include contributions to
shot noise to higher order multipoles. Can only be used if `type:
Effective`.

## Data files

Contains paths to the datafiles for the different observables. Names
allowed for the observables include: `Pk` (real space power spectrum),
`Bk` (real space bispectrum), `P0`,`P2`,`P4` (power spectrum
multipoles, to be used both for the pre-recon and post-recon
measurements)

!!! warning

    When fitting the post-reconstruction power spectrum
    multipoles, the datafile names are still `P0`,`P2`,`P4`, while the
    observables are called `P0_BAO`, `P2_BAO`, `P4_BAO`

```yaml
data_files:
  P0: data/p0_measurement
  P2: data/p2_measurement
  P4: data/p4_measurement
```

## Input cosmology

Contains the cosmological parameters for the initialisation of the `Pbj` class:

```yaml
input_cosmology: # to initialise base quantities
  h: 0.695
  Obh2: 0.02224
  Och2: 0.115422125
  ns: 0.9632
  As: 2.20192887130619e-9
  tau: 0.09
  Tcmb: 2.7255
  z: 1.
  inputfile: None
  Mnu: 0.
  w0: -1.
  wa: 0.
```

While `h`, `Obh2`, `Och2`, `As` and `ns` must always be included, the
others have default values (i.e. can be excluded): `tau=0.09`,
`Tcmb=2.7255`, `Mnu=0`, `w0=-1`, `wa=0`. The redshift is specified by
the `z` key (corresponds to the redshift at which the linear power
spectrum, growth functions and AP factors are computed at
initialisation). `inputfile` can be specified (path) if you want to
read the linear power spectrum.

## Covariance

```yaml
covariance:
  type: covariance # options: covariance, variance
  file: data/minerva/cov_p0p2p4 # path to covariance file
  NMocks: 10000 # Number of mocks used to compute the covariance
```

If using a numerical covariance and `NMocks` is larger than one, it's
possible to use likelihood functions that mitigate noise in the
numerical covariance matrix

## Likelihood

```yaml
likelihood:
  observables: ['P0', 'P2', 'P4'] # options: Pk, Bk, P0, P2, P4, B0, B2, B4, P0_BAO, P2_BAO, P4_BAO
  type: Gaussian # options: Gaussian, Hartlap, Sellentin
  cosmology: varied # options: fixed, varied
  do_analytic_marg: False
  priors: config_default/example_priors.yaml # path to the prior yaml
  sampler: emcee # options: emcee, MetropolisHastings, nautilus, pocomc, ultranest
  check_convergence: False # only for emcee sampler
  resume: False
```

Additional options: `do_jeffreys_priors` can be set to True
(`do_analytic_marg` must be False in this case) `model` name of the
likelihood model function to use. If not specified, the code will try
to automatically select the correct model function based on the rest
of the specs (`cosmology`, `do_analytic_marg`, `do_AP`). If you add
your own likelihood model function in `likelihoodPBJ.py` you can
request it as `model: mymodel_function`.

## Theory

```yaml
theory:
  linear: bacco # options: camb, bacco
  IRresum: True
  IRresum_kind: EH # options EH, DST
  do_redshift_rescaling: False
```

Details on linear power spectrum to use: select the code to compute it
(`linear` key), if to perform infra-red resummation and which kind
(`EH` for Eisenstein-Hu, `DST` for discrete sine transform). `do_redshift_rescaling`: if True, the code computes the linear at z=0 and rescales with the proper growth functions.

!!! tip

    The suggestion is to always leave `do_redshift_rescaling` set
    to True, except when fitting for massive neutrinos

## Alcock-Paczynski distortions

Optional, if not specified the code will default to no Alcock-Paczinsky distortions

```yaml
AP:
  do_AP: False
  AP_as_nuisance: False
  fiducial_cosmology:
    h: 0.695
    Obh2: 0.02224
    Och2: 0.115422125
```

`AP_as_nuisance` allows to vary AP parameters independently from the
cosmology ($\alpha_{par}$ and $\alpha_{perp}$ must be set as sampled
parameters in the `prior.yaml`. `fiducial cosmology contains
cosmological parameters used as fiducial (to perform the measurements).

## Window

Contains information on the mixing matrix for window convolution.

## Output

```yaml
output:
  folder: my_experiment # create results folder if doesn't exist
  comment: _test # string to be appended to filenames
```

`folder` sets the folder for the outputs, `comment` is a string that
will be appended to the outputs filenames.

## Additional keys

```yaml
boxsize: 1500. # Corresponds to 2pi/fundamental_frequency
SN: 18.923548389261743 # 1/nbar in (Mpc/h)^3
PiConvention: LSS # options: LSS, CMB
```