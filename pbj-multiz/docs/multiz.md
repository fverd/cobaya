# Multiple redshift bins

The code can run on multiple redshift bins on a restricted
configuration. Specifically, it can:

- only fit the power spectrum multipoles (no bispectrum at the moment)
- only use the nautilus sampler
- only run with analytic marginalisation on the linear nuisance parameters
- assumes the same prior distribution (but with different parameters) for each nuisance parameter
- only works with redshift rescaling (no support for massive neutrinos at the moment)
- window convolution to be implemented

## Parameter file

To run on multiple redshift bins, you need to specify a
`redshift_bins` list in the parameter.yaml. Additionally, the
following entries must be specified as lists with the same length and
order as the `redshift_bins` entry:

- `SN`
- `data_files[observable]`
- `covariance[file]`


???+example ""

    ``` yml
	redshift_bins: [1.0, 1.2, 1.4, 1.65]
	SN: [500., 1000., 1700., 3000.] # 1/nbar in h/Mpc units
    data_files:
      P0: ['data/p0_z1.txt','data/p0_z2.txt','data/p0_z3.txt','data/p0_z4.txt']
      P2: ['data/p2_z1.txt','data/p2_z2.txt','data/p2_z3.txt','data/p2_z4.txt']
      P4: ['data/p4_z1.txt','data/p4_z2.txt','data/p4_z3.txt','data/p4_z4.txt']
    covariance:
	  file: ['data/cov_z1.txt','data/cov_z2.txt','data/cov_z3.txt','data/cov_z4.txt']
	```
	

## Prior file

Priors for the parameters that are not shared among different redshift
bins must be specified in the following form for the sampled and
marginalised parameters:

???+example ""

    ``` yml
	b1:
	  type: varied:
      init_value: [1.4, 1.8, 2., 2.5]
      prior: uniform
      prior_params:
        min: [0.8, 0.8, 0.8, 0.8]
        max: [4., 4., 4., 4.]
      latex: b_1

    bG3:
      type: marg
      prior: gaussian
      prior_params:
        mean: [0., 0., 0., 0.]
        sigma: [3., 3., 3., 3.]
      latex: b_{\Gamma_3}
	```


## Run script

The kmax values must be passed as lists to the `run_sampler` function,
and they can take different values for the different redshift
bins. The lists order must follow the same order as the
`redshift_bins` key in the paramfile.

???+example ""

    ``` py
	kf = pbjobject.kf
	pbjobject.run_sampler(NmaxP=[0.2/kf, 0.2/kf, 0.25/kf, 0.3/kf],
	NmaxP4=[0.15/kf, 0.15/kf, 0.1/kf, 0.1/kf, 0.1/kf])
	```

