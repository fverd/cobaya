# Outputs

Regardless of the sampler chosen, the code will output a `npz` file
containing samples from the posterior distribution foe each
parameter. Additionally, one can request automatic plotting by calling
`save_plot` and computation of some statistics by calling
`chain_stats`. These calls must be added to the running script, for
example:


We provide a support function to read the output chain, that can be
called by specifying the file name, parameter names and latex names,
and optionally ranges. The output is a getdist MCSample object that
can be passed to the getdist package to obtain contourplots. See
examples/plotting-chains for an example


