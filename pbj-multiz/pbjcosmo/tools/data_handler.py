import numpy as np
import pandas as pd

def load_covariance(filename, observables, twopi3, n_bins_power, n_triangles, meansets,
                    covtype='covariance'):
    """
    Loads the covariance matrix.

    Parameters
    ----------
    filename : str or list of str
        filename(s) of the covariance matrix
    observables : list of str
        list of observables to be loaded
    twopi3 : float
        constant to scale the data
    n_bins_power : int
        number of bins to consider for power spectrum observables
    n_triangles : int
        number of triangles to consider for bispectrum observables
    meansets : int
        number of independent sets
    covtype : str, optional
        type of covariance matrix: 'covariance', 'variance', by default 'covariance'

    Returns
    -------
    covariance : 2D array
        the covariance matrix
    """
    if filename is None or isinstance(filename, dict):
        raise NotImplementedError("Missing covariance file")

    if isinstance(filename, list):
        covariance = np.array(
            [pd.read_csv(iname, sep=r"\s+", header=None).values
             for iname in filename])
    else:
        covariance = pd.read_csv(filename, sep=r"\s+", header=None).values

    if covariance.ndim == 1 or (covariance.ndim == 2 and (covariance.shape[0] == 1 or covariance.shape[1] == 1)):
        covariance = np.diag(covariance.flatten())
    elif covariance.ndim == 2 and covtype == 'variance':
        covariance = np.diag(np.diag(covariance))

    factor = np.array([(twopi3**1 if o.startswith(('P', 'BAO')) else twopi3**2) for o in observables for _ in range(n_bins_power if o.startswith(('P', 'BAO')) else n_triangles)])
    factor2 = np.outer(factor, factor)
    covariance *= factor2 / meansets

    return covariance

def load_data(datafiles, observables, twopi3, n_bins_power, nsets=None, meanbool=None):
    """
    Loads data files and returns the data dictionaries for each observable.

    Parameters
    ----------
    datafiles : dict
        dictionary with observables as keys and datafiles as values. Datafiles
        can be a list of files or a single filename.
    observables : list
        list of observables to be loaded.
    twopi3 : float
        constant to scale the data.
    n_bins_power : int
        number of bins to consider for power spectrum observables.
    nsets : int, optional
        number of independent sets, by default None
    meanbool : bool, optional
        whether to compute the mean of the data, by default None

    Returns
    -------
    tuple
        meansets, number of sets, data_dict
    """
    data_dict = {}
    meansets = 1.

    for o in observables:
        files = datafiles[o]
        datum = np.array(
            [pd.read_csv(f, sep=r"\s+", header=None).values
             for f in (files if isinstance(files, list) else [files])])

        if o.startswith('P'):
            datum = datum[..., :n_bins_power]

        if meanbool is not None and isinstance(files, str):
            meansets = datum.shape[0]
            datum = datum.mean(axis=0)

        if datum.ndim == 1:
            datum = datum[np.newaxis, :]

        nsets = nsets or (datum.shape[0] if datum.ndim == 2 else datum.shape[1])

        factor = twopi3 if o.startswith(('P', 'BAO')) else twopi3**2
        data_dict[o] = np.squeeze(datum) * factor

    return meansets, nsets, data_dict
