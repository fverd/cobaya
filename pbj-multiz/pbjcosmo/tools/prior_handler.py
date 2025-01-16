from math import e, pi, sqrt
import numpy as np

def uniform_prior(param_dic):
    """
    Uniform prior for single parameter
    """
    delta = np.asarray(param_dic['max']) - np.asarray(param_dic['min'])

    def prior(x):
        return (param_dic['min'] <= x <= param_dic['max']) / delta

    return prior

#-------------------------------------------------------------------------------

def gaussian_prior(param_dic):
    """
    Gaussian prior for single parameter
    """
    def prior(x):
        return e**(-0.5*((x - param_dic['mean']) / param_dic['sigma'])**2) / \
            sqrt(2.*pi*param_dic['sigma']**2)

    return prior

#-------------------------------------------------------------------------------

def loguniform_prior(param_dic):
    """
    Log-uniform prior for single parameter
    """
    def prior(x):
        return (param_dic['min'] <= x <= param_dic['max']) / (x * log(param_dic['max']/param_dic['min']))

    return prior

#-------------------------------------------------------------------------------

def lognormal_prior(param_dic):
    """
    Log-normal prior for single parameter
    """
    def prior(x):
        return e**(-0.5*((log(x) - param_dic['mean']) / param_dic['sigma'])**2)/\
            (sqrt(2.*pi)*x*param_dic['mean'])

    return prior

#-------------------------------------------------------------------------------

def get_prior_distribution(v):
    """
    Return a callable for the prior distribution given in v.
    """
    if v['prior'] == 'uniform':
        distr = uniform_prior(v['prior_params'])
    elif v['prior'] == 'gaussian':
        distr = gaussian_prior(v['prior_params'])
    elif v['prior'] == 'log-uniform':
        distr = loguniform_prior(v['prior_params'])
    elif v['prior'] == 'log-normal':
        distr = lognormal_prior(v['prior_params'])
    else:
        raise ValueError('Prior distribution not allowed for',v)

    return distr
 
#-------------------------------------------------------------------------------

def add_prior_distribution_to_dict(varied_params, prior_dictionary):
    """
    Initializes all the prior functions and saves them in prior_dictionary

    Parameters
    ----------
    varied_params: list of strings containing the names of the varied params
    prior-dictionary: dict containing the following keys:
    'prior': prior distribution type (str), uniform or Gaussian or
                log-uniform or log_normal
    'prior_params': parameters for the prior distribution (dict):
                    'min', 'max' for uniform and log-uniform,
                    'mean', 'sigma' for gaussian and log-normal

    Returns
    -------
    Updates the input prior_dictionary with a key 'distr' containing the
    callable to the prior distribution
    """
    for k in varied_params:
        v = prior_dictionary[k]
        distr = get_prior_distribution(v)
        prior_dictionary[k]['distr'] = distr

#-------------------------------------------------------------------------------

def prior_transform(prior_dict, p, cube):
    """
    Transforms the unit hypercube to desired distribution for nested sampling
    """
    v = prior_dict[p]
    if v['prior'] == 'uniform':
        return ((v['prior_params']['max'] - v['prior_params']['min'])*cube +
                v['prior_params']['min'])

    elif v['prior'] == 'gaussian':
        return scipy.stats.norm(v['prior_params']['mean'],
                                v['prior_params']['sigma']).ppf(cube)

    elif v['prior'] == 'log-uniform':
        return 10**(cube * (log10(v['prior_params']['max']) +
                            log10(v['prior_params']['min'])) +
                    log10(v['prior_params']['max']))

    elif v['prior'] == 'log-normal':
        return 10**(scipy.stats.norm(v['prior_params']['mean'],
                                     v['prior_params']['sigma']).ppf(cube))
