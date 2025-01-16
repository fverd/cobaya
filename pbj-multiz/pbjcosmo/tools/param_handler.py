import yaml
import numpy as np

def read_file(file_path):
    """
    Read from YAML file
    """
    with open(file_path, 'r') as file:
        params = yaml.safe_load(file)

    return params

#-------------------------------------------------------------------------------

def parse_parameters(params):
    """
    Constructs the initial param dictionary from the prior.yaml file
    Does not assign values to varied and derived parameters

    Parameters
    ----------
    params: dict, output of read_file() containing all info on priors

    Returns
    -------
    parsed_params: dict, parameter dict with initial values
    """
    parsed_params = {}

    for key, value in params.items():

        if 'type' in value:
            param_type = value['type']

            if param_type == 'fixed':
                parsed_params[key] = {'value': value['value']}

            elif param_type == 'varied':
                parsed_params[key] = {
                    'value': value['init_value'],
                    'prior': value['prior'],
                    'prior_params': value.get('prior_params', {})}
            elif param_type in ('marg', 'jeffreys'):
                parsed_params[key] = {
                    'value': value['prior_params']['mean'],
                    'prior': value['prior'],
                    'prior_params': value.get('prior_params', {})}
    return parsed_params

#-------------------------------------------------------------------------------

def compute_derived_parameters(params, parsed_params):
    """
    Computes derived parameters based on other parameters.
    To be called after parse_parameters() had been called and the
    parsed_params dictionary has been constructed

    Parameters
    ----------
    params: dict, output of read_file()
    parsed_params: dict, output of parse_parameters(params)

    Returns
    -------
    parsed_params: dict, with the derived parameters

    """
    for key, value in params.items():
        if value['type'] == 'derived':
            try:
                parsed_params[key] = {
                    'value': eval(params[key]['function'], {},
                                  {p: parsed_params[p].get('value')
                                   for p in params[key]['parameters']})}
            except Exception as e:
                print(f"Error in computing derived parameter {key}: {e}")
    return parsed_params

#-------------------------------------------------------------------------------

def update_derived_parameters(params, parsed_params):
    """
    Updates the values of the derived parameters in parsed_params based on the
    other parameters and their current values.

    Parameters
    ----------
    params: dict, output of read_file()
    parsed_params: dict, output of parse_parameters(params)

    Returns
    -------
    parsed_params: dict, with the derived parameters updated
    """
    for key, value in params.items():
        if value['type'] == 'derived':
            try:
                parsed_params.update({key: eval(
                    params[key]['function'], {},
                    {p: parsed_params[p] for p in params[key]['parameters']})})
            except Exception as e:
                print(f"Error in computing derived parameter {key}: {e}")
    return parsed_params

#-------------------------------------------------------------------------------

def add_names_to_chain(prior_dict, varied_params, chain_dict):
    """
    Add parameter names to the chain dictionary

    Parameters
    ----------
    prior_dict: dict, prior dictionary
    varied_params: list of strings, names of the varied parameters
    chain_dict: dict, dictionary of the chain

    If all the varied parameters have a 'latex' key in prior_dict,
    use the latex parameter names; otherwise, use the parameter names
    themselves.

    Returns
    -------
    chain_dict: dict, updated with the parameter names
    """
    if all('latex' in prior_dict[p] for p in varied_params):
        chain_dict.update({'param_names': [prior_dict[p]['latex']
                                           for p in varied_params]})
    else:
         chain_dict.update({'param_names': varied_params})

#-------------------------------------------------------------------------------

def construct_alphas_dict(do_AP):
    """
    Construct a dictionary with all the functions to compute the alphas parameters

    Parameters
    ----------
    do_AP: bool, whether to use AP parameters or not

    Returns
    -------
    out_dict: dict, dictionary with the functions to compute the alphas parameters
    """
    out_dict = {
        'Pk': pk_alphas,
        'PX': px_alphas,
        'Bk': bk_alphas,
        'P0': p0_alphas,
        'P2': p2_alphas,
        'P4': p4_alphas,
    }
    if do_AP:
        out_dict.update({'B0': b0_alphas_AP, 'B2': b24_alphas_AP, 'B4': b24_alphas_AP})
    else:
        out_dict.update({'B0': b0_alphas, 'B2': b24_alphas, 'B4': b24_alphas})
    return out_dict

def pk_alphas(Psn, SNLeaking, b1=1, b2=0, bG2=0, bG3=0, c0=0, aP=0, ek2=0,
              **pars):
    return np.array([0, b1*b1, -2.*c0, b1*b2, b1*bG2, b2*b2, b2*bG2,
                     bG2*bG2, b1*bG3, ek2*Psn, (1.+aP)*Psn])

def px_alphas(Psn, SNLeaking, b1=1, cs2=0, bk2=0, b2=0, bG2=0, bG3=0, ek2X=0, aX=0,
              **pars):
    return np.array([0., b1, -2.*b1*cs2-bk2, 0.5*b2, 0.5*bG2, 0., 0., 0., 0.5*bG3,
                     ek2X*Psn, aX*Psn])

def bk_alphas(Psn, SNLeaking, b1=1, b2=0, bG2=0, a1=0, a2=0, **pars):
    return np.array([b1*b1*b1, b1*b1*b2, b1*b1*bG2,
                     b1*b1*(1.+a1)*Psn, (1.+a2)*Psn*Psn])

def pell_common(Psn, SNLeaking, b1=1, b2=0, bG2=0, bG3=0, c0=0, c2=0, c4=0, ck4=0, f=0,
                **pars):
    return np.array([b1*b1, b1*f, f*f, b1*b2, b1*bG2, b2*b2, b2*bG2,
                     bG2*bG2, b2*f, bG2*f, (b1*f)**2, b1*b1*f, b1*b2*f,
                     b1*bG2*f, b1*f*f, b2*f*f, bG2*f*f, f**4, f**3,
                     b1*f**3, b1*bG3, bG3*f,
                     -2.*c0, -2.*c2*f, -2.*c4*f*f,
                     ck4*f**4*b1**2, 2.*ck4*f**5*b1, ck4*f**6])

def p0_alphas(Psn, SNLeaking, aP=0, e0k2=0, e2k2=0, **pars):
    common = pell_common(Psn, SNLeaking, **pars)
    if SNLeaking:
        return np.concatenate((common,
                               np.array([e0k2*Psn, e2k2*Psn, (1. + aP)*Psn])))
    else:
        return np.concatenate((common,
                               np.array([e0k2*Psn, (1. + aP)*Psn])))

def p2_alphas(Psn, SNLeaking, aP=0, e0k2=0, e2k2=0, **pars):
    common = pell_common(Psn, SNLeaking, **pars)
    if SNLeaking:
        return np.concatenate((common,
                               np.array([e0k2*Psn, e2k2*Psn, (1. + aP)*Psn])))
    else:
        return np.concatenate((common, np.array([e2k2*Psn])))

def p4_alphas(Psn, SNLeaking, aP=0, e0k2=0, e2k2=0, **pars):
    common = pell_common(Psn, SNLeaking, **pars)
    if SNLeaking:
        return np.concatenate((common,
                               np.array([e0k2*Psn, e2k2*Psn, (1. + aP)*Psn])))
    else:
        return np.array(common)

def b0_alphas(Psn, SNLeaking, a2=0, **pars):
    common = b24_alphas(Psn, SNLeaking, **pars)
    return np.concatenate((common, [(1.+a2)*Psn*Psn]))

def b24_alphas(Psn, SNLeaking, b1=1, b2=0, bG2=0, a1=0, a3=-1, f=0, **pars):
    return np.array([b1*b1*b1, b1*b1*b2, b1*b1*bG2, f*b1*b1, f*b1*b1*b1,
                     f*f*b1*b1, f*b1*b2, f*b1*bG2, f*f*b1, f*f*f*b1, f*f*b2,
                     f*f*bG2, f*f*f, f*f*f*f, b1*b1*(1.+a1)*Psn,
                     0.5*f*b1*(2.+a1+a3)*Psn, f*f*(1.+a3)*Psn])

def b0_alphas_AP(Psn, SNLeaking, alpha_par=1, alpha_perp=1, a2=0, **pars):
    APamp = 1./alpha_par/alpha_perp**2

    coef_list = b24_alphas(Psn, SNLeaking, **pars)
    a_perp = (1. - alpha_perp)*coef_list
    a_dif = (alpha_perp - alpha_par)*coef_list
    return np.concatenate((coef_list*APamp**2, a_perp*APamp**2,
                           a_dif*APamp**2, [(1.+a2)*Psn*Psn*APamp**2]))

def b24_alphas_AP(Psn, SNLeaking, alpha_par=1, alpha_perp=1, **pars):
    APamp = 1./alpha_par/alpha_perp**2

    coef_list = b24_alphas(Psn, SNLeaking, **pars)
    a_perp = (1. - alpha_perp)*coef_list
    a_dif = (alpha_perp - alpha_par)*coef_list
    return np.concatenate((coef_list*APamp**2, a_perp*APamp**2,
                           a_dif*APamp**2))

#-------------------------------------------------------------------------------
