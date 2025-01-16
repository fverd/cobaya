import math
import numpy as np
from scipy.integrate import quad, simps
from scipy.special import hyp2f1, legendre

def Hubble_adim(z, Om, w0, wa):
    """Hubble adim

    Adimensional Hubble factor.
    Computes the adimensional Hubble factor at a given redshift.
    *NOTE* This assumes flatness, i.e. O_L = 1 - O_m

    .. math::
    E(z) = \\sqrt{\\Omega_{m,0} (1+z)^3 + \\Omega_{\\Lambda} (1+z)^{3(1+w)}}

    Arguments
    ---------
    `z`: float, redshift at which the adimensional Hubble factor is computed
    `cosmo`: dictionary, containing values for `'Obh2'`, `'Och2'`, `h`.
    `w`: float, optional. If `None`, defaults to the input cosmology dictionary

    Returns
    -------
    `E(z)`: float, value of the adimensional Hubble factor at input redshift
    """
    DE_evolution = (1+z)**(3*(1+w0+wa)) * np.exp(-3. * wa * z / (1+z))
    return np.sqrt(Om * (1+z)**3 + (1-Om) * DE_evolution)

#-------------------------------------------------------------------------------

def angular_diam_distance(z, Om, w0, wa):
    """Angular diameter distance.

    Computes the angular diameter distance at a given redshift.
    Note that to speed up the calculation we are neglecting a
    `c/(100h*(1+z))` factor, multiply by this if you want the actual angular
    diameter distance in Mpc/h units.

    .. math::
    D_A(z) = \\int_0^z d z' \\frac{1}{E(z)}

    Arguments
    ---------
    `z`: float, redshift at which the adimensional Hubble factor is computed
    `cosmo`: dictionary, containing values for `'Obh2'`, `'Och2'`, `h`.
    `w`: float, optional. If `None`, defaults to the input cosmology dictionary

    Returns
    -------
    `D_A(z)`: float, value of the angular diameter distance at input redshift.
    """
    # Note: the proper units (Mpc/h) would be obtained integrating
    # h/(1+z) * \int c/100h * 1/E(z)
    function = lambda x: 1/Hubble_adim(x, Om, w0, wa)
    res, _  = quad(function, 0, z)
    return res

#-------------------------------------------------------------------------------

def growth_factor(z, Om):
    """Growth factor

    Computes the growth factor D for the redshift specified in
    the cosmo dictionary using the hypergeometric function.
    Assumes flat LCDM.

    Parameters
    ----------
    `cosmo`: dictionary, must include Omh2, h, z.
                Default `None`, uses the input cosmology.

    Returns
    -------
    Dz/D0: float, growth factor normalised to z=0
    D0:    float, growth factor at z=0
    """
    a = 1 / (1 + z)
    Dz = a * hyp2f1(1./3, 1., 11./6, (Om - 1.) / Om * a**3)
    D0 = hyp2f1(1./3, 1., 11./6, (Om - 1.) / Om)
    return  Dz / D0, D0

#-------------------------------------------------------------------------------

def growth_rate(z, Om):
    """Growth rate

    Computes the growth rate f for the redshift specified in
    the cosmo dictionary using the hypergeometric function.
    Assumes flat LCDM.

    Returns
    -------
    f: float, growth rate
    """
    a = 1 / (1 + z)
    # Un-normalised growth factor:
    D, D0 = growth_factor(z, Om)
    dhyp = hyp2f1(4./3, 2., 17./6, (Om - 1.) / Om * a**3)
    return 1. + (6./11) * a**4 / (D*D0) * (Om - 1.) / Om * dhyp

#-------------------------------------------------------------------------------

def set_growth_functions(cosmo_model, **kwargs):
    if cosmo_model == 'lcdm':
        return function_lcdm
    elif cosmo_model in ['wcdm', 'w0wacdm', 'darkscattering', 'ndgp']:
        return function_evogrowth
    elif cosmo_model == 'growthindex':
        return function_gammacdm

#-------------------------------------------------------------------------------

def eisenstein_hu(k, plin, Omh2, Obh2, h, Tcmb, ns):
    """
    Computes the smooth matter power spectrum following the
    prescription of Eisenstein & Hu 1998.

    Arguments
    ---------
    `cosmo`: dict, dictionary with cosmological parameters.
             Default: None, parameters are fixed and read from the paramfile

    Returns
    -------
    `kL`: numpy array containing the k-grid
    `P_EH`: numpy array containing the smooth linear power spectrum
    """
    k *= h
    s = 44.5 * np.log(9.83/Omh2) / np.sqrt(1.+10.*(Obh2)**0.75)
    Gamma = Omh2 / h
    AG = (1. - 0.328 * np.log(431.*Omh2) * Obh2 / Omh2 +
          0.38 * np.log(22.3 * Omh2) * (Obh2 / Omh2)**2)
    Gamma = Gamma * (AG + (1.-AG) / (1.+(0.43*k*s)**4))
    Theta = Tcmb / 2.7
    q  = k * Theta**2 / Gamma / h
    L0 = log(2.*math.e + 1.8*q)
    C0 = 14.2 + 731. / (1. + 62.5 * q)
    T0 = L0 / (L0 + C0 * q * q)
    T0 /= T0[0]
    P_EH  = k**ns * T0**2
    P_EH *= plin[0] / P_EH[0]
    k /= h

    return k, P_EH

#-------------------------------------------------------------------------------

def multipole_projection(mu, pkmu, ell_list):
    return np.array([simps((2*l+1.)/2.*legendre(l)(mu)*pkmu, mu)
                     for l in ell_list])

#-------------------------------------------------------------------------------
