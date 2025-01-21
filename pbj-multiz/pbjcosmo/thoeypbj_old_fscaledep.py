import camb
import copy
import math
from math import e, pi
import numpy as np
from numpy import absolute, arange, array, concatenate, einsum, exp, inf, insert, hstack, linspace, log, log10, logical_and, logical_or, logspace, newaxis, ones, sqrt, sum, vstack, where, cos, shape, real, zeros, loadtxt, interp, vectorize
from scipy.constants import speed_of_light
from scipy.fftpack import dst, idst
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d, RectBivariateSpline, splev, splrep, UnivariateSpline
from scipy.integrate import simps, quad
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve
from scipy.special import hyp2f1, legendre, spherical_jn
import numba as nb
import time
from astropy import cosmology
from astropy.cosmology import FlatLambdaCDM

from multiprocessing import Pool

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

try: import baccoemu
except: print("\033[1;31m[WARNING]: \033[00m"+"Bacco not installed!")

try: import fastpt.FASTPT_simple as fpt
except:	import FASTPT_simple as fpt

try:    import fastpt.FASTPT as tns
except:    import FASTPT as tns

nopython_bool = True

@nb.jit(nopython=nopython_bool)
def legendre_poly(ell, mu):
    if ell == 0:
        return 1 + 0*mu
    elif ell == 2:
        return 0.5*(3*mu*mu-1)
    elif ell == 4:
        return 0.125*(35*mu*mu*mu*mu - 30*mu*mu + 3)

def legendre_poly_024(mu):
    #already multiplied by (2l+1)/2
    return 1/2*array([1 + 0*mu, 5*0.5*(3*mu*mu-1), 9*0.125*(35*mu*mu*mu*mu - 30*mu*mu + 3)])

@nb.jit(nopython=nopython_bool)
def mu_f(k1,k2,k3):
    return (k3*k3 - k1*k1 - k2*k2)/(2*k1*k2)

@nb.jit(nopython=nopython_bool)
def F2(k1,k2,k3):
    mu = mu_f(k1,k2,k3)
    return 5/7 + (mu/2)*(k1/k2+k2/k1) + 2/7*mu*mu

@nb.jit(nopython=nopython_bool)
def G2(k1,k2,k3):
    mu = mu_f(k1,k2,k3)
    return 3/7 + (mu/2)*(k1/k2+k2/k1) + 4/7*mu*mu

@nb.jit(nopython=nopython_bool)
def s2(k1,k2,k3):
    mu = mu_f(k1,k2,k3)
    return mu*mu - 1

@nb.jit(nopython=nopython_bool)
def z1(mu1,f,b1):
    return b1+f*mu1*mu1

@nb.jit(nopython=nopython_bool)
def z2(k1,k2,k3,mu1,mu2,f1,f2,f3,b1,b2,bG2):
    mu = mu_f(k1,k2,k3)
    return b1*F2(k1,k2,k3) + b2/2 + f3*(k1*mu1+k2*mu2)**2/(k1**2 + k2**2 + 2*k1*k2*mu)*G2(k1,k2,k3) +\
            b1*f3/2*(k1*mu1 + k2*mu2)*(mu1/k1 + mu2/k2) + f1*f3*mu1*mu2*(k1*mu1 + k2*mu2)**2/(4*k1*k2) + f2*f3*mu1*mu2*(k1*mu1 + k2*mu2)**2/(4*k1*k2) +\
            bG2*s2(k1,k2,k3)

@nb.jit(nopython=nopython_bool)
def bisp(k1,k2,k3,mu1,mu2,mu3,f1,f2,f3,b1,b2,bG2,a1,psn,pk1,pk2,pk3):
    return 2*z2(k1,k2,k3,mu1,mu2,f1,f2,f3,b1,b2,bG2)*z1(mu1,f1,b1)*z1(mu2,f2,b1)*pk1*pk2 +\
    2*z2(k2,k3,k1,mu2,mu3,f2,f3,f1,b1,b2,bG2)*z1(mu2,f2,b1)*z1(mu3,f3,b1)*pk2*pk3 +\
    2*z2(k3,k1,k2,mu3,mu1,f3,f1,f2,b1,b2,bG2)*z1(mu3,f3,b1)*z1(mu1,f1,b1)*pk3*pk1 +\
        psn*(1+a1)*b1*(z1(mu1,f1,b1)*pk1+z1(mu2,f2,b1)*pk2+z1(mu3,f3,b1)*pk3)


@nb.jit(parallel=True, nopython=nopython_bool)
def Bl_terms_AP_(k1,k2,k3, klist, flist, a_perp,a_orth,b1,b2,bG2,a1,Psn, xs1,xs2, integ_mat_0,integ_mat_2, Pk1,Pk2,Pk3, DPk1,DPk2,DPk3):

    mu2 = lambda mu1,phi, k1,k2,k3: mu1*mu_f(k1,k2,k3)-(1-mu1**2)**(1/2)*(1-mu_f(k1,k2,k3)**2)**(1/2)*cos(phi)
    nu = lambda mu: real(mu/a_perp*(mu**2/a_perp**2 + (1-mu**2)/a_orth**2)**(-1/2))
    q_k = lambda k, mu: real(k*(mu**2/a_perp**2 + (1-mu**2)/a_orth**2)**(1/2))
    f_k = lambda k, mu: real(np.interp(q_k(k, mu), klist, flist))

    integ = lambda ell, mu1, phi, k1,k2,k3, Pk1,Pk2,Pk3, DPk1,DPk2,DPk3: bisp(
        q_k(k1, mu1),
        q_k(k2, mu2(mu1,phi,k1,k2,k3)),
        q_k(k3, -k1/k3*mu1-k2/k3*mu2(mu1,phi,k1,k2,k3)),
        nu(mu1),
        nu(mu2(mu1,phi,k1,k2,k3)),
        nu(-k1/k3*mu1-k2/k3*mu2(mu1,phi,k1,k2,k3)),
        f_k(k1, mu1),
        f_k(k2, mu2(mu1,phi,k1,k2,k3)),
        f_k(k3, -k1/k3*mu1-k2/k3*mu2(mu1,phi,k1,k2,k3)),
        b1,
        b2,
        bG2,
        a1,
        Psn,
        Pk1 + k1*DPk1*(1-a_perp + (a_perp-a_orth)*mu1**2),
        Pk2 + k2*DPk2*(1-a_perp + (a_perp-a_orth)*mu2(mu1, phi, k1, k2, k3)**2),
        Pk3 + k3*DPk3*(1-a_perp + (a_perp-a_orth)*(-k1/k3*mu1-k2/k3*mu2(mu1,phi,k1,k2,k3))**2)
        )*legendre_poly(ell, mu1)*(2*ell+1)/(4*pi)*(1/a_perp)**2*(1/a_orth)**4

    nmax, imax, jmax = shape(integ_mat_0)
    for n in nb.prange(nmax):
        for i in nb.prange(imax):
            for j in nb.prange(jmax):
                integ_mat_0[n,i,j] += integ(0, xs1[i], pi*(xs2[j]+1), k1[n],k2[n],k3[n], Pk1[n],Pk2[n],Pk3[n],\
                                                        DPk1[n],DPk2[n],DPk3[n])*pi
                integ_mat_2[n,i,j] += integ(2, xs1[i], pi*(xs2[j]+1), k1[n],k2[n],k3[n], Pk1[n],Pk2[n],Pk3[n],\
                                                        DPk1[n],DPk2[n],DPk3[n])*pi

class PBJtheory:
    """PBJtheory

    Implements methods for the comological functions and theoretical calculations
    """

    def CallBACCO(self, npoints=300, kmin=1.e-4, kmax=10., redshift=0.,
                  cold=False, cosmo=None):
        """Call to the BACCO linear emulator given a set cosmological
        parameters.  By default, the P(k) is computed in 1000 points
        from kmin=1e-4 to kmax=198.  To account for the k-cut of BACCO
        (k_max = 50 h/Mpc), a linear extrapolation is implemented for
        the log(P(k)).

        Arguments
        ---------
        `npoints`, `kmin`, `kmax`: float, k-grid parameters.
                                   Defaults: 1000, 1e-4, 198
        `cold`: bool, keyword to request the CDM + baryons power spectrum.
                Default: False
        `cosmo`: dict, dictionary with cosmological parameters.
                 Default: None, parameters are fixed and read from the paramfile

        Returns
        -------
        `kL`: numpy array containing the k-grid
        `PL`: numpy array containing the linear power spectrum
        """
        if cosmo is not None:
            ns   = cosmo['ns']
            As   = cosmo['As']
            h    = cosmo['h']
            Obh2 = cosmo['Obh2']
            Omh2 = cosmo['Omh2']
            Och2 = cosmo['Och2']
            Mnu  = cosmo['Mnu'] if 'Mnu' in cosmo else 0.
            w0   = cosmo['w0'] if 'w0' in cosmo else -1.
            wa   = cosmo['wa'] if 'wa' in cosmo else 0.
        else:
            ns   = self.ns
            As   = self.As
            h    = self.h
            Obh2 = self.Obh2
            Omh2 = self.Omh2
            Mnu  = self.Mnu
            w0   = self.w0
            wa   = self.wa
            
        params = {
            'ns'            : ns,
            'A_s'           : As,
            'hubble'        : h,
            'omega_baryon'  : Obh2/h/h,
            'omega_cold'    : (Omh2 - Mnu/93.14)/h/h, # This is Omega_cb!!!
            'neutrino_mass' : Mnu,
            'w0'            : w0,
            'wa'            : wa,
            'expfactor'     : 1/(1+redshift)
        }

        kL = logspace(log10(kmin), log10(kmax), npoints)

        # print('\n\n\n BACCO \n\n\n')
        # print(As,Obh2/h/h,(Omh2 - Mnu/93.14)/h/h)
        # The emulator only works up to k=50, so split the k vector
        kcut = kL[where(kL <= 50)]
        kext = kL[where(kL > 50)]
        _, PL = self.emulator.get_linear_pk(k=kcut, cold=cold, **params)

        # Extrapolation with power law
        m = math.log(PL[-1] / PL[-2]) / math.log(kcut[-1] / kcut[-2])
        PL_ext = PL[-1] / kcut[-1]**m * kext**m

        return kL, hstack((PL, PL_ext))

#-------------------------------------------------------------------------------

    def CallCAMB(self, npoints=1000, kmin=1.e-4, kmax=198., cosmo=None, redshifts=[0.]):
        """Basic call to CAMB given the cosmological parameters, to
        compute the linear power spectrum. By default, the P(k) is
        computed in 1000 points from kmin=1e-4 to kmax=198.

        Arguments
        ---------
        `npoints`, `kmin`, `kmax`: float, k-grid parameters.
                                   Defaults: 1000, 1e-4, 198
        `cosmo`: dict, dictionary with cosmological parameters.
                 Default: None, parameters are fixed and read from the paramfile

        Returns
        -------
        `kL`: numpy array containing the k-grid
        `PL`: numpy array containing the linear power spectrum
        """
        if cosmo is not None:
            ns   = cosmo['ns']
            As   = cosmo['As']
            h    = cosmo['h']
            Obh2 = cosmo['Obh2']
            Omh2 = cosmo['Omh2']
            tau  = cosmo['tau']
            Mnu  = cosmo['Mnu'] if 'Mnu' in cosmo else 0.
            Ok   = cosmo['Ok'] if 'Ok' in cosmo else 0.
            w0   = cosmo['w0'] if 'w0' in cosmo else -1.
            wa   = cosmo['wa'] if 'wa' in cosmo else 0. 
        else:
            ns   = self.ns
            As   = self.As
            h    = self.h
            Obh2 = self.Obh2
            Omh2 = self.Omh2
            Ok   = self.Ok
            tau  = self.tau
            Mnu  = self.Mnu
            w0   = self.w0
            wa   = self.wa

        nmassive = 1 if Mnu != 0 else 0
        params = camb.model.CAMBparams()
        params.InitPower.set_params(ns=ns, As=As, r=0.)
        params.set_cosmology(H0 = 100.*h,
                             ombh2 = Obh2,
                             omch2 = Omh2-Obh2-Mnu/93.14,
                             mnu   = Mnu,
                             nnu   = 3.046,
                             num_massive_neutrinos = nmassive,
                             omk   = Ok,
                             tau   = tau)
        params.set_dark_energy(w=w0, wa=wa, dark_energy_model='ppf')
        params.set_matter_power(redshifts = redshifts, kmax = kmax)
        results = camb.get_results(params)
        kcamb, zcamb, pkcamb = results.get_matter_power_spectrum(minkh = kmin,
                                                                 maxkh = kmax,
                                                                 npoints = npoints,
                                                                 var1 = 'delta_nonu',
                                                                 var2 = 'delta_nonu')
        return kcamb, zcamb, pkcamb

#-------------------------------------------------------------------------------

    def CallEH_NW(self, cosmo=None):
        """ Computes the smooth matter power spectrum following the
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
        if cosmo is not None:
            ns   = cosmo['ns']
            As   = cosmo['As']
            h    = cosmo['h']
            Obh2 = cosmo['Obh2']
            Omh2 = cosmo['Omh2']
            tau  = cosmo['tau']
            Tcmb = cosmo['Tcmb']
            PL   = cosmo['PL']
        else:
            ns   = self.ns
            As   = self.As
            h    = self.h
            Obh2 = self.Obh2
            Omh2 = self.Omh2
            tau  = self.tau
            Tcmb = self.Tcmb
            PL   = self.PL

        kL = self.kL
        kL *= h
        s = 44.5 * log(9.83/Omh2) / sqrt(1.+10.*(Obh2)**0.75)
        Gamma = Omh2 / h
        AG = (1. - 0.328 * log(431.*Omh2) * Obh2 / Omh2 +
              0.38 * log(22.3 * Omh2) * (Obh2 / Omh2)**2)
        Gamma = Gamma * (AG + (1.-AG) / (1.+(0.43*kL*s)**4))
        Theta = Tcmb / 2.7
        q  = kL * Theta**2 / Gamma / h
        L0 = log(2.*e + 1.8*q)
        C0 = 14.2 + 731. / (1. + 62.5 * q)
        T0 = L0 / (L0 + C0 * q * q)
        T0 /= T0[0]
        P_EH  = kL**ns * T0**2
        P_EH *= PL[0] / P_EH[0]
        kL /= h

        return kL, P_EH

#-------------------------------------------------------------------------------

#     def pksmooth_dst(self, cosmo=None, redshift=0, cold=True, plinear=self.PL):
#         """
#         Returns the de-wiggled power spectrum at a given expansion factor

#         he de-wiggling is performed by identifying and removing the BAO bump
#         in real space (by means of a dst transform), and consequently
#         transforming back to Fourier space.

#         See:
#         - Baumann et al 2018 (https://arxiv.org/pdf/1712.08067.pdf)
#         - Giblin et al 2019 (https://arxiv.org/pdf/1906.02742.pdf)

#         :param wavemode: The wavemode in :math:`h \\mathrm{Mpc}^{-1}`
#         :type wavemode: array_like
#         :param expfactor: The required expanction factor
#         :type expfactor: float
#         :param cold: Whether to return the cold or total matter component, default to True
#         :type cold: bool, optional
#         :param scale: Whether to apply the scaled cosmology parameters, defaults to True
#         :type scale: bool, optional
#         :param bias: The linear bias
#         :type bias: float, optional
#         :rtype: array_like
#         """
#         if cosmo is not None:
#             ns   = cosmo['ns']
#             As   = cosmo['As']
#             h    = cosmo['h']
#             Obh2 = cosmo['Obh2']
#             Omh2 = cosmo['Omh2']
#             Mnu  = cosmo['Mnu'] if 'Mnu' in cosmo else 0.
#             w0   = cosmo['w0'] if 'w0' in cosmo else -1.
#             wa   = cosmo['wa'] if 'wa' in cosmo else 0.
#         else:
#             ns   = self.ns
#             As   = self.As
#             h    = self.h
#             Obh2 = self.Obh2
#             Omh2 = self.Omh2
#             Mnu  = self.Mnu
#             w0   = self.w0
#             wa   = self.wa

#         params = {
#             'ns'            : ns,
#             'A_s'           : As,
#             'hubble'        : h,
#             'omega_baryon'  : Obh2/h/h,
#             'omega_cold'    : (Omh2 - Mnu/93.14)/h/h, # This is Omega_cb!!!
#             'neutrino_mass' : Mnu,
#             'w0'            : w0, # These params are fixed
#             'wa'            : wa,
#             'expfactor'     : 1/(1+redshift)
#         }

#         # Sample k, P(k) in 2^15 points
#         nk = int(2**15)
#         kmin = 1e-4
#         kmax = 10 # my kmax must be 198
#         klin = linspace(kmin, kmax, nk)
#         _, pklin = self.emulator.get_linear_pk(k=klin, cold=cold, **params)

#         # DST-II log(k *P(k))
#         f = log10(klin * pklin)
#         dstpk = dst(f, type=2)

#         # Split in even and odd indices
#         even = dstpk[0::2]
#         odd = dstpk[1::2]
#         i_even = arange(len(even)).astype(int)
#         i_odd  = arange(len(odd)).astype(int)
#         even_cs = splrep(i_even, even, s=0)
#         odd_cs = splrep(i_odd, odd, s=0)

#         # Compute second derivatives and interpolate separately with cubic splines
#         even_2nd_der = splev(i_even, even_cs, der=2, ext=0)
#         odd_2nd_der = splev(i_odd, odd_cs, der=2, ext=0)

#         # Find i_min and i_max for the even and odd arrays
#         # these indexes have been fudged for the k-range considered
#         # [~1e-4, 10], any other choice would require visual inspection
#         imin_even = i_even[100:300][np.argmax(even_2nd_der[100:300])] - 20
#         imax_even = i_even[100:300][np.argmin(even_2nd_der[100:300])] + 70
#         imin_odd = i_odd[100:300][np.argmax(odd_2nd_der[100:300])] - 20
#         imax_odd = i_odd[100:300][np.argmin(odd_2nd_der[100:300])] + 75

#         # Carve out the BAOs
#         # carve indices
#         i_even_holed = concatenate((i_even[:imin_even], i_even[imax_even:]))
#         i_odd_holed = concatenate((i_odd[:imin_odd], i_odd[imax_odd:]))
#         # carve log(k*P(k))
#         even_holed = concatenate((even[:imin_even], even[imax_even:]))
#         odd_holed = concatenate((odd[:imin_odd], odd[imax_odd:]))
#         # interpolate the arrays rescaled by a factor of (i + 1)^2 using cubic splines
#         even_holed_cs = splrep(i_even_holed, even_holed * (i_even_holed+1)**2, s=0)
#         odd_holed_cs = splrep(i_odd_holed, odd_holed * (i_odd_holed+1)**2, s=0)

#         # Merge the two arrays containing the respective elements without the bumps,
#         # and without the rescaling factor of (i + 1)^2, and inversely FST
#         even_smooth = splev(i_even, even_holed_cs, der=0, ext=0) / (i_even + 1)**2
#         odd_smooth = splev(i_odd, odd_holed_cs, der=0, ext=0) / (i_odd + 1)**2

#         dstpk_smooth = []
#         for ii in range(len(i_even)):
#             dstpk_smooth.append(even_smooth[ii])
#             dstpk_smooth.append(odd_smooth[ii])
#         dstpk_smooth = array(dstpk_smooth)

#         pksmooth = idst(dstpk_smooth, type=2) / (2 * len(dstpk_smooth))
#         pksmooth = 10**(pksmooth) / klin

#         smooth_itp = interp1d(klin[where(klin<=5)], pksmooth[where(klin<=5)],
#                               kind='cubic')

#         kcut = self.kL[where(self.kL <= 5)]
#         pksmooth_cut = smooth_itp(kcut)

#         # # Compute the pklin at high k (>5)
#         # k_highk = self._tabPowerSpectrum_z0['x'][self._tabPowerSpectrum_z0['x'] > 5]
#         # p_highk = self.get_powerspec_z(wavemode=k_highk, expfactor=expfactor,
#         #                              cold=cold, scale=scale, bias=bias)
#         # # concatenate the k and pk with the smooth for k<5
#         # k_extended = concatenate((klin[klin < 5], k_highk))
#         # p_extended = concatenate((pksmooth[klin < 5], p_highk))
#         # # log-interpolate the total
#         # pksmooth_cs = splrep(log(k_extended), log(p_extended), s=0)
#         # pksmooth_interp = exp(splev(log(wavemode), pksmooth_cs, der=0, ext=0))

#         return self.kL, hstack((pksmooth_cut, plinear[self.kL > 5]))

# #-------------------------------------------------------------------------------

#     def get_psmooth(self, kind='EH', cosmo=None, redshift=0, cold=False):
#         if kind == 'EH':
#             k, pk = self.CallEH_NW(cosmo=cosmo)
#         elif kind == 'DST':
#             k, pk = self.pksmooth_dst(cosmo=cosmo, redshift=redshift, cold=cold)

#         return k, pk

#-------------------------------------------------------------------------------

    def _gaussianFiltering(self, lamb, cosmo=None):
        """ Private function to compute the Gaussian filtering
        required for IR-resummation.

        Arguments
        ----------
        `lamb`: float, width of Gaussian filter

        Returns
        -------
        `Psmooth`: numpy array, no-wiggle power spectrum
        """
        if cosmo is not None: PL = cosmo['PL']
        else: PL = self.PL

        def extrapolateX(x):
            logx = log10(x); xSize = x.size; dlogx = log10(x[1]/x[0])
            newlogx = linspace(logx[0]  - xSize * dlogx / 2.,
                               logx[-1] + xSize * dlogx / 2., 2*xSize)
            newx = 10.**newlogx
            return newx

        def extrapolateFx(x, Fx):
            newx   = extrapolateX(x)
            backB  = math.log(Fx[1] / Fx[0]) / math.log(x[1] / x[0])
            backA  = Fx[0] / x[0]**backB
            forwB  = math.log(Fx[-1] / Fx[-2]) / math.log(x[-1] / x[-2])
            forwA  = Fx[-1] / x[-1]**forwB
            backFx = backA * newx[:x.size//2]**backB
            forwFx = forwA * newx[3*x.size//2:]**forwB
            return hstack((backFx, Fx, forwFx))

        kNumber = len(self.kL)
        qlog  = log10(self.kL)
        dqlog = qlog[1]-qlog[0]
        qlog  = log10(extrapolateX(self.kL))
        PLinear = extrapolateFx(self.kL, PL)
        _, PEH = self.CallEH_NW(cosmo=cosmo)
        #PEH=psmooth
        PEH = extrapolateFx(self.kL, PEH)
        Psmooth = gaussian_filter1d(PLinear/PEH, lamb/dqlog)*PEH

        return Psmooth[kNumber//2:kNumber//2+kNumber]

#-------------------------------------------------------------------------------
    
    # def remove_bao(self, k_low = 2.8e-2, k_high = 4.5e-1, cosmo=None):
    #     """
    #     This routine removes the BAOs from the input power spectrum and returns
    #     the no-wiggle power spectrum in :math:`(\mathrm{Mpc}/h)^3`.
    #     Adapted from Colibrì by G. Parimbelli

    #     Arguments
    #     ---------
    #     `param k_in` array, scales in units of :math:`h/\mathrm{Mpc}`.
    #     `param pk_in` array, power spectrum in units of :math:`(\mathrm{Mpc}/h)^3`.
    #     `param k_low` float, lowest scale to spline in :math:`h/\mathrm{Mpc}`. Default = 2.8e-2
    #     `param k_high` float, highest scale to spline in :math:`h/\mathrm{Mpc}`. Default = 4.5e-1

    #     Returns
    #     -------
    #     `Psmooth` array, power spectrum without BAO.
    #     """
    #     if cosmo is not None: PL = cosmo['PL']
    #     else: PL = self.PL

    #     k_ref = [k_low, k_high] # k-range that contains BAOs

    #     # Get interpolating function for input P(k) in log-log space:
    #     _interp_pk = interp1d(log(self.kL), log(PL), kind='quadratic',
    #                           bounds_error=False)
    #     interp_pk = lambda x: exp(_interp_pk(log(x)))

    #     # Spline all (log-log) points outside k_ref range:
    #     idxs = where(logical_or(self.kL <= k_ref[0], self.kL >= k_ref[1]))
    #     _pk_smooth = UnivariateSpline(log(self.kL[idxs]), log(PL[idxs]),
    #                                   k=3, s=0)
    #     pk_smooth = lambda x: exp(_pk_smooth(log(x)))

    #     # Find second derivative of each spline:
    #     fwiggle = UnivariateSpline(self.kL, PL/pk_smooth(self.kL), k=3, s=0)
    #     derivs  = array([fwiggle.derivatives(_k) for _k in self.kL]).T
    #     d2      = UnivariateSpline(self.kL, derivs[2], k=3, s=1.0)

    #     # Find maxima and minima of the gradient (zeros of 2nd deriv.), put a
    #     # low-order spline through zeros to subtract smooth trend from wiggles
    #     wzeros = d2.roots()
    #     wzeros = wzeros[where(logical_and(wzeros>=k_ref[0],
    #                                          wzeros<=k_ref[1]))]
    #     wzeros = concatenate((wzeros, [k_ref[1],]))
    #     try:
    #         wtrend = UnivariateSpline(wzeros, fwiggle(wzeros), k=3, s=None,
    #                                   ext='extrapolate')
    #     except:
    #         wtrend = UnivariateSpline(self.kL, fwiggle(self.kL), k=3, s=None,
    #                                   ext='extrapolate')

    #     # Construct smooth no-BAO:
    #     idxs = where(logical_and(self.kL > k_ref[0], self.kL < k_ref[1]))
    #     pk_nobao = pk_smooth(self.kL)
    #     pk_nobao[idxs] *= wtrend(self.kL[idxs])

    #     # Construct interpolating functions:
    #     ipk = interp1d(self.kL, pk_nobao, kind='cubic', bounds_error=False,
    #                    fill_value=0.)

    #     Psmooth = ipk(self.kL)

    #     return Psmooth

#-------------------------------------------------------------------------------

    def IRresum(self, PEH, lamb=0.25, kS=0.2, lOsc=102.707, cosmo=None):
        """ Splitting of the linear power spectrum into a smooth and a
        wiggly part, and computation of the damping factors

        Parameters
        ----------
        `PEH`: array of floats, smooth (Eisenstein-Hu) power spectrum
        `lamb`: float, width of Gaussian filter
                Default 0.25
        `kS`: float, scale of separation between large and small scales
              Default 0.2
        `lOsc`: float, scale of the BAO
                Default 102.707
        `cosmo`: dict, dictionary with cosmological parameters.
                 Default: None, parameters are fixed and read from the paramfile

        Returns
        -------
        `kL` : array of floats, k-grid
        `Pnw`: array of floats, no-wiggle power spectrum
        `Pw`: array of floats, wiggle power spectrum
        `Sigma2`: float
        `dSigma2` float
        """

        if cosmo is not None: PL = cosmo['PL']
        else: PL = self.PL

        # Sigma2 as integral up to 0.2;
        # Uses Simpson integration (no extreme accuracy needed)
        icut = (self.kL <= kS)
        kLcut = self.kL[icut]

        Pnw = self._gaussianFiltering(lamb, cosmo=cosmo)
        Pw = PL - Pnw

        Pnwcut = Pnw[icut]
        norm = 1./(6.*pi**2)
        Sigma2  = norm*simps(Pnwcut*(1.-spherical_jn(0,kLcut*lOsc)+
                                     2.*spherical_jn(2,kLcut*lOsc)), x=kLcut)
        dSigma2 = norm*simps(3.*Pnwcut*spherical_jn(2,kLcut*lOsc), x=kLcut)

        return self.kL, Pnw, Pw, Sigma2, dSigma2

#-------------------------------------------------------------------------------

    def _muxdamp(self, k, mu, Sigma2, dSigma2, f):
        """Muxdamp

        Computes the full damping factor for IR-resummation in redshift space
        and the (mu^n * exponential damping)

        Parameters
        ----------
        `mu`: float or array of floats, cosine of the angle between k and l.o.s.
        `Sigma2`: float, value of \Sigma^2
        `dSigma2`: float, value of \delta \Sigma^2
        `f`: float, growth rate

        Returns
        -------
        `Sig2mu`: float or array of floats, \Sigma^2(\mu)
        `RSDdamp`: foat or array of floats (can be 2D), \exp(-k^2*\Sigma^2(\mu))
        """
        Sig2mu = Sigma2 + f * mu**2 * (2. * Sigma2 + f *
                                       (Sigma2 + dSigma2 * (mu**2 - 1)))
        RSDdamp = exp(-k**2 * Sig2mu)
        return Sig2mu, RSDdamp

#-------------------------------------------------------------------------------
    def fOverf0EH(self, zev, k, OmM0, h, fnu):
        '''Rutine to get f(k)/f0 and f0.
        f(k)/f0 is obtained following H&E (1998), arXiv:astro-ph/9710216
        f0 is obtained by solving directly the differential equation for the linear growth at large scales.
        FROM FOLPSnu
        Args:
            zev: redshift
            k: wave-number
            OmM0: Omega_b + Omega_c + Omega_nu (dimensionless matter density parameter)
            h = H0/100
            fnu: Omega_nu/OmM0
        Returns:
            f(k)/f0 (when 'EdSkernels = True' f(k)/f0 = 1)
            f0
        '''

        if fnu==0: return np.ones(len(k))

        eta = np.log(1/(1+zev))   #log of scale factor
        Neff = 3.046              #effective number of neutrinos
        omrv = 2.469*10**(-5)/(h**2 * (1 + 7/8*(4/11)**(4/3)*Neff)) #rad: including neutrinos
        aeq = omrv/OmM0           #matter-radiation equality
            
        pcb = 5/4 - np.sqrt(1 + 24*(1 - fnu))/4     #neutrino supression
        c = 0.7         
        Nnu = 3                                     #number of neutrinos
        theta272 = (1.00)**2                        # T_{CMB} = 2.7*(theta272)
        pf = (k * theta272)/(OmM0 * h**2)  
        DEdS = np.exp(eta)/aeq                      #growth function: EdS cosmology
            
        yFS = 17.2*fnu*(1 + 0.488*fnu**(-7/6))*(pf*Nnu/fnu)**2  #yFreeStreaming 
        rf = DEdS/(1 + yFS)
        fFit = 1 - pcb/(1 + (rf)**c)                #f(k)/f0

        return fFit
#-------------------------------------------------------------------------------

    def growth_factor(self, cosmo=None, z=None, k_fid=1.e-2, scale_dependent=False, method="hyper"):
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
        # Get redshift
        if z is None and cosmo is None:
            z_ref = self.z
        elif z is None:
            z_ref  = cosmo['z']
        else:
            z_ref = z

        # Get growth factor
        if method == "hyper":
            if scale_dependent:
                raise ValueError(
                    'Incompatible method and scale_dependent parameters')
            if cosmo is None:
                Om = self.Om
            else:
                Om = cosmo['Omh2']/(cosmo['h']**2)
            a = 1 / (1 + z_ref)
            Dz = a * hyp2f1(1./3, 1., 11./6, (Om - 1.) / Om * a**3)
            D0 = hyp2f1(1./3, 1., 11./6, (Om - 1.) / Om)
            # Equivalent (only if uncomment the other formulation in growth_rate)
            # Dz = a*sqrt(1.-a**3*(Om-1.)/Om) * hyp2f1(5./6., 3./2., 11./6., a**3*(Om-1.)/Om)
            # D0 = sqrt(1./Om) * hyp2f1(5./6., 3./2., 11./6., (Om-1.)/Om)
            return  None, Dz / D0, D0

        elif method == "bacco":
            klist, pk = self.CallBACCO(redshift=0., cosmo=cosmo)
            D0 = sqrt(pk)
            if z_ref == 0.:
                Dz = D0
            else:
                klist, pk = self.CallBACCO(redshift=z_ref, cosmo=cosmo)
                Dz = sqrt(pk)

        elif method == "camb":
            if z_ref == 0.:
                klist, _, pk = self.CallCAMB(redshifts=[0.], cosmo=cosmo)
                Dz = sqrt(pk[0])
            else:
                klist, zcamb, pk = self.CallCAMB(redshifts=[z_ref, 0.], cosmo=cosmo)
                Dz = sqrt(pk[1])
            D0 = sqrt(pk[0])
        else:
            raise ValueError("Method not recognised")

        if scale_dependent:
            return klist, Dz / D0, D0
        else:
            diff = np.abs(klist-k_fid)
            idx = np.where(diff == diff.min())
            Dz = Dz[idx][0]
            D0 = D0[idx][0]
            return  None, Dz / D0, D0

#-------------------------------------------------------------------------------

    def growth_rate(self, cosmo=None, z=None, k_fid=1.e-2, scale_dependent=False, method="hyper", z_step=0.1):
        """Growth rate

        Computes the growth rate f for the redshift specified in
        the cosmo dictionary using the hypergeometric function.
        Assumes flat LCDM.

        Returns
        -------
        f: float, growth rate
        """
        # Get redshift
        if z is None and cosmo is None:
            z_ref = self.z
        elif z is None:
            z_ref  = cosmo['z']
        else:
            z_ref = z
        
        def gf(z):
            gf = self.growth_factor(
                cosmo=cosmo,
                z=z,
                k_fid=k_fid,
                scale_dependent=scale_dependent,
                method=method)
            return gf

        # Get growth rate
        if method == "hyper":
            if cosmo is None:
                Om = self.Om
            else:
                Om = cosmo['Omh2']/(cosmo['h']**2)
            a = 1 / (1 + z_ref)
            _, D, D0 = self.growth_factor(cosmo=cosmo, z=z_ref)
            dhyp = hyp2f1(4./3, 2., 17./6, (Om - 1.) / Om * a**3)
            f = 1. + (6./11) * a**4 / (D*D0) * (Om - 1.) / Om * dhyp
            # Equivalent (only if uncomment the other formulation in growth_factor)
            # f = (3.*D*D0-5.*a)*Om/2./(D*D0)/((a**3.-1.)*Om-a**3.)
            return None, f
        
        if method == "EisHu":
            if cosmo is None:
                Om = self.Om
            else:
                Om = cosmo['Omh2']/(cosmo['h']**2)
            # compute f0 with hyper
            a = 1 / (1 + z_ref)
            _, D, D0 = self.growth_factor(cosmo=cosmo, z=z_ref)
            dhyp = hyp2f1(4./3, 2., 17./6, (Om - 1.) / Om * a**3)
            f0 = 1. + (6./11) * a**4 / (D*D0) * (Om - 1.) / Om * dhyp

            #Now f/f0
            f= f0 * self.fOverf0EH(z_ref, self.kL, Om, cosmo['h'], cosmo['Mnu']/93.14/cosmo['Omh2'])

            return self.kL, f
        else:
            klist, D0, _ = gf(z_ref)
            # if possible, use two-sided derivative with default value of z_step
            if (z >= z_step):
                _, D_p1, _ = gf(z=z_ref+z_step)
                _, D_m1, _ = gf(z=z_ref-z_step)
                dDdz = (D_p1-D_m1)/(2.*z_step)
            else:
                # if z is between z_step/10 and z_step, reduce z_step to z, and then stick to two-sided derivative
                if (z > z_step/10.):
                    z_step = z
                    _, D_p1, _ = gf(z+z_step)
                    _, D_m1, _ = gf(z-z_step)
                    dDdz = (D_p1-D_m1)/(2.*z_step)
                # if z is between 0 and z_step/10, use single-sided derivative with z_step/10
                else:
                    z_step /=10
                    _, D_p1, _ = gf(z+z_step)
                    dDdz = (D_p1-D0)/z_step
            f = -(1+z)*dDdz/D0
            return klist, f


#-------------------------------------------------------------------------------

    def _get_growth_functions(self, f=None, D=None, cosmo=None):
        """Get growth functions

        Private method that returns the growth functions f, D.  If
        `cosmo=None`, uses values from the input cosomlogy. Otherwise,
        if `f` and / or `D` are not specified, computes the growth
        functions from the `growth_rate` and `growth_factor` methods.
        
        Parameters
        ----------
        `f`: float, input growth rate. Default `None`
        `D`: float, input growth factor. Default `None`
        `cosmo`: dictionary, input cosmological dictionary. Default `None`

        Returns
        -------
        f, D: floats, growth functions.

        See also
        --------
        self.growth_rate(), self.growth_factor()
        """
        if cosmo is not None:
            if f is None:
                _, f = self.growth_rate(cosmo=cosmo)
            if D is None:
                _, D, _ = self.growth_factor(cosmo=cosmo)
        else:
            if f is None:
                f = self.f
            if D is None:
                D = self.Dz
        return f, D

#-------------------------------------------------------------------------------

    def _get_growth_functions_gamma(self, gamma=0.545, cosmo=None):
        if cosmo is not None:
            Om = cosmo['Omh2']/(cosmo['h'])**2
            z  = cosmo['z']
        else:
            Om = self.Om
            z = self.z

        integrand =  lambda zz: -(Om/(Om+(1.-Om)*(1.+zz)**(-3)))**gamma / (1+zz)

        cosmo_ini = {}
        if cosmo is not None:
            cosmo_ini = cosmo
        else:
            cosmo_ini['Omh2'] = self.Omh2
            cosmo_ini['h'] = self.h
        cosmo_ini['z'] = 1100

        _, D_lcdm_ini, _ = self.growth_factor(cosmo=cosmo_ini)

        D0 = D_lcdm_ini / (exp(quad(integrand, 0, cosmo_ini['z'])[0]))
        D = D0 * exp(quad(integrand, 0, z)[0])
        f = -integrand(z) * (1.+z)
        return f, D
    #-------------------------------------------------------------------------------



    def Hubble_adim(self, z, cosmo=None):
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
        if cosmo is not None:
            Om = (cosmo['Obh2'] + cosmo['Och2'] + cosmo['Mnu']/93.14)/(cosmo['h'])**2
            w0 = cosmo['w0'] if 'w0' in cosmo else -1.
        else:
            Om = (self.Obh2 + self.Och2)/(self.h**2)
            w0 = self.w0
        return sqrt(Om * (1+z)**3 + (1-Om) * (1+z)**(3+3*w0))

#-------------------------------------------------------------------------------

    def angular_diam_distance(self, z, cosmo=None):
        """Angular diameter distance.

        Computes the angular diameter distance at a given redshift.
        *NOTE* this is in Mpc/h units

        .. math::
        D_A(z) = \\frac{h}{1+z} \\int_0^z d z' \\frac{1}{E(z)}

        Arguments
        ---------
        `z`: float, redshift at which the adimensional Hubble factor is computed
        `cosmo`: dictionary, containing values for `'Obh2'`, `'Och2'`, `h`.
         `w`: float, optional. If `None`, defaults to the input cosmology dictionary

        Returns
        -------
        `D_A(z)`: float, value of the angular diameter distance at input redshift
        """
        if cosmo is not None:
            Om = (cosmo['Obh2'] + cosmo['Och2'] + cosmo['Mnu']/93.14)/(cosmo['h'])**2
            h  = cosmo['h']
            w0 = cosmo['w0'] if 'w0' in cosmo else -1.
        else:
            Om = (self.Obh2 + self.Och2)/(self.h**2)
            h = self.h
            w0 = self.w0

        # Note: the proper units (Mpc/h) would be obtained integrating
        # h/(1+z) * \int c/100h * 1/E(z)
        function = lambda x: 1/self.Hubble_adim(x, cosmo=cosmo)
        res, err  = quad(function, 0, z)
        return res / (1+z)

#-------------------------------------------------------------------------------

    def AP_factors(self, mugrid, cosmo=None):
        """AP factors

        Computes quantities to do Alcock-Paczynski distortions.

        Arguments
        ---------
        `mu`: array, cosine of the angle between k and the line of sight
        `cosmo`: dictionary, containing values for `'Obh2'`, `'Och2'`, `h` and `'z'`

        Returns
        -------
        `AP_factor`: array, sqrt(mu^2/alpha_par^2 + (1-mu^2)/alpha_perp^2)
        `alpha_par`: float, H^{fiducial}(z)/H(z)
        `AP_amplitude`: float, 1 / (alpha_{par} alpha_{perp}^2)
        """
        if cosmo is not None:
            h = cosmo['h']
            z = cosmo['z']
        else:
            h = self.h
            z = self.z

        alpha_par =  self.Hubble_adim(z, cosmo=cosmo) / \
            self.Hubble_adim(z, cosmo=self.FiducialCosmo)
        alpha_perp = self.angular_diam_distance(z, cosmo=self.FiducialCosmo) / \
            self.angular_diam_distance(z, cosmo=cosmo)

        AP_amplitude = 1 / (alpha_par * alpha_perp**2)
        AP_factor = sqrt(1. + mugrid**2 * ((alpha_perp / alpha_par)**2 - 1.))

        return alpha_par, alpha_perp, AP_factor, AP_amplitude

#-------------------------------------------------------------------------------

    def _apply_AP_distortions(self, k, mu, cosmo=None):
        """Apply AP distortions
        
        Private method that applies the AP distortion to the input k, mu
        arrays.

        .. math::
        q = k \\left( \\frac{\\mu^2}{\\alpha_{\\parallel}^2} + \\frac{1-\\mu^2}{\\alpha_{\\perp}^2} \\right)^{1/2}

        .. math::
        \\nu = \\frac{\\mu}{\\alpha_{\\parallel}} \\left( \\frac{\\mu^2}{\\alpha_{\\parallel}^2} + \\frac{1-\\mu^2}{\\alpha_{\\perp}^2}} \\right)^{-1/2}

        Arguments
        ---------
        `k`: array, reference values of k
        `mu`: array, reference values of mu
        `cosmo`: dictionary, containing values for `'Obh2'`, `'Och2'`, `h` and `z`.

        Returns
        -------
        `q`: array, distorted k
        `nu`: array, distorted mu
        `AP_amplitude`: float, 1/(alpha_par alpha_perp^2)
        """
        alpha_par, alpha_perp, AP_factor, AP_amplitude = self.AP_factors(mu, cosmo=cosmo)

        k_sub = k[:, newaxis]
        nu = mu * (alpha_perp / alpha_par) / AP_factor
        q  = k_sub / alpha_perp * AP_factor
        return q, nu, AP_amplitude

#-------------------------------------------------------------------------------

    def PowerfuncIR_real(self, cosmo=None):
        """Powerfunc IR real

        Computes the leading order power spectrum, the next-to-leading
        order corrections and the EFT counterterm contribution in real
        space. If `self.IRresum=False` all contributions will not be
        IR-resummed.

        Parameters
        ----------
        `cosmo`: dictionary, if `None` uses the input cosmology
        
        Returns
        -------
        Arrays with the various building blocks for the real space
        galaxy power spectrum in the following order:
        `PLO, PNLO, kL**2*PLO, Pb1b2, Pb1g2, Pb2b2, Pb2g2, Pg2g2, Pb1g3, kL**2`
        """
        if cosmo is not None:
            _, Dz, _ = self.growth_factor(cosmo=cosmo)
            PL = cosmo['PL']
        else:
            Dz = self.Dz
            PL = self.PL

        DZ2 = Dz*Dz
        kEH, PEH = self.CallEH_NW(cosmo=cosmo)
        knw, Pnw, Pw, Sigma2, dSigma2 = self.IRresum(PEH, cosmo=cosmo)
        PLf = PL*DZ2
        Pnw     *= DZ2
        Pw      *= DZ2
        Sigma2  *= DZ2
        dSigma2 *= DZ2

        Sigma2 *= self.IRres
        PLO     = Pnw + exp(-self.kL*self.kL*Sigma2)*Pw

        P1l_IR, Pb1b2_IR, Pb1g2_IR, Pb2b2_IR, Pb2g2_IR, Pg2g2_IR, Pb1g3_IR = \
            self.fastpt.Pk_real_halo_full_one_loop(self.kL, PLO, C_window=.75)
        PNLO    = (Pnw + exp(- self.kL*self.kL * Sigma2) * Pw *
                   (1. + self.kL*self.kL * Sigma2) + P1l_IR)

        return (PLO, PNLO, self.kL**2*PLO, Pb1b2_IR, Pb1g2_IR, Pb2b2_IR,
                Pb2g2_IR, Pg2g2_IR, Pb1g3_IR, self.kL**2)

#-------------------------------------------------------------------------------

    def _Pgg_kmu_terms(self, cosmo=None, with_cross=False):
        """Pgg kmu terms

        Private method that computes the terms for the loop
        corrections at redshift z=0, splits into wiggle and no-wiggle
        and stores them as attributes of the class. It also sets
        interpolators to be used when `self.do_AP == True`

        Paramters
        ---------
        `cosmo`:  dictionary, if `None` uses the input cosmology
        """
        if cosmo is not None: PL = cosmo['PL']
        else: PL = self.PL

        _, self.PEH = self.CallEH_NW(cosmo=cosmo)

        _, self.Pnw, self.Pw, self.Sigma2, self.dSigma2 = \
            self.IRresum(self.PEH, cosmo=cosmo)


        # start = time.time()
        # Loops on P_L and Pnw
        loop22_L = self.fastpt.Pkmu_22_one_loop_terms(self.kL, PL, C_window=.75,with_cross=with_cross)
        #print('\t--- calculate loops_22L', time.time() - start)
        # start = time.time()
        loop22_nw = self.fastpt.Pkmu_22_one_loop_terms(self.kL, self.Pnw,
                                                       C_window=.75,with_cross=with_cross)
        #print('\t--- calculate loops_22nw', time.time() - start)
        # start = time.time()
        loop13_L = self.fastpt.Pkmu_13_one_loop_terms(self.kL, PL)
        #print('\t--- calculate loops_13L', time.time() - start)
        # start = time.time()
        loop13_nw = self.fastpt.Pkmu_13_one_loop_terms(self.kL, self.Pnw)
        #print('\t--- calculate loops_13nw', time.time() - start)

        # start = time.time()
        setattr(self, 'loop22_nw', array(loop22_nw))
        setattr(self, 'loop13_nw', array(loop13_nw))
        #print('\t--- setattr', time.time() - start)

        # Compute wiggle
        # start = time.time()
        loop22_w = array([i - j for i, j in zip(loop22_L, loop22_nw)])
        loop13_w = array([i - j for i, j in zip(loop13_L, loop13_nw)])
        #print('\t--- compute wiggle', time.time() - start)

        # start = time.time()
        setattr(self, 'loop22_w', loop22_w)
        setattr(self, 'loop13_w', loop13_w)

        if self.do_AP:
            # Set interpolators, only used for AP distortions
            setattr(self, 'Pnw_int', interp1d(self.kL, self.Pnw))
            setattr(self, 'Pw_int', interp1d(self.kL, self.Pw))

            setattr(self, 'loop22_nw_int', interp1d(self.kL, self.loop22_nw))
            setattr(self, 'loop13_nw_int', interp1d(self.kL, self.loop13_nw))
            setattr(self, 'loop22_w_int', interp1d(self.kL, self.loop22_w))
            setattr(self, 'loop13_w_int', interp1d(self.kL, self.loop13_w))

#-------------------------------------------------------------------------------

    def PowerfuncIR_RSD(self, f=None, D=None, cosmo=None):
        """Powerfunc IR RSD

        Computes the redshift space leading order, the next-to-leading order
        corrections and EFT counterterms contributions of the power
        spectrum multipoles in redshift space. If `self.IRresum == False` all
        contributions will not be IR-resummed.

        Parameters
        ----------
        `f`: float, growth rate. If `None`, it's computed from the input cosmology
        `D`: float, growth factor. If `None`, it's computed from the input cosmology
        `cosmo`:  dictionary, if `None` uses the input cosmology

        Returns
        -------
        `P_0, P_2, P_4`: lists, contain all terms to build the multipoles
        """
        f, D =  self._get_growth_functions(f=f, D=D, cosmo=cosmo)

        mu = np.linspace(0, 1, 101).reshape(1, 101) # This is NOT self.mu
        DZ2 = D*D
        DZ4 = D**4.

        Pnw = self.Pnw * DZ2
        Pw = self.Pw * DZ2
        Pnw_sub = Pnw[:, newaxis]
        Pw_sub = Pw[:, newaxis]
        ks_sub = self.kL[:, newaxis]

        # Rescaling of Sigma and dSigma2
        Sigma2 = self.Sigma2 * DZ2 * self.IRres
        dSigma2 = self.dSigma2 * DZ2 * self.IRres

        Sig2mu, RSDdamp = self._muxdamp(ks_sub, mu, Sigma2, dSigma2, f)

        # Integrals of the mu dependent parts of different terms
        # columns -> (growing) power of mu, rows -> multipole
        I = [[1., 1./3, 1./5,   1./7,   1./9],
             [0., 2./3, 4./7,  10./21, 40./99],
             [0., 0.,   8./35, 24./77, 48./143]]

        # The 2* in J and K is to use the simmetry of the mu integral and
        # compute the integration only over 0,1
        J = 2. * array([[simps(mu**n * RSDdamp * legendre(l)(mu), x=mu, axis=1)
                    for n in [0,2,4,6,8]] for l in [0,2,4]])

        # Integral over the mu dependent part for P_NLO
        K = 2. * array([[simps(mu**n * RSDdamp * self.kL[:,newaxis]**2. * Sig2mu *
                          legendre(l)(mu), x=mu, axis=1) for n in [0,2,4]]
                   for l in [0,2,4]])

        # Function to perform IR resummation in redshift space
        def IRresum_z(loop_nw, loop_w, iloop, imu, il, NLO=False):
            # imu -> power of mu, il -> multipole
            P_IR = (loop_nw[iloop] * I[il][imu] +
                    loop_w[iloop]  * J[il][imu] * (2. * il + 0.5))

            if NLO == True:
                P_IR += (loop_w[iloop] * K[il][imu] * (2. * il + 0.5))
            return P_IR

        # P_LO,l also contains terms for P_ctr,l (up to mu^8)
        P_LO  = [[IRresum_z([Pnw],[Pw], 0, i, l) for i in range(5)]
                 for l in range(3)]
        P_NLO = [[IRresum_z([Pnw],[Pw], 0, i, l, NLO=True) for i in range(3)]
                 for l in range(3)]

        # Re-scales the loop correction terms
        loop22_nw_sub = self.loop22_nw * DZ4
        loop13_nw_sub = self.loop13_nw * DZ4
        loop22_w = self.loop22_w * DZ4
        loop13_w = self.loop13_w * DZ4

        # Lists that contain indexes of fastpt terms, split into powers of mu
        iloop_22 = [range(6),range(6,16),range(16,24),range(24,27),range(27,28)]
        iloop_13 = [range(3),range(3,6), range(6,7)]

        # Computation of multipoles: lists that contain all terms
        P_22 = array([[IRresum_z(loop22_nw_sub, loop22_w, j,i,l)
                       for i in range(5) for j in iloop_22[i]]
                      for l in range(3)])

        P_13 = array([[IRresum_z(loop13_nw_sub, loop13_w, j,i,l)
                       for i in range(3) for j in iloop_13[i]]
                      for l in range(3)])

        P_13mu = array([[IRresum_z(loop13_nw_sub, loop13_w, j,i+1,l)
                         for i in range(3) for j in iloop_13[i]]
                        for l in range(3)])

        P_ctrk2 = array([[self.kL**2*P_LO[l][i]
                          for i in range(3)] for l in range(3)])

        P_ctrk4 = array([[self.kL**4*P_LO[l][i]
                          for i in range(2,5)] for l in range(3)])

        P_l = [vstack([
            [P_NLO[l][i]   for i in range(len(P_NLO[l]))],    # P_NLO
            [P_22[l][i]    for i in range(len(P_22[l]))],     # P_22
            [P_13[l][i]    for i in range(len(P_13[l]))],     # P_13
            [P_13mu[l][i]  for i in range(len(P_13mu[l]))],
            [P_ctrk2[l][i] for i in range(len(P_ctrk2[l]))],  # counterterm k^2
            [P_ctrk4[l][i] for i in range(len(P_ctrk4[l]))]]) # counterterm k^4
               for l in range(3)]

        isum = [[0,3,31] , [1,1,9,34,38] , [2,26,41] , [4] , [5,33] ,
                [6] , [7] , [8] , [10] , [11,40] , [12,25] , [13,35] ,
                [14] , [15] , [16,24,36,37,42] , [17,22] , [18,23] ,
                [19,27,30] , [20,28,43,44] , [21,29] , [32] , [39],
                [45], [46], [47], [48], [49], [50]]

        P_l_short = [[sum([[P_l[l][idx] for idx in isum[i]]
                           for i in range(28)][j], axis=0)
                      for j in range(len(isum))]
                     for l in range(3)]
        
        return (P_l_short[0], P_l_short[1], P_l_short[2])

#-------------------------------------------------------------------------------

    def P_kmu_z(self, f=None, D=None, cosmo=None, **kwargs):
        r"""P kmu z

        Computes the non-linear galaxy power spectrum in redshift
        space at the proper redshift specified by the growth
        parameters. If `self.IRresum == False`, the power spectrum is
        not IR-resummed.

        Parameters
        ----------
        f, D:             float, growth rate and growth factor

        b1, b2, bG2, bG3: float, bias parameters
        c0, c2, c4, ck4:  float, EFT counterterms
        aP, e0k2, e2k2:   float, shot-noise parameters
        Psn:              float, Poisson shot noise

        Returns
        -------
        P(k, mu): interpolator object

        """
        if f is None or D is None:
            f, D =  self._get_growth_functions(f=f, D=D, cosmo=cosmo)

        DZ2 = D*D
        DZ4 = D**4.

        mu = self.mu
        
        ks_sub  = self.kL[:, newaxis]
        if self.scale_dependent_growth:
            f_sub = f[:, newaxis]
        else:
            f_sub = f

        # Rescale and reshape the wiggle and no-wiggle P(k)
        Pnw = self.Pnw * DZ2
        Pw = self.Pw * DZ2
        Pnw_sub = Pnw[:, newaxis]
        Pw_sub = Pw[:, newaxis]

        # Rescaling of Sigma and dSigma2
        Sigma2 = self.Sigma2 * DZ2 * self.IRres
        dSigma2 = self.dSigma2 * DZ2 * self.IRres

        Sig2mu, RSDdamp = self._muxdamp(ks_sub, mu, Sigma2, dSigma2, f_sub)

        # Compute loop nw and rescale to D**4
        loop22_nw_sub = self.loop22_nw * DZ4
        loop13_nw_sub = self.loop13_nw * DZ4
        loop22_w = self.loop22_w * DZ4
        loop13_w = self.loop13_w * DZ4

        # Bias parameters
        b1 = kwargs['b1'] if 'b1' in kwargs.keys() else 1
        b2 = kwargs['b2'] if 'b2' in kwargs.keys() else 0
        bG2 = kwargs['bG2'] if 'bG2' in kwargs.keys() else 0
        bG3 = kwargs['bG3'] if 'bG3' in kwargs.keys() else 0
        c0 = kwargs['c0'] if 'c0' in kwargs.keys() else 0
        c2 = kwargs['c2'] if 'c2' in kwargs.keys() else 0
        c4 = kwargs['c4'] if 'c4' in kwargs.keys() else 0
        ck4 = kwargs['ck4'] if 'ck4' in kwargs.keys() else 0
        aP = kwargs['aP'] if 'aP' in kwargs.keys() else 0
        e0k2 = kwargs['e0k2'] if 'e0k2' in kwargs.keys() else 0
        e2k2 = kwargs['e2k2'] if 'e2k2' in kwargs.keys() else 0
        Psn = kwargs['Psn'] if 'Psn' in kwargs.keys() else self.Psn
        
        def Kaiser(b1, f, nu):
            return b1 + f * nu**2

        # Next-to-leading order, counterterm, noise
        PNLO = Kaiser(b1, f_sub, mu)**2 * (Pnw_sub + RSDdamp * Pw_sub *
                                       (1. + ks_sub**2 * Sig2mu))
        Pkmu_ctr = (-2. * (c0 + c2 * f_sub * mu**2 + c4 * f_sub * f_sub * mu**4) *
                    ks_sub**2 + ck4 * f_sub**4 * mu**4 * Kaiser(b1, f_sub, mu)**2 *
                    ks_sub**4) * (Pnw_sub + RSDdamp * Pw_sub)
        Pkmu_noise = Psn * ((1. + aP) + (e0k2 + e2k2 * mu**2) * ks_sub**2)
        
        # print(f'Pkmu_noise is {Pkmu_noise} while end is ')

        # Biases
        bias22 = array([b1**2 * mu**0 * f_sub**0,
                        b1 * b2 * mu**0 * f_sub**0,
                        b1 * bG2 * mu**0 * f_sub**0,
                        b2**2 * mu**0 * f_sub**0,
                        b2 * bG2 * mu**0 * f_sub**0,
                        bG2**2 * mu**0 * f_sub**0,
                        mu**2 * f_sub * b1,
                        mu**2 * f_sub * b2,
                        mu**2 * f_sub * bG2,
                        (mu * f_sub * b1)**2,
                        (mu * b1)**2 * f_sub,
                        mu**2 * f_sub * b1 * b2,
                        mu**2 * f_sub * b1 * bG2,
                        (mu * f_sub)**2 * b1,
                        (mu * f_sub)**2 * b2,
                        (mu * f_sub)**2 * bG2,
                        (mu * f_sub)**4,
                        mu**4 * f_sub**3,
                        mu**4 * f_sub**3 * b1,
                        mu**4 * f_sub**2 * b2,
                        mu**4 * f_sub**2 * bG2,
                        mu**4 * f_sub**2 * b1,
                        mu**4 * f_sub**2 * b1**2,
                        mu**4 * f_sub**2,
                        mu**6 * f_sub**4,
                        mu**6 * f_sub**3,
                        mu**6 * f_sub**3 * b1,
                        mu**8 * f_sub**4])
        
        bias13 = array([b1 * Kaiser(b1, f_sub, mu),
                        bG3 * Kaiser(b1, f_sub, mu),
                        bG2 * Kaiser(b1, f_sub, mu),
                        mu**2 * f_sub * Kaiser(b1, f_sub, mu),
                        mu**2 * f_sub * b1 * Kaiser(b1, f_sub, mu),
                        (mu * f_sub)**2 * Kaiser(b1, f_sub, mu),
                        mu**4 * f_sub**2 * Kaiser(b1, f_sub, mu)])

        # Use correct einsum
        if self.scale_dependent_growth:
            repl = 'ikl,ik->kl'
        else:
            repl = 'ijl,ik->kl'
        Pkmu_22_nw = einsum(repl, bias22, loop22_nw_sub)
        Pkmu_22_w  = einsum(repl, bias22, loop22_w)
        Pkmu_13_nw = einsum(repl, bias13, loop13_nw_sub)
        Pkmu_13_w  = einsum(repl, bias13, loop13_w)

        Pkmu_22 = Pkmu_22_nw + RSDdamp * Pkmu_22_w
        Pkmu_13 = Pkmu_13_nw + RSDdamp * Pkmu_13_w
        Pkmu    = PNLO + Pkmu_22 + Pkmu_13 + Pkmu_ctr + Pkmu_noise

        return RectBivariateSpline(self.kL, self.mu, Pkmu)

#-------------------------------------------------------------------------------

    def P_cross_kmu_z(self, f=None, D=None, cosmo=None, **kwargs):
        r"""P kmu z

        Computes the cross power spectrum in redshift
        space at the proper redshift specified by the growth
        parameters.

        Parameters
        ----------
        f, D:             float, growth rate and growth factor

        b1, b2, bG2, bG3: float, bias parameters
        c0, c2, c4, ck4:  float, EFT counterterms

        Returns
        -------
        P(k, mu): interpolator object

        """

        if f is None or D is None:
            f, D =  self._get_growth_functions(f=f, D=D, cosmo=cosmo)

        DZ2 = D*D
        DZ4 = D**4.

        mu = self.mu
        
        ks_sub  = self.kL[:, newaxis]
        if self.scale_dependent_growth:
            f_sub = f[:, newaxis]
        else:
            f_sub = f

        # Rescale and reshape the wiggle and no-wiggle P(k)
        Pnw = self.Pnw * DZ2
        Pw = self.Pw * DZ2
        Pnw_sub = Pnw[:, newaxis]
        Pw_sub = Pw[:, newaxis]

        # Rescaling of Sigma and dSigma2
        Sigma2 = self.Sigma2 * DZ2 * self.IRres

        Sig2mu = Sigma2 + f_sub * mu**2 *  Sigma2
        RSDdamp = exp(-ks_sub**2 * Sig2mu)

        # Compute loop nw and rescale to D**4
        loop22_nw_sub = self.loop22_nw * DZ4
        loop13_nw_sub = self.loop13_nw * DZ4
        loop22_w = self.loop22_w * DZ4
        loop13_w = self.loop13_w * DZ4

        # Bias parameters
        b1 = kwargs['b1'] if 'b1' in kwargs.keys() else 1
        b2 = kwargs['b2'] if 'b2' in kwargs.keys() else 0
        bG2 = kwargs['bG2'] if 'bG2' in kwargs.keys() else 0
        bG3 = kwargs['bG3'] if 'bG3' in kwargs.keys() else 0
        c4 = kwargs['c4'] if 'c4' in kwargs.keys() else 0
        if 'c0x' in kwargs.keys():
            c0 = kwargs['c0x']
        else:
            c0 = kwargs['c0'] if 'c0' in kwargs.keys() else 0
        if 'c2x' in kwargs.keys():
            c2 = kwargs['c2x']
        else:
            c2 = kwargs['c2'] if 'c2' in kwargs.keys() else 0
        if 'ck4x' in kwargs.keys():
            ck4 = kwargs['ck4x']
        else:
            ck4 = kwargs['ck4'] if 'ck4' in kwargs.keys() else 0

        def Kaiser(b1, f, nu):
            return b1 + f * nu**2

        # Next-to-leading order, counterterm, noise
        PNLO = Kaiser(b1, f_sub, mu) * (Pnw_sub + RSDdamp * Pw_sub *
                                       (1. + ks_sub**2 * Sig2mu)) #only haloes are n redshift space
        
        Pkmu_ctr = (-2. * (c0 + c2 * f_sub * mu**2 + c4 * f_sub * f_sub * mu**4) *
                    ks_sub**2 + ck4 * f_sub**2 * mu**2 * Kaiser(b1, f_sub, mu) *
                    ks_sub**4) * (Pnw_sub + RSDdamp * Pw_sub)
        
        # Biases here is what changes
        bias22 = array([b1 * mu**0 * f_sub**0, #Pb1b1
                        0.5 * b2 * mu**0 * f_sub**0,#Pb1b2
                        0.5  * bG2 * mu**0 * f_sub**0,#Pb1bG2
                        0. * mu**0 * f_sub**0,#Pb2b2
                        0. * bG2 * mu**0 * f_sub**0, #Pb2bG2
                        0. * mu**0 * f_sub**0, #PbG2bG2
                        0.5 * mu**2 * f_sub,#mu**2 * f_sub * b1,
                        0* mu**2 * f_sub * b2,
                        0* mu**2 * f_sub * bG2,
                        0*(mu * f_sub * b1)**2,
                        0.5 *mu**2 * b1 * f_sub, #(mu * b1)**2 * f_sub,
                        0*mu**2 * f_sub * b1 * b2,
                        0* mu**2 * f_sub * b1 * bG2,
                        0*(mu * f_sub)**2 * b1, 
                        0*(mu * f_sub)**2 * b2,
                        0*(mu * f_sub)**2 * bG2,
                        0*(mu * f_sub)**4,
                        0*mu**4 * f_sub**3,
                        0*mu**4 * f_sub**3 * b1,
                        0*mu**4 * f_sub**2 * b2, 
                        0*mu**4 * f_sub**2 * bG2,
                        0.5 * mu**4 * f_sub**2, # mu**4 * f_sub**2 * b1
                        0*mu**4 * f_sub**2 * b1**2,
                        0*mu**4 * f_sub**2,
                        0*mu**6 * f_sub**4,
                        -0.5 * mu**4 * f_sub**2, #mu**6 * f_sub**3
                        0*mu**6 * f_sub**3 * b1,
                        0*mu**8 * f_sub**4,
                        0.5 * mu**4 * f_sub**2]) #THIS IS THE ADDITIONAL TERM
        
        kaisshape = Kaiser(b1, f_sub, mu)**0
        bias13Z3 = array([b1* kaisshape, #the *kaiser was due to Z1 that I dont have here
                        bG3 * kaisshape,
                        bG2 * kaisshape,
                        mu**2 * f_sub * kaisshape,
                        mu**2 * f_sub * b1 * kaisshape,
                        (mu * f_sub)**2 *kaisshape,
                        mu**4 * f_sub**2 * kaisshape])
        
        bias13F3 = array([1. * Kaiser(b1, f_sub, mu),
                        0 * Kaiser(b1, f_sub, mu),
                        0 * Kaiser(b1, f_sub, mu),
                        0 * Kaiser(b1, f_sub, mu),
                        0 * Kaiser(b1, f_sub, mu),
                        0 * Kaiser(b1, f_sub, mu),
                        0 * Kaiser(b1, f_sub, mu)])
        bias13 = 0.5* bias13Z3 + 0.5 * bias13F3
        # Use correct einsum
        if self.scale_dependent_growth:
            repl = 'ikl,ik->kl'
        else:
            repl = 'ijl,ik->kl'
        Pkmu_22_nw = einsum(repl, bias22, loop22_nw_sub)
        Pkmu_22_w  = einsum(repl, bias22, loop22_w)
        Pkmu_13_nw = einsum(repl, bias13, loop13_nw_sub)
        Pkmu_13_w  = einsum(repl, bias13, loop13_w)

        Pkmu_22 = Pkmu_22_nw + RSDdamp * Pkmu_22_w
        Pkmu_13 = Pkmu_13_nw + RSDdamp * Pkmu_13_w
        Pkmu    = PNLO + Pkmu_22 + Pkmu_13 + Pkmu_ctr

        return RectBivariateSpline(self.kL, self.mu, Pkmu)

#-------------------------------------------------------------------------------

    def Pl_gg_z_AP(self, kgrid, f=None, D=None, cosmo=None, **kwargs):
        r"""P kmu z

        Computes the non-linear galaxy power spectrum in redshift
        space at the proper redshift specified by the growth
        parameters, applying Alcock-Paczynski distortions. If
        `self.IRresum == False`, the power spectrum is not
        IR-resummed.

        Parameters
        ----------
        f, D:             float, growth rate and growth factor

        b1, b2, bG2, bG3: float, bias parameters
        c0, c2, c4, ck4:  float, EFT counterterms
        aP, e0k2, e2k2:   float, shot-noise parameters
        Psn:              float, Poisson shot noise

        Returns
        -------
        P(k, mu): interpolator object

        """
        f, D =  self._get_growth_functions(f=f, D=D, cosmo=cosmo)

        DZ2 = D*D
        DZ4 = D**4.

        mu = self.mu

        q, nu, AP_ampl = self._apply_AP_distortions(kgrid, self.mu, cosmo=cosmo)
        Pnw = self.Pnw_int(q)
        Pw  = self.Pw_int(q)
        loop22_nw = self.loop22_nw_int(q)
        loop22_w  = self.loop22_w_int(q)
        loop13_nw = self.loop13_nw_int(q)
        loop13_w  = self.loop13_w_int(q)

        # Rescale the wiggle and no-wiggle P(k)
        Pnw *= DZ2
        Pw *= DZ2

        # Rescaling of Sigma and dSigma2
        Sigma2 = self.Sigma2 * DZ2 * self.IRres
        dSigma2 = self.dSigma2 * DZ2 * self.IRres

        Sig2mu, RSDdamp = self._muxdamp(q, nu, Sigma2, dSigma2, f)

        # Compute loop nw and rescale to D**4
        loop22_nw_sub = loop22_nw * DZ4
        loop13_nw_sub = loop13_nw * DZ4
        loop22_w = loop22_w * DZ4
        loop13_w = loop13_w * DZ4

        # Bias parameters
        b1 = kwargs['b1'] if 'b1' in kwargs.keys() else 1
        b2 = kwargs['b2'] if 'b2' in kwargs.keys() else 0
        bG2 = kwargs['bG2'] if 'bG2' in kwargs.keys() else 0
        bG3 = kwargs['bG3'] if 'bG3' in kwargs.keys() else 0
        c0 = kwargs['c0'] if 'c0' in kwargs.keys() else 0
        c2 = kwargs['c2'] if 'c2' in kwargs.keys() else 0
        c4 = kwargs['c4'] if 'c4' in kwargs.keys() else 0
        ck4 = kwargs['ck4'] if 'ck4' in kwargs.keys() else 0
        aP = kwargs['aP'] if 'aP' in kwargs.keys() else 0
        e0k2 = kwargs['e0k2'] if 'e0k2' in kwargs.keys() else 0
        e2k2 = kwargs['e2k2'] if 'e2k2' in kwargs.keys() else 0
        Psn = kwargs['Psn'] if 'Psn' in kwargs.keys() else self.Psn

        def Kaiser(b1, f, nu):
            return b1 + f * nu**2

        # Next-to-leading order, counterterm, noise
        PNLO = Kaiser(b1, f, nu)**2 * (Pnw + RSDdamp * Pw * (1. + q**2 * Sig2mu))
        Pkmu_ctr = (-2. * (c0 + c2 * f * nu**2 + c4 * f * f * nu**4) *
                    q**2 + ck4 * f**4 * nu**4 * Kaiser(b1, f, nu)**2 *
                    q**4) * (Pnw + RSDdamp * Pw)
        Pkmu_noise = Psn * ((1. + aP) + (e0k2 + e2k2 * nu**2) * q**2)

        # Biases
        bias22 = array([b1**2 * nu**0, b1 * b2 * nu**0, b1 * bG2 * nu**0,
                        b2**2 * nu**0, b2 * bG2 * nu**0, bG2**2 * nu**0,
                        nu**2 * f * b1, nu**2 * f * b2, nu**2 * f * bG2,
                        (nu * f * b1)**2, (nu * b1)**2 * f,
                        nu**2 * f * b1 * b2, nu**2 * f * b1 * bG2,
                        (nu * f)**2 * b1, (nu * f)**2 * b2,
                        (nu * f)**2 * bG2, (nu * f)**4, nu**4 * f**3,
                        nu**4 * f**3 * b1, nu**4 * f**2 * b2,
                        nu**4 * f**2 * bG2, nu**4 * f**2 * b1,
                        nu**4 * f**2 * b1**2, nu**4 * f**2,
                        nu**6 * f**4, nu**6 * f**3, nu**6 * f**3 * b1,
                        nu**8 * f**4])
        bias13 = array([b1 * Kaiser(b1, f, nu), bG3 * Kaiser(b1, f, nu),
                        bG2 * Kaiser(b1, f, nu),
                        nu**2 * f * Kaiser(b1, f, nu),
                        nu**2 * f * b1 * Kaiser(b1, f, nu),
                        (nu * f)**2 * Kaiser(b1, f, nu),
                        nu**4 * f**2 * Kaiser(b1, f, nu)])

        Pkmu_22_nw = einsum('ijl,ikl->kl', bias22, loop22_nw_sub)
        Pkmu_22_w  = einsum('ijl,ikl->kl', bias22, loop22_w)
        Pkmu_13_nw = einsum('ijl,ikl->kl', bias13, loop13_nw_sub)
        Pkmu_13_w  = einsum('ijl,ikl->kl', bias13, loop13_w)

        Pkmu_22 = Pkmu_22_nw + RSDdamp * Pkmu_22_w
        Pkmu_13 = Pkmu_13_nw + RSDdamp * Pkmu_13_w
        Pkmu    = PNLO + Pkmu_22 + Pkmu_13 + Pkmu_ctr + Pkmu_noise

        Pell = array([simps((2*l+1.)/2.*legendre(l)(self.mu)*Pkmu, self.mu)
                      for l in [0,2,4]])

        return Pell*AP_ampl

#-------------------------------------------------------------------------------

    def tns_model(self, b1=1, sigma_v=0, f=None, D=None, cosmo=None):
        r"""tns_model

        Returns the TNS model for the anisotropic galaxy power spectrum.
        The FoG damping is implemented with Lorentzian

        Paramters
        ---------
        b1:      float, linear bias. Default 1
        sigma_v: float, velocity dispersion. Default 0
        f, D:    float, growth rate and growth factor
        cosmo:   dict, cosmology dictionary

        Returns
        -------
        ABsum*D_fog: array, anisotropic galaxy power spectrum
        """
        if cosmo is not None:
            PL = cosmo['PL']
        else:
            PL = self.PL

        f, D =  self._get_growth_functions(f=f, D=D, cosmo=cosmo)

        DZ2 = D*D
        PL *= DZ2
        A1, A3, A5, B0, B2, B4, B6, P_Ap1, P_Ap3, P_Ap5 = \
            self.fastpt.myRSD_components(PL, f, b1, C_window=.75)
        # check b1 in P_Ap1,2,3

        ABsum_mu2 = (self.kL * f * (A1 + P_Ap1) + (f * self.kL)**2 * B0)
        ABsum_mu4 = (self.kL * f * (A3 + P_Ap3) + (f * self.kL)**2 * B2)
        ABsum_mu6 = (self.kL * f * (A5 + P_Ap5) + (f * self.kL)**2 * B4)
        ABsum_mu8 = (f * self.kL)**2 * B6
        
        mu = self.mu
        D_fog = 1 / (1. + (self.kL[:,newaxis] * mu * sigma_v)**2/2.)

        ABsum = ABsum_mu2[:,newaxis] * mu**2 + ABsum_mu4[:,newaxis] * mu**4 + \
            ABsum_mu6[:,newaxis] * mu**6 + ABsum_mu8[:,newaxis] * mu**8

        def Kaiser(b1, f, nu):
            return b1 + f * nu**2

        # Leading order, delta-delta, delta-theta, theta-theta
        PLO = Kaiser(b1, f, mu)**2 * PL[:,newaxis]

        Pdd = mu**0 * self.fastpt.Pk_real_halo_full_one_loop(
            self.kL, PL, C_window=.75)[0][:,newaxis]
        Pdt = mu**2*f*b1 * self.fastpt.Pkmu_tns_dt(
            self.kL, PL, C_window=.75)[:,newaxis]
        Ptt = mu**4*f**2*self.fastpt.Pkmu_tns_tt(
            self.kL, PL, C_window=.75)[:,newaxis]
        P_spt = PLO + Pdd + Pdt + Ptt

        return (ABsum + P_spt) * D_fog

#-------------------------------------------------------------------------------

    def Pgg_real(self, b1=1, b2=0, bG2=0, bG3=0, c0=0, aP=0, ek2=0, Psn=0,
                 loop=True):
        """Pgg real

        Computes the galaxy power spectrum in real space given the
        full set of bias parameters, counterterm, and stochastic
        parameters. If the bias parameters are not specified, the real
        space matter power spectrum is computed.

        Parameters
        ----------
        `b1`:               float, linear bias
        `b2`, `bG2`, `bG3`: float, higher-order biases
        `c0`:               float, effective sound-speed
        `aP`:               float, constant correction to Poisson shot-noise
        `ek2`:              float, k**2 stochastic term
        `Psn`:              float, Poisson shot-noise
        `loop`:             bool, if True  -> 1-loop power spectrum;
                                  if False -> linear model

        Returns
        -------
        `P_gg(k)`:  array, galaxy power spectrum

        """
        return ((1. - float(loop))*b1*b1*self.Pk['LO'] +
                float(loop) * (b1*b1*self.Pk['NLO'] + b1*b2*self.Pk['b1b2'] +
                               b1*bG2*self.Pk['b1bG2'] + b2*b2*self.Pk['b2b2'] +
                               b2*bG2*self.Pk['b2bG2'] + bG2*bG2*self.Pk['bG2bG2'] +
                               b1*bG3*self.Pk['b1bG3'] + ek2*self.Pk['k2'] -
                               2.*c0*self.Pk['k2LO']) + (1.+aP)*Psn)

#-------------------------------------------------------------------------------

    def Pgm_real(self, b1=1, b2=0, bG2=0, bG3=0, c0=0, ek2=0, loop=True):
        """Pgm real

        Computes the galaxy-matter power spectrum in real space given
        a full set of bias parameters, counterterm, and stochastic
        parameters.

        Parameters
        ----------
        `b1`:               float, linear bias
        `b2`, `bG2`, `bG3`: float, higher-order biases
        `c0`:               float, effective sound-speed
        `ek2`:              float, k**2 stochastic term
        `loop`:             bool, if True -> 1-loop power spectrum;
                                  if False -> linear model

        Returns
        -------
        `P_gm(k)`: array, cross galaxy-matter power spectrum
        """
        return ((1. - float(loop)) * b1 * self.Pk['LO'] +
                float(loop) * (b1 * self.Pk['NLO'] +
                               0.5 * (b2 * self.Pk['b1b2'] +
                                      bG2*self.Pk['b1bG2'] +
                                      b1*bG3*self.Pk['b1bG3']) +
                               ek2*self.Pk['k2'] - 2. * c0 * self.Pk['k2LO']))

#-------------------------------------------------------------------------------

    def Pmm_real(self, c0=0, loop=True):
        """Pmm real

        Computes the matter power spectrum in real space given the
        effective sound-speed

        Parameters
        ----------
        `c0`:   float, effective sound-speed
        `loop`: bool, if True -> 1-loop power spectrum; if False -> linear model

        Returns
        -------
        `P_mm(k)`: array, the real space matter power spectrum
        """
        return ((1. - float(loop)) * self.Pk['LO'] +
                float(loop) * (self.Pk['NLO'] - 2. * c0 * self.Pk['k2LO']))

#-------------------------------------------------------------------------------

    def Pgg_l(self, l, f=None, **kwargs):
        """Pgg l

        Computes the galaxy power spectrum multipoles given the full
        set of bias parameters, counterterm amplitudes, and stochastic
        parameters.  By default, the growth rate is computed from the
        cosmology, but a different value can be specified.
        Parameters
        ----------
        `l`:                     int, multipole degree
        `b1`:                    float, linear bias
        `b2`, `bG2`, `bG3`:      float, higher-order biases
        `c0`, `c2`, `c4`, `ck4`: float, EFT counterterms
        `aP`:                    float, constant deviation from Poisson shot-noise
        `e0k2`, `e2k2`:          float, k**2 corrections to the shot-noise
        `Psn`:                   float, Poisson shot-noise
        `f`:                     None or float, growth rate

        Returns
        -------
        `P_gg,l(k)`: array, specified multipole of the galaxy power spectrum
        """
        if f == None:
            f = self.f

        # Bias parameters
        b1 = kwargs['b1'] if 'b1' in kwargs.keys() else 1
        b2 = kwargs['b2'] if 'b2' in kwargs.keys() else 0
        bG2 = kwargs['bG2'] if 'bG2' in kwargs.keys() else 0
        bG3 = kwargs['bG3'] if 'bG3' in kwargs.keys() else 0
        c0 = kwargs['c0'] if 'c0' in kwargs.keys() else 0
        c2 = kwargs['c2'] if 'c2' in kwargs.keys() else 0
        c4 = kwargs['c4'] if 'c4' in kwargs.keys() else 0
        ck4 = kwargs['ck4'] if 'ck4' in kwargs.keys() else 0
        aP = kwargs['aP'] if 'aP' in kwargs.keys() else 0
        e0k2 = kwargs['e0k2'] if 'e0k2' in kwargs.keys() else 0
        e2k2 = kwargs['e2k2'] if 'e2k2' in kwargs.keys() else 0
        Psn = kwargs['Psn'] if 'Psn' in kwargs.keys() else self.Psn

        Plk = self.P_ell_k[l]
        Pdet = (b1*b1*Plk['b1b1'] + f*b1*Plk['fb1'] + f*f*Plk['f2'] +
                b1*b2*Plk['b1b2'] + b1*bG2*Plk['b1bG2'] +
                b2*b2*Plk['b2b2'] + b2*bG2*Plk['b2bG2'] +
                bG2*bG2*Plk['bG2bG2'] + f*b2*Plk['fb2'] +
                f*bG2*Plk['fbG2'] + f*f*b1*b1*Plk['f2b12'] +
                f*b1*b1*Plk['fb12'] + f*b1*b2*Plk['fb1b2'] +
                f*b1*bG2*Plk['fb1bG2'] + f*f*b1*Plk['f2b1'] +
                f*f*b2*Plk['f2b2'] + f*f*bG2*Plk['f2bG2'] +
                f*f*f*f*Plk['f4'] + f*f*f*Plk['f3'] +
                f*f*f*b1*Plk['f3b1'] + b1*bG3*Plk['b1bG3'] +
                f*bG3*Plk['fbG3'] - 2.*c0*Plk['c0'] -
                2.*f*c2*Plk['fc2'] - 2.*f*f*c4*Plk['f2c4'] +
                f**4*b1**2*ck4*Plk['f4b12ck4'] +
                2.*f**5*b1*ck4*Plk['f5b1ck4'] + f**6*ck4*Plk['f6ck4'])

        Pstoch = Psn * ((l*l/8. - 0.75*l+1) * ((1. + aP) + e0k2 * Plk['e0k2']) +
                        (l - 0.25*l*l) * e2k2 * Plk['e2k2'])

        return Pdet+Pstoch

#-------------------------------------------------------------------------------
    def Pgg_l_noAP(self, k, b1,b2,bG2,bG3,c0,c2,c4,ck4,aP,e0k2,e2k2, Cosmo = None):

        #--- set cosmo
        cosmo = FlatLambdaCDM(H0=Cosmo['h']*100, Om0=Cosmo['Omh2']/Cosmo['h']**2, \
            Ob0=Cosmo['Obh2']/Cosmo['h']**2, Tcmb0=Cosmo['Tcmb'])

        start = time.time()
        _, PL = self.CallBACCO(cosmo = Cosmo)

        Cosmo['PL']= PL

        _, Dz, _ = self.growth_factor(cosmo=Cosmo)
        _, f  = self.growth_rate(cosmo=Cosmo)

        self._Pgg_kmu_terms(cosmo=Cosmo) #sets loops at z=0 for fixed cosmo

        [P0,P2,P4] = self.P_kmu_z(b1,b2,bG2,bG3,c0,c2,c4,ck4,aP,e0k2,e2k2,self.Psn,f,Dz,cosmo=Cosmo)

        P0_f = InterpolatedUnivariateSpline(self.kL, P0, k=3)
        P2_f = InterpolatedUnivariateSpline(self.kL, P2, k=3)
        P4_f = InterpolatedUnivariateSpline(self.kL, P4, k=3)

        return array([P0_f(k), P2_f(k), P4_f(k)])

#-------------------------------------------------------------------------------

    # def Pgg_l_marg(self, k, b1, b2, bG2, f=None, D=None):
    #     P_t, P_L, P_G = pbj.P_kmu_z_marg(k, mu, b1=1, b2=0, bG2=0, f=f, D=D)
    #     P_m = array([P_G, # P_kmu_bG3
    #                  -2.* P_L, # c0
    #                  (-2. * f * mu**2)[None,:] * P_L, # c2
    #                  (-2. * f**2 * mu_til**4)[None,:] * P_L, # c4
    #                  (f_L**4 * mu_til**4 * (b1 + f * mu**2)**2)[None,:] *
    #                  k**2 * P_L]) # ck4

    #     # Integrate P_kmu for multipoles
    #     P0 = scipy.integrate.simps((2*0+1.)/2.*legendre(0)(mu)*Pkmu_AP, mu)
    #     P2 = scipy.integrate.simps((2*2+1.)/2.*legendre(2)(mu)*Pkmu_AP, mu)
    #     P4 = scipy.integrate.simps((2*4+1.)/2.*legendre(4)(mu)*Pkmu_AP, mu)

    #     # Integrate P_kmu_marg for multipoles
    #     Pm_l = array([scipy.integrate.simps((2*l+1.)/2.*legendre(l)(mu)*Pm, mu)
    #                   for l in [0,2,4]])

    #     #------ from here
    #     #Parts that don't need to be integrated numerically
    #     # shot noise * AP amplitude
    #     PN_l = array([1,0,0])[:,None] * np.ones(len(k))[None,:]

    #     # Append noise to ctr
    #     Pm_l = np.append(Pm_l, PN_l[:,None,:], axis=1)

    #     # Scale dep SN
    #     Pe0_l = k1**2[None,:] * array([(2.*DA_r[z_in]**2 + H_r[z_in]**2)/3.,
    #                                    2.*(H_r[z_in]**2 - DA_r[z_in]**2)/3.,
    #                                    0])[:,None]
    #     Pm_l = np.append(Pm_l, Pe0_l[:,None,:], axis=1) # append e0k2

    #     Pe2_l = k**2[None,:] * array([1./3., 2./3., 0])[:,None]
    #     Pm_l = np.append(Pm_l, Pe2_l[:,None,:], axis=1) # append e2k2

    #     res_p = np.concatenate(P0, P2, P4)
    #     res_pm = np.concatenate(Pm_l[i] for i in [0,1,2])

    #     # return: 2 arrays with p_ell, p_ell_marg concatenated
    #     return res_p, res_pm

#-------------------------------------------------------------------------------

    def Bg_real(self, k1, k2, k3, b1=1, b2=0, bG2=0, a1=0, a2=0, Psn=0):
        """Bg real

        Computes the galaxy bispectrum at tree-level in real space
        given a set of bias and shot-noise parameters.

        Parameters
        ----------
        `k1`, `k2`, `k3`: float or array, sides of Fourier triangles
        `b1`:             float, linear bias
        `b2`, `bG2`:      float, higher-order biases
        `a1`, `a2`:       float, deviations from Poisson shot-noise
        `Psn`:            float, Poisson shot-noise

        Returns
        -------
        `Bggg(k1,k2,k3)`: array, real space galaxy bispectrum
        """
        mu12 = 0.5*(k3**2-k1**2-k2**2)/(k1*k2)
        mu23 = 0.5*(k1**2-k2**2-k3**2)/(k2*k3)
        mu31 = 0.5*(k2**2-k3**2-k1**2)/(k3*k1)

        F2_12 = 5./7 + 0.5*mu12*(k1/k2+k2/k1) + 2.*mu12**2/7
        F2_23 = 5./7 + 0.5*mu23*(k2/k3+k3/k2) + 2.*mu23**2/7
        F2_31 = 5./7 + 0.5*mu31*(k3/k1+k1/k3) + 2.*mu31**2/7

        S_12 = mu12**2 - 1.
        S_23 = mu23**2 - 1.
        S_31 = mu31**2 - 1.

        PkL = InterpolatedUnivariateSpline(self.kL, self.Pk['LO'], k=3)
        Pk1 = PkL(k1)
        Pk2 = PkL(k2)
        Pk3 = PkL(k3)

        return (2.*b1**2* ((b1*F2_12 + 0.5*b2 + bG2*S_12) * Pk1*Pk2 +
                           (b1*F2_23 + 0.5*b2 + bG2*S_23) * Pk2*Pk3 +
                           (b1*F2_31 + 0.5*b2 + bG2*S_31) * Pk3*Pk1) +
                b1**2 * (1. + a1)*(Pk1 + Pk2 + Pk3) * Psn + (1. + a2) * Psn**2)

#-------------------------------------------------------------------------------
    def Bispfunc_RSD_AP(self, k1, k2, k3, f=None, D=None, Cosmo = None, **kwargs):

        cosmo = FlatLambdaCDM(H0=Cosmo['h']*100, Om0=Cosmo['Omh2']/Cosmo['h']**2, \
            Ob0=Cosmo['Obh2']/Cosmo['h']**2, Tcmb0=Cosmo['Tcmb'])

        cosmo_fid = FlatLambdaCDM(H0=self.h*100, Om0=self.Omh2/self.h**2, \
            Ob0=self.Obh2/self.h**2, Tcmb0=self.Tcmb)

        #start = time.time()
        h0 = cosmo.H(0.).value/100.
        h0_fid = cosmo_fid.H(0.).value/100.
        #print('calc h0', time.time()-start)

        hz = cosmo.H(Cosmo['z']).value/100.
        hz_fid = cosmo_fid.H(self.z).value/100.

        #start = time.time()
        DA = cosmo.angular_diameter_distance(Cosmo['z']).value#/h0 #---why divide by h0?
        DA_fid = cosmo_fid.angular_diameter_distance(self.z).value#/h0_fid
        #print('calc DA', time.time()-start)

        a_perp = hz_fid/hz*h0/h0_fid
        a_orth = DA/DA_fid*h0/h0_fid

        #print(a_perp, a_orth)

        #------------------------------------------------------------
        order = 20
        xs = loadtxt('xs%iv2.txt' %order)
        Ws = loadtxt('Ws%iv2.txt' %order)
        #print(len(xs))
        xs1, xs2 = xs[0:order], xs[0:order]
        Ws1, Ws2 = Ws[0:order], Ws[0:order]

        integ_mat_0 = zeros((len(k1), len(xs1), len(xs2)))
        integ_mat_2 = zeros((len(k1), len(xs1), len(xs2)))

        if f is None or D is None:
            _, D, _ = self.growth_factor(cosmo=Cosmo)
            _, f  = self.growth_rate(cosmo=Cosmo)
        #start = time.time()
        if Cosmo is not None: PL = Cosmo['PL']
        else: PL = self.PL

        PL = D**2*PL

        #start = time.time()
        self.PkL = InterpolatedUnivariateSpline(self.kL, PL, k=3)
        #print('interpolate', time.time()-start)

        #start = time.time()
        self.DPkL = self.PkL.derivative(n=1)
        #print('derivative', time.time()-start)

        Pk1, Pk2, Pk3 = self.PkL(k1), self.PkL(k2), self.PkL(k3)
        DPk1, DPk2, DPk3 = self.DPkL(k1), self.DPkL(k2), self.DPkL(k3)

        '''
        _, PL = self.CallBACCO(cosmo = Cosmo)
        PL = self.Dz**2*PL

        kL_bisp = logspace(log(1.e-4), log(198), 1000)

        @nb.jit(nopython=True)
        def PkL(x):
            return interp(x, kL_bisp, PL)
        '''

        #start = time.time()

        ###-------please delete this line in actual run
        #self.Psn = 4693.9902
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # Bias parameters
        a1 = kwargs['a1'] if 'a1' in kwargs.keys() else 1
        b1 = kwargs['b1'] if 'b1' in kwargs.keys() else 1
        b2 = kwargs['b2'] if 'b2' in kwargs.keys() else 0
        bG2 = kwargs['bG2'] if 'bG2' in kwargs.keys() else 0

        # Get effective f
        if self.scale_dependent_growth:
            f_eff = f
        else:
            f_eff = f*np.ones_like(self.kL)
        Bl_terms_AP_(k1,k2,k3, self.kL, f_eff, a_perp,a_orth,b1,b2,bG2,a1,self.Psn, xs1,xs2, integ_mat_0,integ_mat_2, Pk1,Pk2,Pk3, DPk1,DPk2,DPk3)

        return [real(einsum('i,rij,j->r', Ws1, integ_mat_0, Ws2)), real(einsum('i,rij,j->r', Ws1, integ_mat_2, Ws2))]

#-------------------------------------------------------------------------------
    def p_ell(self, k,mu, D, f,b1,b2,bG2,bG3,c0,c2,c4,ck4,aP,e0k2,e2k2,Psn, Pw_func,Pnw_func,loop22_nw_sub_func,loop13_nw_sub_func,loop22_w_func,loop13_w_func):
        DZ2 = D*D
        DZ4 = D**4.

        # Get effective f
        if self.scale_dependent_growth:
            f_eff = interp1d(self.kL, f)(k)
        else:
            f_eff = f

        # Rescale and reshape the wiggle and no-wiggle P(k)
        Pnw = Pnw_func(k)
        Pw = Pw_func(k)

        Pnw_sub = Pnw
        Pw_sub = Pw
        ks_sub = k
        # print(ks_sub.shape) TODO_EB
        # print(f.shape)
        # if self.scale_dependent_growth:
        #     f_sub = f[:, newaxis]
        # else:
        #     f_sub = f
        #print('compute w/nw pk', time.time()-start)

        #print('shape(Pnw_sub)', shape(Pnw_sub))

        # Rescaling of Sigma and dSigma2
        start = time.time()
        Sigma2 = self.Sigma2 * DZ2 * self.IRres
        dSigma2 = self.dSigma2 * DZ2 * self.IRres
        Sig2mu, RSDdamp = self._muxdamp(ks_sub, mu, Sigma2, dSigma2, f_eff)
        #print('sigma', time.time()-start)

        # Compute loop nw and rescale to D**4
        start = time.time()
        '''
        loop22_nw_sub = array([loop22_nw_sub_func[i](k) for i in range(len(loop22_nw_sub_func))])
        loop13_nw_sub = array([loop13_nw_sub_func[i](k) for i in range(len(loop13_nw_sub_func))])
        loop22_w = array([loop22_w_func[i](k) for i in range(len(loop22_w_func))])
        loop13_w = array([loop13_w_func[i](k) for i in range(len(loop13_w_func))])
        '''
        #'''
        loop22_nw_sub = loop22_nw_sub_func(k)
        loop13_nw_sub = loop13_nw_sub_func(k)
        loop22_w = loop22_w_func(k)
        loop13_w = loop13_w_func(k)
        #'''
        '''
        loop22_nw_sub = map(loop22_nw_sub_func,k)
        loop13_nw_sub = map(loop13_nw_sub_func,k)
        loop22_w = map(loop22_w_func,k)
        loop13_w = map(loop13_w_func,k)
        '''
        #print('shape(loop13_w)', shape(loop13_w))
        #print('compute loop nw', time.time()-start)

        def Kaiser(b1, f, mu):
            return b1 + f*mu**2
        #print('shape(Kaiser)', shape(Kaiser(b1,f,mu)))

        # Next-to-leading order, counterterm, noise
        '''
        PNLO= Kaiser(b1, f, mu)**2 * (Pnw_sub + RSDdamp * Pw_sub *
                                    (1. + ks_sub**2 * Sig2mu))
        Pkmu_ctr = (-2. * (c0 + c2 * f * mu**2 + c4 * f * f * mu**4) *
                    ks_sub**2 + ck4 * f**4 * mu**4 * Kaiser(b1, f, mu)**2 *
                    ks_sub**4) * (Pnw_sub + RSDdamp * Pw_sub)
        Pkmu_noise = Psn * ((1. + aP) + (e0k2 + e2k2 * mu**2) * ks_sub**2)
        '''

        # Next-to-leading order, counterterm, noise
        if self.scale_dependent_growth:
            repl = 'ij, ij->ij'
        else:
            repl = 'j, ij->ij'
        PNLO = einsum(repl, Kaiser(b1, f_eff, mu)**2 , (Pnw_sub + RSDdamp * Pw_sub *
                                    (1. + ks_sub**2 * Sig2mu)))
        Pkmu_ctr = (einsum(repl, -2.*(c0 + c2*f_eff*mu**2 + c4*f_eff*f_eff*mu**4), ks_sub**2) +\
                    einsum(repl, ck4*f_eff**4*mu**4*Kaiser(b1, f_eff, mu)**2, ks_sub**4))* (Pnw_sub + RSDdamp * Pw_sub)

        Pkmu_noise = einsum('j,ij->ij', (e0k2 + e2k2 * mu**2), ks_sub**2)*Psn + Psn*(1.+aP)
        #print('shape(PNLO)', shape(PNLO))
        #print('next to leading order', time.time()-start)


        # Biases
        if self.scale_dependent_growth:
            dim = f_eff**0
        else:
            dim = mu**0
        bias22 = array([b1**2 * dim,
                        b1 * b2 * dim,
                        b1 * bG2 * dim,
                        b2**2 * dim,
                        b2 * bG2 * dim,
                        bG2**2 * dim,
                        mu**2 * f_eff * b1,
                        mu**2 * f_eff * b2,
                        mu**2 * f_eff * bG2,
                        (mu * f_eff * b1)**2,
                        (mu * b1)**2 * f_eff,
                        mu**2 * f_eff * b1 * b2,
                        mu**2 * f_eff * b1 * bG2,
                        (mu * f_eff)**2 * b1,
                        (mu * f_eff)**2 * b2,
                        (mu * f_eff)**2 * bG2,
                        (mu * f_eff)**4,
                        mu**4 * f_eff**3,
                        mu**4 * f_eff**3 * b1,
                        mu**4 * f_eff**2 * b2,
                        mu**4 * f_eff**2 * bG2,
                        mu**4 * f_eff**2 * b1,
                        mu**4 * f_eff**2 * b1**2,
                        mu**4 * f_eff**2,
                        mu**6 * f_eff**4,
                        mu**6 * f_eff**3,
                        mu**6 * f_eff**3 * b1,
                        mu**8 * f_eff**4])
        bias13 = array([b1 * Kaiser(b1, f_eff, mu),
                        bG3 * Kaiser(b1, f_eff, mu),
                        bG2 * Kaiser(b1, f_eff, mu),
                        mu**2 * f_eff * Kaiser(b1, f_eff, mu),
                        mu**2 * f_eff * b1 * Kaiser(b1, f_eff, mu),
                        (mu * f_eff)**2 * Kaiser(b1, f_eff, mu),
                        mu**4 * f_eff**2 * Kaiser(b1, f_eff, mu)])
        #print('biases', time.time()-start)
        #print('shape(bias22)', shape(bias22))
        '''
        #changed into a more general form below
        Pkmu_22_nw = sum(bias22*loop22_nw_sub)
        Pkmu_22_w  = sum(bias22*loop22_w)
        Pkmu_13_nw = sum(bias13*loop13_nw_sub)
        Pkmu_13_w  = sum(bias13*loop13_w)
        '''

        if self.scale_dependent_growth:
            repl = 'ijk, ijk->jk'
        else:
            repl = 'ik, ijk->jk'
        Pkmu_22_nw = einsum(repl, bias22,loop22_nw_sub)
        Pkmu_22_w  = einsum(repl, bias22,loop22_w)
        Pkmu_13_nw = einsum(repl, bias13,loop13_nw_sub)
        Pkmu_13_w  = einsum(repl, bias13,loop13_w)

        Pkmu_22 = Pkmu_22_nw + RSDdamp * Pkmu_22_w
        Pkmu_13 = Pkmu_13_nw + RSDdamp * Pkmu_13_w
        Pkmu    = PNLO + Pkmu_22 + Pkmu_13 + Pkmu_ctr + Pkmu_noise
        return Pkmu

#-------------------------------------------------------------------------------
    def Pl_terms_AP_(self, k, a_perp,a_orth, D, f,b1,b2,bG2,bG3,c0,c2,c4,ck4,aP,e0k2,e2k2,Psn, xs, integ_mat_0,integ_mat_2,integ_mat_4, Pw_func,Pnw_func,loop22_nw_sub_func,loop13_nw_sub_func,loop_22_w_func,loop13_w_func):
        nu = lambda mu: real(mu/a_perp*(mu**2/a_perp**2 + (1-mu**2)/a_orth**2)**(-1/2))
        #q_k = lambda k,mu: real(k*(mu**2/a_perp**2 + (1-mu**2)/a_orth**2)**(1/2))
        q_k = lambda k,mu: real(einsum('i,j->ij', k, (mu**2/a_perp**2 + (1-mu**2)/a_orth**2)**(1/2)))


        #integ = lambda ell, mu, k: self.p_ell(q_k(k, mu), nu(mu), \
        #                                            D, f,b1,b2,bG2,bG3,c0,c2,c4,ck4,aP,e0k2,e2k2,Psn, \
        #                                            Pw_func,Pnw_func,loop22_nw_sub_func,loop13_nw_sub_func,loop_22_w_func,loop13_w_func)*\
        #                                            legendre_poly(ell, mu)*(2*ell+1)/(4*pi)*(1/a_perp)**2*(1/a_orth)**4

        #integ = lambda ell, mu, k: einsum('ij,j->ij', self.p_ell(q_k(k, mu), nu(mu), \
        #                                            D, f,b1,b2,bG2,bG3,c0,c2,c4,ck4,aP,e0k2,e2k2,Psn, \
        #                                            Pw_func,Pnw_func,loop22_nw_sub_func,loop13_nw_sub_func,loop_22_w_func,loop13_w_func),
        #                                           legendre_poly(ell, mu)*(2*ell+1)/(4*pi)*(1/a_perp)**2*(1/a_orth)**4)

        integ = lambda mu, k: self.p_ell(q_k(k, mu), nu(mu), \
                                                    D, f,b1,b2,bG2,bG3,c0,c2,c4,ck4,aP,e0k2,e2k2,Psn, \
                                                    Pw_func,Pnw_func,loop22_nw_sub_func,loop13_nw_sub_func,loop_22_w_func,loop13_w_func)


        nmax, imax = shape(integ_mat_0)

        #for n in nb.prange(nmax):
        #    for i in nb.prange(imax):
        #        integ_mat_0[n,i] += integ(0, xs[i], k[n])
        #        integ_mat_2[n,i] += integ(2, xs[i], k[n])
        #       integ_mat_4[n,i] += integ(4, xs[i], k[n])

        #for i in nb.prange(imax):
        #    integ_mat_0[:,i] += integ(0, xs[i], k)
        #    integ_mat_2[:,i] += integ(2, xs[i], k)
        #    integ_mat_4[:,i] += integ(4, xs[i], k)

        #integ_mat_0 = integ(0, xs, k)
        #integ_mat_2 = integ(2, xs, k)
        #integ_mat_4 = integ(4, xs, k)

        integ_mat = integ(xs, k)
        integ_mat_l = einsum('ij,lj->lij', integ_mat, legendre_poly_024(xs)*(1/a_perp)*(1/a_orth)**2)
        integ_mat_0 += integ_mat_l[0, :, :]
        integ_mat_2 += integ_mat_l[1, :, :]
        integ_mat_4 += integ_mat_l[2, :, :]

        #[integ_mat_0, integ_mat_2, integ_mat_4] = einsum('ij,lj->lij', integ_mat, legendre_poly_024(xs)*1/(4*pi)*(1/a_perp)**2*(1/a_orth)**4)

#-------------------------------------------------------------------------------
    def Pgg_l_AP(self, k, f=None, D=None, Cosmo = None, **kwargs):

        #--- set cosmo
        cosmo = FlatLambdaCDM(H0=Cosmo['h']*100, Om0=Cosmo['Omh2']/Cosmo['h']**2, \
            Ob0=Cosmo['Obh2']/Cosmo['h']**2, Tcmb0=Cosmo['Tcmb'])

        cosmo_fid = FlatLambdaCDM(H0=self.h*100, Om0=self.Omh2/self.h**2, \
            Ob0=self.Obh2/self.h**2, Tcmb0=self.Tcmb)

        #--- calculate bunch of cosmological quantities
        #start = time.time()
        h0 = cosmo.H(0.).value/100.
        h0_fid = cosmo_fid.H(0.).value/100.
        #print('calc h0', time.time()-start)

        hz = cosmo.H(Cosmo['z']).value/100.
        hz_fid = cosmo_fid.H(self.z).value/100.

        #start = time.time()
        DA = cosmo.angular_diameter_distance(Cosmo['z']).value#/h0 #---why divide by h0?
        DA_fid = cosmo_fid.angular_diameter_distance(self.z).value#/h0_fid
        #print('calc DA', time.time()-start)

        a_perp = hz_fid/hz*h0/h0_fid
        a_orth = DA/DA_fid*h0/h0_fid
        #print(a_perp, a_orth)

        #--- upload weights and roots/preparing gauss-legendre quad
        order = 20
        xs = loadtxt('xs%iv2.txt' %order)
        Ws = loadtxt('Ws%iv2.txt' %order)

        xs = xs[0:order]
        Ws = Ws[0:order]

        integ_mat_0 = zeros((len(k), len(xs)))
        integ_mat_2 = zeros((len(k), len(xs)))
        integ_mat_4 = zeros((len(k), len(xs)))

        if f is None or D is None:
            _, D, _ = self.growth_factor(cosmo=Cosmo)
            _, f  = self.growth_rate(cosmo=Cosmo)

        self._Pgg_kmu_terms(cosmo=Cosmo) # sets loops at z=0 for fixed cosmo
        DZ2 = D*D
        DZ4 = D**4.

        # Rescale and reshape the wiggle and no-wiggle P(k)
        Pnw = self.Pnw * DZ2
        Pw = self.Pw * DZ2

        # Compute loop nw and rescale to D**4
        loop22_nw_sub = array([self.loop22_nw[i] for i in
                               range(len(self.loop22_nw))]) * DZ4
        loop13_nw_sub = array([self.loop13_nw[i] for i in
                               range(len(self.loop13_nw))]) * DZ4
        loop22_w = self.loop22_w * DZ4
        loop13_w = self.loop13_w * DZ4

        Pw_func = InterpolatedUnivariateSpline(self.kL, Pw, k=3)
        Pnw_func = InterpolatedUnivariateSpline(self.kL, Pnw, k=3)

        '''
        loop22_nw_sub_func = array([InterpolatedUnivariateSpline(self.kL, loop22_nw_sub[i, :], k=3) for i in range(len(loop22_nw_sub))])
        loop13_nw_sub_func = array([InterpolatedUnivariateSpline(self.kL, loop13_nw_sub[i, :], k=3) for i in range(len(loop13_nw_sub))])
        loop22_w_func = array([InterpolatedUnivariateSpline(self.kL, loop22_w[i, :], k=3) for i in range(len(loop22_w))])
        loop13_w_func = array([InterpolatedUnivariateSpline(self.kL, loop13_w[i, :], k=3) for i in range(len(loop13_w))])
        '''
        #'''
        loop22_nw_sub_func = lambda x: array([InterpolatedUnivariateSpline(self.kL, loop22_nw_sub[i, :], k=3)(x) for i in range(len(loop22_nw_sub))])
        loop13_nw_sub_func = lambda x: array([InterpolatedUnivariateSpline(self.kL, loop13_nw_sub[i, :], k=3)(x) for i in range(len(loop13_nw_sub))])
        loop22_w_func = lambda x: array([InterpolatedUnivariateSpline(self.kL, loop22_w[i, :], k=3)(x) for i in range(len(loop22_w))])
        loop13_w_func = lambda x: array([InterpolatedUnivariateSpline(self.kL, loop13_w[i, :], k=3)(x) for i in range(len(loop13_w))])
        #'''


        #print('interpolating the loop', time.time()-start)
        '''
        #---moved from pell
        # Rescale and reshape the wiggle and no-wiggle P(k)
        Pnw = Pnw_func(k)
        Pw = Pw_func(k)

        Pnw_sub = Pnw
        Pw_sub = Pw
        ks_sub = k

        # Rescaling of Sigma and dSigma2
        Sigma2 = self.Sigma2 * DZ2 * self.IRres
        dSigma2 = self.dSigma2 * DZ2 * self.IRres
        Sig2mu, RSDdamp = self._muxdamp(ks_sub, mu, Sigma2, dSigma2, f)
        #-----------------------------
        '''

        # Bias parameters
        b1 = kwargs['b1'] if 'b1' in kwargs.keys() else 1
        b2 = kwargs['b2'] if 'b2' in kwargs.keys() else 0
        bG2 = kwargs['bG2'] if 'bG2' in kwargs.keys() else 0
        bG3 = kwargs['bG3'] if 'bG3' in kwargs.keys() else 0
        c0 = kwargs['c0'] if 'c0' in kwargs.keys() else 0
        c2 = kwargs['c2'] if 'c2' in kwargs.keys() else 0
        c4 = kwargs['c4'] if 'c4' in kwargs.keys() else 0
        ck4 = kwargs['ck4'] if 'ck4' in kwargs.keys() else 0
        aP = kwargs['aP'] if 'aP' in kwargs.keys() else 0
        e0k2 = kwargs['e0k2'] if 'e0k2' in kwargs.keys() else 0
        e2k2 = kwargs['e2k2'] if 'e2k2' in kwargs.keys() else 0
        Psn = kwargs['Psn'] if 'Psn' in kwargs.keys() else self.Psn
        #start = time.time()
        #print('---- part 2. generate_integ')
        #please commented this line in actual run
        #self.Psn = 4693.9902
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.Pl_terms_AP_(k, a_perp,a_orth, D, f,b1,b2,bG2,bG3,c0,c2,c4,ck4,aP,e0k2,e2k2,self.Psn, xs, integ_mat_0,integ_mat_2,integ_mat_4, \
                        Pw_func,Pnw_func,loop22_nw_sub_func,loop13_nw_sub_func,loop22_w_func,loop13_w_func)

        #self.Pl_terms_AP_nb(k, a_perp,a_orth, self.Dz, self.f,b1,b2,bG2,bG3,c0,c2,c4,ck4,aP,e0k2,e2k2,the_SN, xs, integ_mat_0,integ_mat_2,integ_mat_4, \
        #                Pw_func,Pnw_func,loop22_nw_sub_func,loop13_nw_sub_func,loop22_w_func,loop13_w_func, Sig2mu,RSDdamp)

        #print('---- total part2. generate_integ', time.time()-start)

        start = time.time()
        result = array([real(einsum('i,ri->r', Ws, integ_mat_0)), real(einsum('i,ri->r', Ws, integ_mat_2)), real(einsum('i,ri->r', Ws, integ_mat_4))])
        #print('---- part3. multiplication')
        #print('gauss-quad for three mult', time.time()-start)

        return result
#-------------------------------------------------------------------------------

    def Pgg_l_AP_v2(self, k, b1,b2,bG2,bG3,c0,c2,c4,ck4,aP,e0k2,e2k2, f,  Cosmo = None):

        #define cosmology
        cosmo = FlatLambdaCDM(H0=Cosmo['h']*100, Om0=Cosmo['Omh2']/Cosmo['h']**2, \
            Ob0=Cosmo['Obh2']/Cosmo['h']**2, Tcmb0=Cosmo['Tcmb'])

        cosmo_fid = FlatLambdaCDM(H0=self.h*100, Om0=self.Omh2/self.h**2, \
            Ob0=self.Obh2/self.h**2, Tcmb0=self.Tcmb)
                #start = time.time()

        #define AP params
        h0 = cosmo.H(0.)/100.
        h0_fid = cosmo_fid.H(0.)/100.

        hz = cosmo.H(Cosmo['z'])/100.
        hz_fid = cosmo_fid.H(self.z)/100.

        DA = cosmo.angular_diameter_distance(Cosmo['z'])/h0
        DA_fid = cosmo_fid.angular_diameter_distance(self.z)/h0_fid

        a_perp = hz_fid/hz*h0/h0_fid
        a_orth = DA/DA_fid*h0/h0_fid

        #upload weights for gauss-quad
        #------------------------------------------------------------
        order = 20
        xs = loadtxt('xs%iv2.txt' %order)
        Ws = loadtxt('Ws%iv2.txt' %order)

        xs = xs[0:order]
        Ws = Ws[0:order]

        nmax, imax = len(k), len(xs)
        qh = zeros((nmax, imax))
        print(a_perp, a_orth)
        generate_qh(k,xs, a_perp,a_orth, qh)

        print(shape(qh))
        return qh

#-------------------------------------------------------------------------------
    def Bispfunc_RSD_simple(self, ell, k1, k2, k3, b1, b2, bG2, a1, f,  Cosmo = None):

        _, PL = self.CallBACCO(cosmo = Cosmo)
        PL = self.Dz**2*PL
        self.PkL = InterpolatedUnivariateSpline(self.kL, PL, k=3)

        Pk1, Pk2, Pk3 = self.PkL(k1), self.PkL(k2), self.PkL(k3)

        return (ell+1)*b1*b2*bG2*f*(Pk1*Pk2 + Pk2*Pk3 + Pk3*Pk1 + a1*self.Psn)
#-------------------------------------------------------------------------------
    def Bispfunc_RSD(self, k1, k2, k3):
        """Bispfunc RSD

        Computes the building blocks for the redshift-space bispectrum evaluated
        on the input momenta, as three lists for monopole, quadrupole and
        hexadecapole.

        Parameters
        ----------
        `k1`, `k2`, `k3`: float or array, sides of Fourier triangles

        Returns
        -------
        `B0`, `B2`, `B4`: arrays, each containing the building blocks for each
                                  bispectrum multipole
        """
        B0 = self.Bl_terms(0, k1, k2, k3)

        l2 = self.Bl_terms(2, k1, k2, k3)
        l4 = self.Bl_terms(4, k1, k2, k3)

        B2 = 2.5 * (3. * l2 - B0) / (2.*pi)
        B4 = 1.125 * (35. * l4 - 30. * l2 + 3. * B0) / (2.*pi)

        B0 /= (2.*pi)
        B0 = np.insert(B0, len(B0), ones(len(k1)), axis=0)

        return B0, B2, B4

#-------------------------------------------------------------------------------
    def Bispfunc_RSD_noAP(self, k1, k2, k3, b1, b2, bG2, a1, a2=0, Cosmo = None):

        cosmo = FlatLambdaCDM(H0=Cosmo['h']*100, Om0=Cosmo['Omh2']/Cosmo['h']**2, \
            Ob0=Cosmo['Obh2']/Cosmo['h']**2, Tcmb0=Cosmo['Tcmb'])

        _, PL = self.CallBACCO(cosmo = Cosmo)
        _, Dz, _ = self.growth_factor(cosmo=Cosmo)
        _, f  = self.growth_rate(cosmo=Cosmo)
        PL = Dz**2*PL

        self.PkL = InterpolatedUnivariateSpline(self.kL, PL, k=3)

        B0 =  self.Bl_terms(0, k1, k2, k3, PkL=self.PkL)
        l2 =  self.Bl_terms(2, k1, k2, k3, PkL=self.PkL)

        B2 = 2.5 * (3. * l2 - B0) / (2.*pi)

        B0 /= (2*pi)
        B0 = insert(B0, len(B0), ones(len(k1)), axis=0)

        alpha0 = array([b1*b1*b1, b1*b1*b2, b1*b1*bG2, f*b1*b1, f*b1*b1*b1, f*f*b1*b1,
                 f*b1*b2, f*b1*bG2, f*f*b1, f*f*f*b1, f*f*b2, f*f*bG2, f*f*f,
                 f*f*f*f, b1*b1*(1.+a1)*self.Psn, 0.5*f*b1*(1.+a1)*self.Psn,
                 f*f*0.*self.Psn, (1.+a2)*self.Psn2])
        alpha2 = array([b1*b1*b1, b1*b1*b2, b1*b1*bG2, f*b1*b1, f*b1*b1*b1, f*f*b1*b1,
                 f*b1*b2, f*b1*bG2, f*f*b1, f*f*f*b1, f*f*b2, f*f*bG2, f*f*f,
                 f*f*f*f, b1*b1*(1.+a1)*self.Psn, 0.5*f*b1*(1.+a1)*self.Psn,
                 f*f*0.*self.Psn])

        return [einsum('i,ij->j', alpha0, B0),  einsum('i,ij->j', alpha2, B2)]

#-------------------------------------------------------------------------------

    def Bl_terms(self, a, k1, k2, k3, PkL=None):
        """
        Parameters
        ----------
        `a`: int, power of mu1
        `k1`, `k2`, `k3`: float or array, sides of Fourier triangles

        Returns
        -------
        Array with the building blocks for the tree-level bispectrum
        multipoles in the following order
        """
        def muij(K):
            ki, kj, kl = K
            return 0.5*(kl**2 - ki**2 - kj**2)/(ki*kj)

        def F_2(ki,kj,kl):
            return (5/7. + 0.5*muij([ki,kj,kl]) * (ki/kj + kj/ki) +
                    2/7. * muij([ki,kj,kl])**2)

        def G_2(ki,kj,kl):
            return (3/7. + 0.5*muij([ki,kj,kl]) * (ki/kj + kj/ki) +
                    4/7. * muij([ki,kj,kl])**2)

        def Ia(bc, a, K):
            k1, k2, k3 = K
            if bc == '00':
                return 2.*pi*(1.+(-1)**a) / (1.+a)
            elif bc == '01':
                return 2.*pi*(-1.+(-1)**a)*(k1 + k2*muij(K)) / ((2.+a)*k3)
            elif bc == '02':
                return ((2.*pi * (1.+(-1)**a) *
                         ((1.+a)*k1**2 + 2.*(1.+a)*k1*k2*muij(K) +
                          k2**2*(1.+a*muij(K)**2))) / ((1.+a)*(3.+a)*k3**2))
            elif bc == '03':
                return ((2.*pi * (-1.+(-1)**a) * (k1 +  k2*muij(K)) *
                         ((2.+a)*k1**2 + 2.*(2.+a)*k1*k2*muij(K) +
                          k2**2*(3.+(a-1.)*muij(K)**2))) /
                        ((2.+a)*(4.+a)*k3**3))
            elif bc == '04':
                return ((2.*pi * (1.+(-1)**a) *
                         ((1.+a)*(3.+a)*k1**4 +
                          4*(1.+a)*(3.+a)*k1**3*k2*muij(K) +
                          4*(1.+a)*k1*k2**3*muij(K)*(3.+a*muij(K)**2) +
                          6*k1**2*k2**2*(1.+a+(1.+a)*(2.+a)*muij(K)**2) +
                          k2**4*(3.+6.*a*muij(K)**2+(a-2.)*a*muij(K)**4)))/
                        ((1.+a)*(3.+a)*(5.+a)*k3**4))
            elif bc == '05':
                return ((2.*pi * ((-1)**a-1.) * (k1 + k2 * muij(K)) *
                         ((2.+a) * (4.+a) * k1**4 +
                          4. * (2.+a) * (4.+a) * k1**3 * k2 * muij(K) +
                          4. * (2.+a) * k1 * k2**3 * muij(K) *
                          (5.+(a-1)*muij(K)**2) + 2.*(2.+a)*k1**2 * k2**2 *
                          (5. + (7 + 3.*a) * muij(K)**2) +
                          k2**4 * (15. + (a-1.) * muij(K)**2 *
                                   (10 + (a-3) * muij(K)**2)))) /
                        ((2.+a) * (4.+a) * (6.+a) * k3**5))

            elif bc == '10':
                return -((2.*pi * (-1.+(-1)**a) * muij(K)) / (2.+a))
            elif bc == '11':
                return -((2.*pi * (1.+(-1)**a) *
                          (k2 + (1.+a)*k1*muij(K) + a*k2*muij(K)**2)) /
                         ((1.+a)*(3.+a)*k3))
            elif bc == '12':
                return -((2.*pi * (-1.+(-1)**a) *
                          ((2.+a)*k1**2*muij(K) +
                           k2**2 * muij(K) * (3.+(a-1.) * muij(K)**2)+
                           2 * k1 * (k2 + (1.+a) * k2 * muij(K)**2))) /
                         ((2.+a) * (4.+a) * k3**2))
            elif bc == '13':
                return -((2.*pi * (1.+(-1)**a) *
                          ((1.+a)*(3.+a)*k1**3*muij(K) +
                           3.*(1.+a)*k1*k2**2*muij(K)*(3.+a*muij(K)**2)+
                           3.*k1**2*k2*(1.+a+(1.+a)*(2.+a)*muij(K)**2) +
                           k2**3*(3.+6.*a*muij(K)**2+(a-2.)*a*
                                  muij(K)**4))) / ((1.+a)*(3.+a)*(5.+a)*k3**3))
            elif bc == '14':
                return -((2.*pi * (-1.+(-1)**a) *
                          ((2.+a)*(4.+a)*k1**4*muij(K) +
                           6.*(2.+a)*k1**2*k2**2*muij(K)*
                           (3.+(1.+a)*muij(K)**2)+
                           4.*k1**3*k2*(2.+a+(2.+a)*(3.+a)*muij(K)**2) +
                           4.*k1*k2**3*(3.+6.*(1.+a)*muij(K)**2+
                                        (-1.+a**2)*muij(K)**4) +
                           k2**4*muij(K)*(15.+(-1.+a)*muij(K)**2*
                                                 (10.+(-3.+a)*muij(K)**2)))) /
                         ((2.+a)*(4.+a)*(6.+a)*k3**4))
            elif bc == '15':
                return -((2*pi * (1.+(-1)**a) *
                          ((1+a)*(3+a)*(5+a) * k1**5 * muij(K) +
                           10.*(1+a)*(3+a) * k1**3 * k2**2 * muij(K) *
                           (3+(2+a) * muij(K)**2) +
                           5.*(1+a)*(3+a) * k1**4 * k2 * (1.+(4+a)*muij(K)**2) +
                           5.*(1+a) * k1 * k2**4 * muij(K) *
                           (15.+10*a*muij(K)**2 + (a-2)*a*muij(K)**4) +
                           10.*(1+a) * k1**2 * k2**3 *
                           (3.+(2+a) * muij(K)**2 * (6.+a*muij(K)**2)) +
                           k2**5 * (15.+a * muij(K)**2 *
                                    (45.+(a-2)*muij(K)**2 *
                                     (15.+(a-4)*muij(K)**2))))) /
                         ((1+a)*(3+a)*(5+a)*(7+a) * k3**5))
            elif bc == '16':
                return -((2.*pi * ((-1)**a-1) *
                          ((2+a)*(4+a)*(6+a) * k1**6 * muij(K) +
                           15.*(2+a)*(4+a) * k1**4 * k2**2 * muij(K) *
                           (3.+(3+a) * muij(K)**2) +
                           6.*(2+a)*(4+a) * k1**5 * k2 * (1.+(5+a)*muij(K)**2) +
                           15.*(2+a) * k1**2 * k2**4 * muij(K) *
                           (15.+10*(1+a)*muij(K)**2 + (a**2-1)*muij(K)**4) +
                           20.*(2+a) * k1**3 * k2**3 *
                           (3.+(3+a)*muij(K)**2 * (6.+(1+a)*muij(K)**2)) +
                           k2**6*muij(K)*(105.+(a-1)*muij(K)**2 *
                                          (105+(a-3)*muij(K)**2 *
                                           (21+(a-3)*muij(K)**2))) +
                           6.*k1*k2**5 * (15.+(1+a)*muij(K)**2 *
                                          (45.+(a-1)*muij(K)**2 *
                                           (15+(a-3)*muij(K)**2))))) /
                         ((2+a)*(4+a)*(6+a)*(8+a) * k3**6))

            elif bc == '20':
                return ((2*pi*(1.+(-1)**a)*(1.+a*muij(K)**2)) / ((1.+a)*(3.+a)))
            elif bc == '21':
                return ((2.*pi * (-1.+(-1)**a) *
                         (k1 + (1.+a)*k1*muij(K)**2 +
                          k2*muij(K)*(3.+(a-1.)*muij(K)**2))) /
                        ((2.+a)*(4.+a)*k3))
            elif bc == '22':
                return ((2.*pi * (1.+(-1)**a) *
                         (2.*(1.+a)*k1*k2*muij(K)*(3.+a*muij(K)**2)+
                          k1**2*(1.+a+(1.+a)*(2.+a)*muij(K)**2) +
                          k2**2*(3.+6.*a*muij(K)**2+(-2.+a)*a*
                                 muij(K)**4))) / ((1.+a)*(3.+a)*(5.+a)*k3**2))
            elif bc == '23':
                return ((2.*pi * (-1.+(-1)**a) *
                         (3.*(2.+a)*k1**2*k2*muij(K)*
                          (3.+(1.+a)*muij(K)**2) +
                          k1**3*(2.+a+(2.+a)*(3.+a)*muij(K)**2) +
                          3.*k1*k2**2*(3.+6.*(1.+a)*muij(K)**2 +
                                       (a**2-1.)*muij(K)**4) +
                          k2**3*muij(K)*(15.+(a-1.)*muij(K)**2*
                                                (10.+(a-3.)*muij(K)**2)))) /
                        ((2.+a)*(4.+a)*(6.+a)*k3**3))
            elif bc == '24':
                return ((2.*pi * (1+(-1)**a) *
                         (4. * (1+a)*(3+a)*k1**3 * k2 * muij(K) *
                          (3.+(2+a)*muij(K)**2) + (1+a)*(3+a) * k1**4 *
                          (1.+(4+a)*muij(K)**2) + 4.*(1+a)*k1*k2**3*muij(K) *
                          (15.+10*a*muij(K)**2 + (a-2)*a*muij(K)**4) +
                          6.*(1+a)*k1**2 * k2**2 * (3.+(2+a)*muij(K)**2 *
                                                    (6.+a*muij(K)**2)) +
                          k2**4 * (15 + a*muij(K)**2 *
                                   (45.+(a-2)*muij(K)**2 *
                                    (15.+(a-4)*muij(K)**2))))) /
                        ((1+a)*(3+a)*(5+a)*(7+a)*k3**4))
            elif bc == '25':
                return ((2.*pi * ((-1)**a-1) *
                         (5.*(2+a)*(4+a) * k1**4 * k2 * muij(K) *
                          (3+(3+a)*muij(K)**2) + (2+a)*(4+a)*k1**5 *
                          (1.+(5+a)*muij(K)**2) +
                          10.*(2+a)*k1**2 * k2**3 * muij(K)*
                          (15.+10.*(1+a)*muij(K)**2 + (a**2-1)*muij(K)**4) +
                          10.*(2+a)*k1**3 * k2**2 * (3.+(3+a)*muij(K)**2 *
                                                     (6.+(1+a)*muij(K)**2)) +
                          k2**5 * muij(K) * (105.+(a-1)*muij(K)**2 *
                                             (105.+(a-3) * muij(K)**2 *
                                              (21.+(a-5) * muij(K)**2))) +
                          5.*k1 * k2**4 * (15.+(1+a)*muij(K)**2 *
                                           (45.+(a-1)*muij(K)**2 *
                                            (15.+(a-3)*muij(K)**2))))) /
                        ((2+a)*(4+a)*(6+a)*(8+a) * k3**5))

            elif bc == '30':
                return -((2.*pi * (-1.+(-1)**a) *
                          muij(K)*(3.+(a-1.) *
                                          muij(K)**2))/((2.+a)*(4.+a)))
            elif bc == '31':
                return -((2.*pi * (1.+(-1)**a) *
                          ((1.+a) * k1 * muij(K) * (3. + a * muij(K)**2) +
                           k2 * (3. + 6. * a * muij(K)**2 +
                                 (a-2.) * a * muij(K)**4))) /
                         ((1.+a) * (3.+a) * (5.+a) * k3))
            elif bc == '32':
                return -((2.*pi * (-1.+(-1)**a) *
                          ((2.+a)*k1**2*muij(K)*(3.+(1.+a)*muij(K)**2) +
                           2.*k1*k2*(3.+6.*(1.+a)*muij(K)**2+(a**2-1.)*
                                     muij(K)**4) +
                           k2**2*muij(K)*
                           (15.+(a-1.)*muij(K)**2*
                            (10.+(a-3.)*muij(K)**2)))) /
                         ((2.+a)*(4.+a)*(6.+a)*k3**2))
            elif bc == '33':
                return -((2.*pi * (1+(-1)**a) *
                          ((1+a)*(3+a)*k1**3 * muij(K) * (3.+(2+a) * muij(K)**2)+
                           3.*(1+a) * k1 * k2**2 * muij(K) *
                           (15. + 10*a*muij(K)**2 + (a-2)*a*muij(K)**4) +
                           3.*(1+a) * k1**2 * k2 * (3.+(2+a)*muij(K)**2 *
                                                    (6.+a*muij(K)**2)) +
                           k2**3 * (15.+a*muij(K)**2 *
                                    (45.+(a-2)*muij(K)**2 *
                                     (15.+(a-4)*muij(K)**2))))) /
                         ((1 + a)*(3 + a)*(5 + a)*(7 + a)*k3**3))
            elif bc == '34':
                return -((2.*pi * (-1.+(-1)**a) *
                          ((2.+a)*(4.+a)*k1**4*muij(K)*
                           (3.+(3.+a)*muij(K)**2) +
                           6.*(2.+a)*k1**2*k2**2*muij(K)*
                           (15.+10.*(1.+a)*muij(K)**2 + (a**2-1.)*
                            muij(K)**4) +
                           4.*(2.+a)*k1**3*k2*(3.+(3.+a)*muij(K)**2*
                                               (6.+(1.+a)*muij(K)**2)) +
                           k2**4*muij(K)*(105.+(a-1.)*muij(K)**2*
                                                 (105.+(a-3.)*muij(K)**2*
                                                  (21.+(a-5.)*muij(K)**2))) +
                           4.*k1*k2**3*(15.+(1.+a)*muij(K)**2*
                                        (45.+(a-1.)*muij(K)**2*
                                         (15.+(a-3.)*muij(K)**2))))) /
                         ((2.+a)*(4.+a)*(6.+a)*(8.+a)*k3**4))
            elif bc == '36':
                return -((2.*pi * ((-1)**a-1) *
                          ((2 + a)*(4 + a)*(6 + a)*k1**6*muij(K)*
                           (3 + (5 + a)*muij(K)**2) +
                           15*(2 + a)*(4 + a)*k1**4*k2**2*muij(K)*
                           (15 + (3 + a)*muij(K)**2*(10 + (1 + a)*muij(K)**2)) +
                           6*(2 + a)*(4 + a)*k1**5*k2*
                           (3 + (5 + a)*muij(K)**2*(6 + (3 + a)*muij(K)**2)) +
                           20*(2 + a)*k1**3*k2**3*
                           (15 + (3 + a)*muij(K)**2*
                            (45 + 15*(1 + a)*muij(K)**2 + (-1 + a**2)*muij(K)**4)) +
                           15*(2 + a)*k1**2*k2**4*muij(K)*
                           (105 + (1 + a)*muij(K)**2*
                            (105 + (-1 + a)*muij(K)**2*(21 + (-3 + a)*muij(K)**2))) +
                           k2**6*muij(K)*(945 + (-1 + a)*muij(K)**2*
                                          (1260 + (-3 + a)*muij(K)**2*
                                           (378 + (-5 + a)*muij(K)**2*
                                            (36 + (-7 + a)*muij(K)**2)))) +
                           6*k1*k2**5*(105 + (1 + a)*muij(K)**2*
                                       (420 + (-1 + a)*muij(K)**2*
                                        (210 + (a-3)*muij(K)**2 *
                                         (28 + (-5 + a)*muij(K)**2)))))) /
                         ((2 + a)*(4 + a)*(6 + a)*(8 + a)*(10 + a)*k3**6))

            elif bc == '40':
                return ((2.*pi * (1.+(-1)**a) *
                         ((a-2.)*a*muij(K)**4 + 6*a* muij(K)**2 + 3))/
                        ((1.+a)*(3.+a)*(5.+a)))
            elif bc == '41':
                return ((2.*pi * (-1.+(-1)**a) *
                         (k1*(3.+6*(1.+a)*muij(K)**2 + (-1.+a**2)*
                              muij(K)**4) +
                          k2*muij(K)*(15.+(-1.+a)*muij(K)**2*
                                             (10.+(-3.+a)*muij(K)**2)))) /
                        ((2.+a)*(4.+a)*(6.+a)*k3))
            elif bc == '42':
                return ((2.*pi * (1+(-1)**a)*
                         (2*(1+a)*k1*k2*muij(K)*
                          (15 + 10*a*muij(K)**2 + (a-2)*a*muij(K)**4) + 
                          (1+a)*k1**2 * (3.+(2+a)*muij(K)**2 *
                                         (6+a*muij(K)**2)) + 
                          k2**2*(15 + a*muij(K)**2*
                                 (45 + (a-2)*muij(K)**2 *
                                  (15 + (a-4)*muij(K)**2)))))/
                        ((1 + a)*(3 + a)*(5 + a)*(7 + a)*k3**2))
            elif bc == '43':
                return ((2.*pi * (-1.+(-1)**a) *
                         (3.*(2.+a)*k1**2*k2*muij(K)*
                          (15.+10.*(1.+a)*muij(K)**2 +
                           (a**2-1.)*muij(K)**4) + (2.+a)*k1**3*
                          (3.+(3.+a)*muij(K)**2*
                           (6.+(1.+a)*muij(K)**2)) + k2**3*muij(K)*
                          (105.+(a-1.)*muij(K)**2*
                           (105.+(a-3.)*muij(K)**2*
                            (21.+(a-5)*muij(K)**2))) +
                          3.*k1*k2**2*(15.+(1.+a)*muij(K)**2*
                                       (45.+(a-1.)*muij(K)**2*
                                        (15.+(a-3.)*muij(K)**2))))) /
                        ((2.+a)*(4.+a)*(6.+a)*(8.+a)*k3**3))
            elif bc == '45':
                return ((2.*pi * ((-1)**a-1) *
                         (5*(2 + a)*(4 + a)*k1**4*k2*muij(K)*
                          (15 + (3 + a)*muij(K)**2*(10 + (1 + a)*muij(K)**2)) + 
                          (2 + a)*(4 + a)*k1**5*
                          (3 + (5 + a)*muij(K)**2*(6 + (3 + a)*muij(K)**2)) + 
                          10*(2 + a)*k1**3*k2**2*
                          (15 + (3 + a)*muij(K)**2*
                           (45 + 15*(1 + a)*muij(K)**2 + (-1 + a**2)*muij(K)**4)) + 
                          10*(2 + a)*k1**2*k2**3*muij(K)*
                          (105 + (1 + a)*muij(K)**2*
                           (105 + (-1 + a)*muij(K)**2*(21 + (-3 + a)*muij(K)**2))) + 
                          k2**5*muij(K)*(945 + 
                                      (-1 + a)*muij(K)**2*
                                      (1260 + (-3 + a)*muij(K)**2 *
                                       (378 + (-5 + a)*muij(K)**2 *
                                        (36 + (-7 + a)*muij(K)**2)))) + 
                          5*k1*k2**4*(105 + 
                                      (1 + a)*muij(K)**2 *
                                      (420 + (-1 + a)*muij(K)**2 *
                                       (210 + (-3 + a)*muij(K)**2 *
                                        (28 + (-5 + a)*muij(K)**2)))))) /
                        ((2 + a)*(4 + a)*(6 + a)*(8 + a)*(10 + a)*k3**5))

            elif bc == '50':
                return -((2.*pi * ((-1)**a-1) *  muij(K) *
                          (15. + (a-1)*muij(K)**2*(10. + (a-3) * muij(K)**2))) /
                         ((2.+a) * (4.+a) * (6.+a)))
            elif bc == '51':
                return -((2.*pi * (1. + (-1)**a) *
                          ((1.+a) * k1 * muij(K) *
                           (15. + 10.*a*muij(K)**2 + (a-2.) * a * muij(K)**4) +
                           k2 * (15. + a * muij(K)**2 *
                                 (45. + (a-2) * muij(K)**2 *
                                  (15. + (a-4.) * muij(K)**2))))) /
                         ((1.+a) * (3.+a) * (5.+a) * (7.+a) * k3))
            elif bc == '52':
                return -((2.*pi * ((-1)**a-1) *
                          ((2 + a)*k1**2*muij(K) * (15 + 10*(1 + a)*muij(K)**2 +
                                                 (a**2-1)*muij(K)**4) + 
                           k2**2*muij(K)*(105 +  (a-1)*muij(K)**2 *
                                       (105 + (a-3)*muij(K)**2 *
                                        (21 + (a-5)*muij(K)**2))) + 
                           2*k1*k2*(15 + (1 + a)*muij(K)**2 *
                                    (45 + (a-1)*muij(K)**2 *
                                     (15 + (a-3)*muij(K)**2))))) /
                         ((2 + a)*(4 + a)*(6 + a)*(8 + a)*k3**2))
            elif bc == '54':
                return -((2.*pi * ((-1)**a-1) *
                          ((2 + a)*(4 + a)*k1**4*muij(K)*
                           (15 + (3 + a)*muij(K)**2*(10 + (1 + a)*muij(K)**2)) + 
                           4*(2 + a)*k1**3*k2* (15 + (3 + a)*muij(K)**2*
                                                (45 + 15*(1 + a)*muij(K)**2 +
                                                 (a**2-1)*muij(K)**4)) + 
                           6*(2 + a)*k1**2*k2**2*muij(K)*
                           (105 + (1 + a)*muij(K)**2 *
                            (105 + (-1 + a)*muij(K)**2 *
                             (21 + (-3 + a)*muij(K)**2))) + 
                           k2**4*muij(K)*(945 + (a-1)*muij(K)**2 *
                                       (1260 + (a-3)*muij(K)**2 *
                                        (378 + (a-5)*muij(K)**2 *
                                         (36 + (a-7)*muij(K)**2)))) + 
                           4*k1*k2**3*(105 + (1 + a)*muij(K)**2 *
                                       (420 + (a-1)*muij(K)**2 *
                                        (210 + (a-3)*muij(K)**2 *
                                         (28 + (a-5)*muij(K)**2)))))) /
                         ((2 + a)*(4 + a)*(6 + a)*(8 + a)*(10 + a)*k3**4))

            elif bc == '61':
                return ((2.*pi * ((-1)**a-1) *
                         (k2*muij(K)* (105 + (a-1)*muij(K)**2*
                                    (105 + (a-3)*muij(K)**2 *
                                     (21 + (a-5)*muij(K)**2))) + 
                          k1*(15 + (1 + a)*muij(K)**2 *
                              (45 + (a-1)*muij(K)**2 *
                               (15 + (a-3)*muij(K)**2))))) /
                        ((2 + a)*(4 + a)*(6 + a)*(8 + a)*k3))
            elif bc == '62':
                return ((2.*pi * (1 + (-1)**a) *
                         ((1 + a)*k1**2 * (15 + (2 + a)*muij(K)**2 *
                                           (45 + 15*a*muij(K)**2 + (a-2)*a*muij(K)**4)) + 
                          2*(1 + a)*k1*k2*muij(K) *
                          (105 + a*muij(K)**2 *
                           (105 + (a-2)*muij(K)**2 * (21 + (a-4)*muij(K)**2))) + 
                          k2**2*(105 + a*muij(K)**2 *
                                 (420 + (a-2)*muij(K)**2 *
                                  (210 + (a-4)*muij(K)**2 *
                                   (28 + (a-6)*muij(K)**2)))))) /
                        ((1 + a)*(3 + a)*(5 + a)*(7 + a)*(9 + a)*k3**2))
            
        mu12 = muij([k1, k2, k3])
        mu23 = muij([k2, k3, k1])
        mu31 = muij([k3, k1, k2])

        F_12 = F_2(k1, k2, k3)
        F_23 = F_2(k2, k3, k1)
        F_31 = F_2(k3, k1, k2)
        G_12 = G_2(k1, k2, k3)
        G_23 = G_2(k2, k3, k1)
        G_31 = G_2(k3, k1, k2)

        S_12 = mu12**2 - 1.
        S_23 = mu23**2 - 1.
        S_31 = mu31**2 - 1.

        if PkL == None:
            PkL = InterpolatedUnivariateSpline(self.kL, self.Pk['LO'], k=3)

        P12 = PkL(k1) * PkL(k2)
        P23 = PkL(k2) * PkL(k3)
        P31 = PkL(k3) * PkL(k1)

        K = [k1, k2, k3]

        x = [(x, y) for x in [0,1,2,3,4,5,6]
             for y in ['00', '01', '02', '03', '04', '05',
                       '10', '11', '12', '13', '14', '15', '16',
                       '20', '21', '22', '23', '24', '25',
                       '30', '31', '32', '33', '34', '36',
                       '40', '41', '42', '43', '45',
                       '50', '51', '52', '54',
                       '61', '63']]

        # Unrequested combinations can be removed as:
        #x.remove((4, '43'))

        # These are ordered as 'bca', so I['123'] is b=1, c=2, a=3
        I = dict((x[1] + str(x[0]), Ia(x[1], a + x[0], K)) for x in x)

        Bb13    = (F_12 * P12 + F_31 * P31 + F_23 * P23) * I['000']

        Bb12b2  = 0.5 * (P12 + P23 + P31) * I['000']

        Bb12bG2 = (S_12 * P12 + S_31 * P31 + S_23 * P23) * I['000']

        Bfb12   = (((I['002'] + I['200']) * F_12 + I['020'] * G_12) * P12 +
                   ((I['002'] + I['020']) * F_31 + I['200'] * G_31) * P31 +
                   ((I['200'] + I['020']) * F_23 + I['002'] * G_23) * P23)

        Bfb13   = -0.5 * (((k3/k1)*I['011'] + (k3/k2)*I['110']) * P12 +
                          ((k2/k1)*I['101'] + (k2/k3)*I['110']) * P31 +
                          ((k1/k2)*I['101'] + (k1/k3)*I['011']) * P23)

        # this needs a - sign
        Bf2b12  = -0.5 * (((k3/k1) * (I['013'] + 2. * I['211']) +
                           (k3/k2) * (I['310'] + 2. * I['112'])) * P12 +
                          ((k2/k1) * (I['103'] + 2. * I['121']) +
                           (k2/k3) * (I['130'] + 2. * I['112'])) * P31 +
                          ((k1/k2) * (I['301'] + 2. * I['121']) +
                           (k1/k3) * (I['031'] + 2. * I['211'])) * P23)

        Bfb1b2  = 0.5 * ((I['002'] + I['200']) * P12 +
                         (I['002'] + I['020']) * P31 +
                         (I['200'] + I['020']) * P23)

        Bfb1bG2 = ((I['002'] + I['200']) * S_12 * P12 +
                   (I['002'] + I['020']) * S_31 * P31 +
                   (I['200'] + I['020']) * S_23 * P23)

        Bf2b1   = ((I['202'] * F_12 + (I['022'] + I['220']) * G_12) * P12 +
                   (I['022'] * F_31 + (I['202'] + I['220']) * G_31) * P31 +
                   (I['220'] * F_23 + (I['202'] + I['022']) * G_23) * P23)

        # This also needs a - sign
        Bf3b1   = -0.5*(((k3/k1) * (I['411'] + 2. * I['213']) +
                         (k3/k2) * (I['114'] + 2. * I['312'])) * P12 +
                        ((k2/k1) * (I['141'] + 2. * I['123']) +
                         (k2/k3) * (I['114'] + 2. * I['132'])) * P31 +
                        ((k1/k2) * (I['141'] + 2. * I['321']) +
                         (k1/k3) * (I['411'] + 2. * I['231'])) * P23)

        Bf2b2   = 0.5 * (I['202'] * P12 + I['022'] * P31 + I['220'] * P23)

        Bf2bG2  = (I['202'] * S_12 * P12 +
                   I['022'] * S_31 * P31 +
                   I['220'] * S_23 * P23)

        Bf3     = (G_12 * P12 + G_31 * P31 + G_23 * P23) * I['222']

        # this also needs a - sign
        Bf4     = -0.5 * (((k3/k1) * I['413'] + (k3/k2) * I['314']) * P12 +
                          ((k2/k1) * I['143'] + (k2/k3) * I['134']) * P31 +
                          ((k1/k2) * I['341'] + (k1/k3) * I['431']) * P23)

        Bb12a1  = 0.5 * (PkL(k1) + PkL(k2) + PkL(k3)) * I['000']

        Bfb1a1  = (I['002'] * PkL(k1) + I['200'] * PkL(k2) + I['020'] * PkL(k3))

        Bf2a1   = 0.5 * (I['004'] * PkL(k1) +
                         I['400'] * PkL(k2) +
                         I['040'] * PkL(k3))

        # # AP expansion terms
        # Bfb12F = -2 * (Bfb12 - (((I['004'] + I['400']) * F_12 + I['040'] * G_12) * P12 +
        #                         ((I['004'] + I['040']) * F_13 + I['400'] * G_13) * P13 +
        #                         ((I['400'] + I['040']) * F_23 + I['004'] * G_23) * P23))
        # Bfb13F = -2 * (Bfb13 + 0.25 * (((I['013'] + I['031'])/k1 +
        #                                 (I['310'] + I['130'])/k2) * k3 * P12 +
        #                                ((I['103'] + I['301'])/k1 +
        #                                 (I['310'] + I['130'])/k3) * k2 * P13 +
        #                                ((I['103'] + I['301'])/k2 +
        #                                 (I['013'] + I['031'])/k3) * k1 * P23))

        return array((Bb13, Bb12b2, Bb12bG2, Bfb12, Bfb13, Bf2b12, Bfb1b2,
                      Bfb1bG2, Bf2b1, Bf3b1, Bf2b2, Bf2bG2, Bf3, Bf4, Bb12a1,
                      Bfb1a1, Bf2a1))

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

try:    import fastpt.FASTPT_simple as fpt
except:    import FASTPT_simple as fpt

class FASTPTPlus(fpt.FASTPT):
    """FASTPTPlus

    Inherits from fastpt.FASTPT. Implements methods to compute the
    loop corrections for the redshift space galaxy power spectrum.
    """

    def Pkmu_22_one_loop_terms(self, K, P, P_window=None, C_window=None, with_cross=False):
        """
        Computes the mode-coupling loop corrections for the redshift-space
        galaxy power spectrum

        Parameters
        ----------
        K: array of floats, Fourier wavenumbers, log-spaced
        P: array of floats, power spectrum

        Returns
        -------
        Arrays with the building blocks of the mode-coupling loop corrections in
        the following order:
        `Pb1b1, Pb1b2, Pb1bG2, Pb2b2, Pb2bG2, PbG2bG2, Pmu2fb1, Pmu2fb2,
        Pmu2fbG2, Pmu2f2b12, Pmu2fb12, Pmu2fb1b2, Pmu2fb1bG2, Pmu2f2b1,
        Pmu2f2b2, Pmu2f2bG2, Pmu4f4, Pmu4f3,Pmu4f3b1,Pmu4f2b2,Pmu4f2bG2,
        Pmu4f2b1, Pmu4f2b12, Pmu4f2, Pmu6f4,Pmu6f3,Pmu6f3b1,Pmu8f4`
        """
        Power, MAT = self.J_k(P, P_window=P_window, C_window=C_window)
        J000, J002, J004, J2m22, J1m11, J1m13, J2m20r = MAT

        Pb1b1      = (1219/735.*J000 + J2m20r/3. + 124/35.*J1m11 + 2/3.*J2m22 +
                      1342/1029.*J002 + 16/35.*J1m13 + 64/1715.*J004)
        Pb1b2      = 34/21.*J000 + 2.*J1m11 + 8/21.*J002
        Pb1bG2     = (-72/35.*J000 + 8/5.*(J1m13 - J1m11) + 88/49.*J002 +
                      64/245.*J004)
        Pb2b2      = 0.5 * (J000 - J000[0])
        Pb2bG2     = 4/3. * (J002 - J000)
        PbG2bG2    = 16/15.*J000 - 32/21.*J002 + 16/35.*J004

        Pmu2fb1    = 2.*(1003/735.*J000 + J2m20r/3. + 116/35.*J1m11 +
                         1606/1029.*J002 + 2/3.*J2m22 + 24/35.*J1m13 +
                         128/1715*J004)
        Pmu2fb2    = 26/21.*J000 + 2.*J1m11 + 16/21.*J002
        Pmu2fbG2   = (-152/105.*J000 + 8/5.*(J1m13 - J1m11) + 136/147.*J002 +
                      128/245.*J004)
        Pmu2f2b12  = (- J000 + J2m20r + J002 - J2m22)/3.
        Pmu2fb12   = (82/21.*J000 + 2/3.*J2m20r + 264/35.*J1m11 + 44/21.*J002 +
                      4/3.*J2m22 + 16/35.*J1m13)
        Pmu2fb1b2  = 2. * (J000 + J1m11)
        Pmu2fb1bG2 = 8. * ((J002- J000)/3. + (J1m13 - J1m11)/5.)
        Pmu2f2b1   = 0.25 * Pb1bG2
        Pmu2f2b2   = 0.25 * Pb2bG2
        Pmu2f2bG2  = 0.50 * PbG2bG2

        Pmu4f4     = 3/32. * PbG2bG2
        Pmu4f3     = Pmu2fbG2 / 4.
        Pmu4f3b1   = 4/3.*(J002-J000) + 2/3.*(J2m20r-J2m22) + 2/5.*(J1m13-J1m11)
        Pmu4f2b2   = 5/3.*J000 + 2.*J1m11 + J002/3.
        Pmu4f2bG2  = 8/5.*(J1m13-J1m11) - 32/15.*J000 + 40/21.*J002 + 8/35.*J004
        Pmu4f2b1   = (98/15.*J000 + 4/3.*J2m20r + 498/35.*J1m11 + 794/147.*J002+
                      8/3.*J2m22 + 62/35.*J1m13 + 16/245.*J004)
        Pmu4f2b12  = 8/3.*J000 + 4.*J1m11 + J002/3. + J2m22
        Pmu4f2     = (851/735.*J000 + J2m20r/3. + 108/35.*J1m11+1742/1029.*J002+
                      2/3.*J2m22 + 32/35.*J1m13 + 256/1715.*J004)

        Pmu6f4     = (-14/15.*J000 + (J2m20r - J2m22)/3. + 2/5.*(J1m13 - J1m11)+
                      19/21.*J002 + J004/35.)
        Pmu6f3     = (292/105.*J000 + 2/3.*J2m20r + 234/35.*J1m11 +
                      454/147.*J002 + 4/3.*J2m22 + 46/35.*J1m13 + 32/245.*J004)
        Pmu6f3b1   = 14/3.*J000 + 38/5.*J1m11 + 4/3.*J002 +2.*J2m22 + 2/5.*J1m13

        Pmu8f4     = (21/10.*J000 + 18/5.*J1m11 + 6/7.*J002 + J2m22 +
                      2/5.*J1m13 + 3/70.*J004)

        if(self.extrap):
            _,Pb1b1      = self.EK.PK_original(Pb1b1)
            _,Pb1b2      = self.EK.PK_original(Pb1b2)
            _,Pb1bG2     = self.EK.PK_original(Pb1bG2)
            _,Pb2b2      = self.EK.PK_original(Pb2b2)
            _,Pb2bG2     = self.EK.PK_original(Pb2bG2)
            _,PbG2bG2    = self.EK.PK_original(PbG2bG2)

            _,Pmu2fb1    = self.EK.PK_original(Pmu2fb1)
            _,Pmu2fb2    = self.EK.PK_original(Pmu2fb2)
            _,Pmu2fbG2   = self.EK.PK_original(Pmu2fbG2)
            _,Pmu2f2b12  = self.EK.PK_original(Pmu2f2b12)
            _,Pmu2fb12   = self.EK.PK_original(Pmu2fb12)
            _,Pmu2fb1b2  = self.EK.PK_original(Pmu2fb1b2)
            _,Pmu2fb1bG2 = self.EK.PK_original(Pmu2fb1bG2)
            _,Pmu2f2b1   = self.EK.PK_original(Pmu2f2b1)
            _,Pmu2f2b2   = self.EK.PK_original(Pmu2f2b2)
            _,Pmu2f2bG2  = self.EK.PK_original(Pmu2f2bG2)

            _,Pmu4f4     = self.EK.PK_original(Pmu4f4)
            _,Pmu4f3     = self.EK.PK_original(Pmu4f3)
            _,Pmu4f3b1   = self.EK.PK_original(Pmu4f3b1)
            _,Pmu4f2b2   = self.EK.PK_original(Pmu4f2b2)
            _,Pmu4f2bG2  = self.EK.PK_original(Pmu4f2bG2)
            _,Pmu4f2b1   = self.EK.PK_original(Pmu4f2b1)
            _,Pmu4f2b12  = self.EK.PK_original(Pmu4f2b12)
            _,Pmu4f2     = self.EK.PK_original(Pmu4f2)

            _,Pmu6f4     = self.EK.PK_original(Pmu6f4)
            _,Pmu6f3     = self.EK.PK_original(Pmu6f3)
            _,Pmu6f3b1   = self.EK.PK_original(Pmu6f3b1)

            _,Pmu8f4     = self.EK.PK_original(Pmu8f4)

        if with_cross:
            Pcrossaddterm = (-38/105.*J000 + 34/147.*J002+ 32/245*J004+2/5.*(-J1m11+J1m13))
            if(self.extrap):
                _,Pcrossaddterm      = self.EK.PK_original(Pcrossaddterm)
            return Pb1b1, Pb1b2, Pb1bG2, Pb2b2, Pb2bG2, PbG2bG2, Pmu2fb1, Pmu2fb2, \
                Pmu2fbG2, Pmu2f2b12, Pmu2fb12, Pmu2fb1b2, Pmu2fb1bG2, Pmu2f2b1, \
                Pmu2f2b2, Pmu2f2bG2, Pmu4f4, Pmu4f3,Pmu4f3b1,Pmu4f2b2,Pmu4f2bG2, \
                Pmu4f2b1, Pmu4f2b12, Pmu4f2, Pmu6f4,Pmu6f3,Pmu6f3b1,Pmu8f4, Pcrossaddterm

        return Pb1b1, Pb1b2, Pb1bG2, Pb2b2, Pb2bG2, PbG2bG2, Pmu2fb1, Pmu2fb2, \
            Pmu2fbG2, Pmu2f2b12, Pmu2fb12, Pmu2fb1b2, Pmu2fb1bG2, Pmu2f2b1, \
            Pmu2f2b2, Pmu2f2bG2, Pmu4f4, Pmu4f3,Pmu4f3b1,Pmu4f2b2,Pmu4f2bG2, \
            Pmu4f2b1, Pmu4f2b12, Pmu4f2, Pmu6f4,Pmu6f3,Pmu6f3b1,Pmu8f4

#-------------------------------------------------------------------------------

    def Pkmu_13_one_loop_terms(self, K, P):
        """
        Computes the propagator loop corrections for the redshift-space galaxy
        power spectrum

        Parameters
        ----------
        K: array of floats, Fourier wavenumbers, log-spaced
        P: array of floats, power spectrum

        Returns
        -------
        Arrays for the building blocks of the propagator loop corrections in
        the following order:
        `PZ1b1, PZ1bG3, PZ1bG2, PZ1mu2f, PZ1mu2fb1, PZ1mu2f2, PZ1mu4f2`
        """
        PZ1b1     = self.P_dd_13_reg(K,P)
        PZ1bG3    = self.P_b1bG3(K,P)
        PZ1bG2    = 5./2.*PZ1bG3
        PZ1mu2f   = self.P_tt_13_reg(K,P)
        PZ1mu2fb1 = self.P_mu2fb1_13(K,P)
        PZ1mu2f2  = 3./8.*PZ1bG3
        PZ1mu4f2  = self.P_mu4f2_13(K,P)

        return PZ1b1, PZ1bG3, PZ1bG2, PZ1mu2f, PZ1mu2fb1, PZ1mu2f2, PZ1mu4f2

    
#-------------------------------------------------------------------------------

    def Pkmu_tns_dt(self, K, P, P_window=None, C_window=None):
        """
        Computes the delta-theta contribution to the one-loop power spectrum in
        standard perturbation theory using Fast-PT
        """        
        Power, MAT = self.J_k(P, P_window=P_window, C_window=C_window)
        J000, J002, J004, J2m22, J1m11, J1m13, J2m20r = MAT

        Pmu2fb1 = 2.*(1003/735.*J000 + J2m20r/3. + 116/35.*J1m11 +
                      1606/1029.*J002 + 2/3.*J2m22 + 24/35.*J1m13 +
                      128/1715*J004)

        if(self.extrap):
            _,Pmu2fb1    = self.EK.PK_original(Pmu2fb1)

        PZ1mu2f = self.P_tt_13_reg(K,P)
        PZ1b1   = self.P_dd_13_reg(K,P)

        return Pmu2fb1 + PZ1mu2f + PZ1b1

#-------------------------------------------------------------------------------

    def Pkmu_tns_tt(self, K, P, P_window=None, C_window=None):
        """
        Computes the theta-theta contribution to the one-loop power spectrum in
        standard perturbation theory using Fast-PT
        """        
        Power, MAT = self.J_k(P, P_window=P_window, C_window=C_window)
        J000, J002, J004, J2m22, J1m11, J1m13, J2m20r = MAT

        Pmu4f2 = (851/735.*J000 + J2m20r/3. + 108/35.*J1m11+1742/1029.*J002+
                      2/3.*J2m22 + 32/35.*J1m13 + 256/1715.*J004)

        if(self.extrap):
            _,Pmu4f2 = self.EK.PK_original(Pmu4f2)

        PZ1mu2f = self.P_tt_13_reg(K,P)

        return Pmu4f2 + PZ1mu2f

#-------------------------------------------------------------------------------

    def Pk_real_halo_full_one_loop(self, K, P, P_window=None, C_window=None):
        """
        Computes the mode-coupling loop corrections for the real-space galaxy
        power spectrum

        Parameters
        ----------
        K: array of floats, Fourier wavenumbers, log-spaced
        P: array of floats, power spectrum

        Returns
        -------
        Arrays with the building blocks for the mode coupling loop corrections
        for the real space galaxy power spectrum in the following order:
        `(P_dd_13 + P_dd_22), Pb1b2, (Pb1g2mc + Pb1g2pr), Pb2b2, Pb2g2, Pg2g2, Pb1g3`
        """
        Power, MAT = self.J_k(P,P_window=P_window,C_window=C_window)
        J000, J002, J004, J2m22, J1m11, J1m13, J2m20r = MAT

        P_dd_22 = (1219/735.*J000 + 1342/1029.*J002 + 64/1715.*J004 +
                   124/35.*J1m11 + 16/35.*J1m13 + J2m20r/3. + 2/3.*J2m22)
        P_dd_13 = self.P_dd_13_reg(K, P)

        Pb1b2   = 34/21.*J000 + 2.*J1m11 + 8/21.*J002
        Pb1g2mc = (-72/35.*J000 -8/5.*J1m11 + 88/49.*J002 + 8/5.*J1m13 +
                   64/245.*J004)
        Pb2b2   = 0.5 * (J000-J000[0])
        Pb2g2   = 4/3.*(J002 - J000)
        Pg2g2   = 16/35.*J004 - 32/21.*J002 + 16/15.*J000

        Pb1g3   = self.P_b1bG3(K, P)
        Pb1g2pr = 2.5*Pb1g3

        if(self.extrap):
            _,P_dd_22   = self.EK.PK_original(P_dd_22)
            _,Pb1b2     = self.EK.PK_original(Pb1b2)
            _,Pb1g2mc   = self.EK.PK_original(Pb1g2mc)
            _,Pb2b2     = self.EK.PK_original(Pb2b2)
            _,Pb2g2     = self.EK.PK_original(Pb2g2)
            _,Pg2g2     = self.EK.PK_original(Pg2g2)

        # return array([(P_dd_13 + P_dd_22), Pb1b2, Pb1g2mc,
        #               Pb2b2, Pb2g2, Pg2g2, Pb1g3])
        return array([(P_dd_13 + P_dd_22), Pb1b2, (Pb1g2mc + Pb1g2pr),
                      Pb2b2, Pb2g2, Pg2g2, Pb1g3])

#-------------------------------------------------------------------------------

    def P_dd_13_reg(self, k,P):
        """
        Computes the regularized version of P_13 for the matter power spectrum

        Parameters
        ----------
        K: array of floats, Fourier wavenumbers, log-spaced
        P: array of floats, power spectrum

        Returns
        -------
        P_bar: array with the regularizer version of P_13
        """
        N   = k.size
        n   = arange(-N+1,N )
        dL  = log(k[1])-log(k[0])
        s   = n*dL
        cut = 7
        high_s = s[s > cut]
        low_s  = s[s < -cut]
        mid_high_s = s[(s <= cut)  & (s > 0)]
        mid_low_s  = s[(s >= -cut) & (s < 0)]

        Z=lambda r : (12./r**2+10.+100.*r**2-42.*r**4+3./r**3*(r**2-1.)**3*(7*r**2+2.)*log((r+1.)/absolute(r-1.)))*r
        Z_low=lambda r : (352/5.+96/.5/r**2-160/21./r**4-526/105./r**6+236/35./r**8)*r
        Z_high=lambda r: (928/5.*r**2-4512/35.*r**4+416/21.*r**6+356/105.*r**8)*r

        f_mid_low  = Z(exp(-mid_low_s))
        f_mid_high = Z(exp(-mid_high_s))
        f_high     = Z_high(exp(-high_s))
        f_low      = Z_low(exp(-low_s))

        f = hstack((f_low,f_mid_low,80,f_mid_high,f_high))
        g = fftconvolve(P, f) * dL
        g_k = g[N-1:2*N-1]
        P_bar =  1/252. * k**3 / (2*pi)**2 * P * g_k

        return P_bar

#-------------------------------------------------------------------------------

    def P_b1bG3(self,k, P):
        """
        Computes the contribution P_b1bG3 for the galaxy power spectrum

        Parameters
        ----------
        K: array of floats, Fourier wavenumbers, log-spaced
        P: array of floats, power spectrum

        Returns
        -------
        Pb1g3_bar: array, P_b1bG3 contribution to the galaxy power spectrum
        """

        N   = k.size
        n   = arange(-N+1,N)
        dL  = log(k[1])-log(k[0])
        s   = n * dL
        cut = 7
        high_s     = s[s>cut]
        low_s      = s[s<-cut]
        mid_high_s = s[(s<=cut)  & (s>0)]
        mid_low_s  = s[(s>=-cut) & (s<0)]

        Zb1g3      = lambda r: r*(-12./r**2+44.+44.*r**2-12*r**4+6./r**3*(r**2-1.)**4*log((r+1.)/absolute(r-1.)))
        Zb1g3_low  = lambda r: r*(512/5.-1536/35./r**2+512/105./r**4+512/1155./r**6 +512/5005./r**8)
        Zb1g3_high = lambda r: r*(512/5.*r**2-1536/35.*r**4+512/105.*r**6+512/1155.*r**8)

        fb1g3_mid_low  = Zb1g3(exp(-mid_low_s))
        fb1g3_mid_high = Zb1g3(exp(-mid_high_s))
        fb1g3_high     = Zb1g3_high(exp(-high_s))
        fb1g3_low      = Zb1g3_low(exp(-low_s))

        fb1g3 = hstack((fb1g3_low,fb1g3_mid_low,64,fb1g3_mid_high,fb1g3_high))
        gb1g3 = fftconvolve(P,fb1g3)*dL
        gb1g3_k = gb1g3[N-1:2*N-1]
        Pb1g3_bar = -1./42.*k**3/(2*pi)**2*P*gb1g3_k

        return Pb1g3_bar

#-------------------------------------------------------------------------------

    def P_tt_13_reg(self,k, P):
        """
        Computes the contribution P_Z1mu2f for the galaxy power spectrum

        Parameters
        ----------
        K: array of floats, Fourier wavenumbers, log-spaced
        P: array of floats, power spectrum

        Returns
        -------
        P13tt_bar: array, P_Z1mu2f contribution to the galaxy power spectrum
        """

        N = k.size
        n = arange(-N+1,N)
        dL = log(k[1])-log(k[0])
        s = n*dL
        cut = 7
        high_s = s[s>cut]
        low_s  = s[s<-cut]
        mid_high_s = s[(s<=cut) & (s>0)]
        mid_low_s = s[(s>=-cut) & (s<0)]

        Z13tt = lambda r: -1.5*(-24./r**2 + 52. - 8.*r**2 + 12.*r**4 - 6.*(r**2-1.)**3*(r**2+2.)/r**3*log((r+1.)/absolute(r-1.))) * r
        Z13tt_low = lambda r: r*( -672./5. + 3744./35./r**2 - 608./35./r**4 - 160./77./r**6 - 2976./5005./r**8)
        Z13tt_high = lambda r: r*(-96./5.*r**2 - 288./7.*r**4 + 352./35.*r**6 + 544./385.*r**8)

        f13tt_mid_low = Z13tt(exp(-mid_low_s))
        f13tt_mid_high= Z13tt(exp(-mid_high_s))
        f13tt_high    = Z13tt_high(exp(-high_s))
        f13tt_low     = Z13tt_low(exp(-low_s))

        f13tt = hstack((f13tt_low,f13tt_mid_low,-48,f13tt_mid_high,f13tt_high))
        g13tt = fftconvolve(P,f13tt)*dL
        g13tt_k = g13tt[N-1:2*N-1]
        P13tt_bar = 1./252.*k**3/(2*pi)**2*P*g13tt_k

        return P13tt_bar

#-------------------------------------------------------------------------------

    def P_mu2fb1_13(self,k, P):
        """
        Computes the contribution P_Z1mu2fb1 for the galaxy power spectrum

        Parameters
        ----------
        K: array of floats, Fourier wavenumbers, log-spaced
        P: array of floats, power spectrum

        Returns
        -------
        P13_bar: array, P_Z1mu2fb1 contribution to the galaxy power spectrum
        """
        N = k.size
        n = arange(-N+1,N)
        dL = log(k[1])-log(k[0])
        s = n*dL
        cut = 7
        high_s = s[s>cut]
        low_s  = s[s<-cut]
        mid_high_s = s[(s<=cut) & (s>0)]
        mid_low_s = s[(s>=-cut) & (s<0)]

        Z13 = lambda r: ( 36. + 96.*r**2 - 36.*r**4 + 18.*(r**2-1.)**3/r*log((r+1.)/absolute(r-1.)) )*r
        Z13_low = lambda r: r*( 576./5. - 576./35./r**2 - 64./35./r**4 - 192./385./r**6 - 192./1001./r**8)
        Z13_high = lambda r: r*(192.*r**2 - 576./5.*r**4 + 576./35.*r**6 + 64./35.*r**8)

        f13_mid_low = Z13(exp(-mid_low_s))
        f13_mid_high= Z13(exp(-mid_high_s))
        f13_high    = Z13_high(exp(-high_s))
        f13_low     = Z13_low(exp(-low_s))

        f13 = hstack((f13_low,f13_mid_low,96,f13_mid_high,f13_high))
        g13 = fftconvolve(P,f13)*dL
        g13_k = g13[N-1:2*N-1]
        P13_bar = 1./84.*k**3/(2*pi)**2*P*g13_k

        return P13_bar

#-------------------------------------------------------------------------------

    def P_mu4f2_13(self, k, P):
        """
        Computes the contribution P_Z1mu4f2 for the galaxy power spectrum

        Parameters
        ----------
        K: array of floats, Fourier wavenumbers, log-spaced
        P: array of floats, power spectrum

        Returns
        -------
        P13_bar: array, P_Z1mu4f2 contribution to the galaxy power spectrum
        """
        N = k.size
        n = arange(-N+1,N)
        dL = log(k[1])-log(k[0])
        s = n*dL
        cut = 7
        high_s = s[s>cut]
        low_s  = s[s<-cut]
        mid_high_s = s[(s<=cut) & (s>0)]
        mid_low_s = s[(s>=-cut) & (s<0)]

        Z13 = lambda r: ( 36./r**2 + 12. + 252.*r**2 - 108.*r**4 + 18.*(r**2-1.)**3*(1.+3.*r**2)/r**3 * log((r+1.)/absolute(r-1.)))*r
        Z13_low = lambda r: r*( 768./5. + 2304./35./r**2 - 768./35./r**4 - 256./77./r**6 - 768./715./r**8)
        Z13_high = lambda r: r*(2304./5.*r**2 - 2304./7.*r**4 + 256./5.*r**6 + 2304./385.*r**8)

        f13_mid_low = Z13(exp(-mid_low_s))
        f13_mid_high= Z13(exp(-mid_high_s))
        f13_high    = Z13_high(exp(-high_s))
        f13_low     = Z13_low(exp(-low_s))

        f13 = hstack((f13_low,f13_mid_low,192,f13_mid_high,f13_high))
        g13 = fftconvolve(P,f13)*dL
        g13_k = g13[N-1:2*N-1]
        P13_bar = 1./336.*k**3/(2*pi)**2*P*g13_k

        return P13_bar

    def myRSD_components(self, P, f, b1, P_window=None, C_window=None):

        _, A = self.J_k_tensor(P, self.X_RSDA, P_window=P_window, C_window=C_window)

        A1 = b1**2 * np.dot(self.A_coeff[:, 0], A) + b1 * f * np.dot(self.A_coeff[:, 1], A) + f ** 2 * np.dot(self.A_coeff[:, 2], A)
        A3 = b1**2 * np.dot(self.A_coeff[:, 3], A) + b1 * f * np.dot(self.A_coeff[:, 4], A) + f ** 2 * np.dot(self.A_coeff[:, 5], A)
        A5 = b1**2 * np.dot(self.A_coeff[:, 6], A) + b1 * f * np.dot(self.A_coeff[:, 7], A) + f ** 2 * np.dot(self.A_coeff[:, 8], A)

        _, B = self.J_k_tensor(P, self.X_RSDB, P_window=P_window, C_window=C_window)

        B0 = b1**2 * np.dot(self.B_coeff[:, 0], B) + b1 * f * np.dot(self.B_coeff[:, 1], B) + f ** 2 * np.dot(self.B_coeff[:, 2], B)
        B2 = b1**2 * np.dot(self.B_coeff[:, 3], B) + b1 * f * np.dot(self.B_coeff[:, 4], B) + f ** 2 * np.dot(self.B_coeff[:, 5], B)
        B4 = b1**2 * np.dot(self.B_coeff[:, 6], B) + b1 * f * np.dot(self.B_coeff[:, 7], B) + f ** 2 * np.dot(self.B_coeff[:, 8], B)
        B6 = b1**2 * np.dot(self.B_coeff[:, 9], B) + b1 * f * np.dot(self.B_coeff[:, 10], B) + f ** 2 * np.dot(self.B_coeff[:, 11], B)

        # check if b1**2 * np.dot(self.B_coeff[:, 9], B) + b1 * f * np.dot(self.B_coeff[:, 10], B)  are zero

        if (self.extrap):
            _, A1 = self.EK.PK_original(A1)
            _, A3 = self.EK.PK_original(A3)
            _, A5 = self.EK.PK_original(A5)
            _, B0 = self.EK.PK_original(B0)
            _, B2 = self.EK.PK_original(B2)
            _, B4 = self.EK.PK_original(B4)
            _, B6 = self.EK.PK_original(B6)

        # ??????
        P_Ap1 = RSD_ItypeII.P_Ap1(self.k_original, P, f)
        P_Ap3 = RSD_ItypeII.P_Ap3(self.k_original, P, f)
        P_Ap5 = RSD_ItypeII.P_Ap5(self.k_original, P, f)

        return A1, A3, A5, B0, B2, B4, B6, P_Ap1, P_Ap3, P_Ap5
