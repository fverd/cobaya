import camb
import copy
import math
from math import e, pi
import numpy as np
from numpy import arange, array, concatenate, einsum, exp, hstack, linspace, log, log10, logspace, newaxis, ones, real, sqrt, sum, vstack, where, zeros
from scipy.constants import speed_of_light
from scipy.fftpack import dst, idst
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d, RectBivariateSpline, splev, splrep, UnivariateSpline
from scipy.integrate import simpson as simps
from scipy.integrate import quad
from scipy.ndimage import gaussian_filter1d
from scipy.special import hyp2f1, legendre, spherical_jn
from scipy import misc
import time
from .tools import bispectrum_functions as bispfunc
from .tools import cosmology

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

try: import baccoemu
except: print("\033[1;31m[WARNING]: \033[00m"+"Bacco not installed!")

try: import fastpt.FASTPT_simple as fpt
except:	import FASTPT_simple as fpt

try:    import fastpt.FASTPT as tns
except:    import FASTPT as tns

lightspeed_kms = speed_of_light/1000

class PBJtheory:
    """PBJtheory

    Class for the comological functions and theoretical predictions
    """
    def linear_power_spectrum(self, linear='camb', npoints=1000, kmin=1.0001e-4,
                              kmax=198., redshift=0, cold=True,
                              cosmo=None):
        """
        Linear power spectrum.

        Computes the linear power spectrum using either CAMB or the BACCO emulator.

        Parameters
        ----------
        linear : str, optional
            CAMB or BACCO.
        npoints : int, optional
            Number of points to sample the power spectrum.
        kmin : float, optional
            Minimum k value for the linear power spectrum.
        kmax : float, optional
            Maximum k value for the linear power spectrum.
        redshift : float, optional
            Redshift of the power spectrum.
        cold : bool, optional
            Compute the cold matter power spectrum.
        cosmo : dict, optional
            Cosmological parameters.

        Returns
        -------
        k : (npoints,) array
            Wavenumber array.
        P : (npoints,) array
            Linear power spectrum.
        """
        if linear == 'camb':
            return self.call_camb(npoints, kmin, kmax, redshift, cosmo=cosmo,
                                  cold=cold)
        elif linear == 'bacco':
            return self.call_bacco(npoints, kmin, kmax, redshift, cosmo=cosmo,
                                   cold=cold)
        elif linear == 'cobaya':
            return self.provide_cobaya_PL(npoints, kmin, kmax, redshift, cosmo=cosmo)
        
    def provide_cobaya_PL(self, npoints, kmin, kmax, redshift, cosmo=None):
        kL = logspace(log10(kmin), log10(kmax), npoints)
        if cosmo is not None:
            h    = cosmo['h'] if 'h' in cosmo else self.h
        else:
            raise ValueError('You should provide h for units conversion')
        # Requesto only up to k=5, so split the k vector
        kcut = kL[where(kL <= 5)]
        kext = kL[where(kL > 5)]
        #Occhio la funzione Pk_interpolator è bastarda, le unita sono 1/Mpc e non h/Mpc
        PL = self.cobaya_provider_Pk_interpolator.P(redshift, kcut*h) # The cobaya Pk interpolator is very similar to camb one
        PL *= h**3

        # Extrapolation with power law
        m = math.log(PL[-1] / PL[-2]) / math.log(kcut[-1] / kcut[-2])
        PL_ext = PL[-1] / kcut[-1]**m * kext**m
        return kL, hstack((PL, PL_ext))
    
# -------------------------------------------------------------------------------

    def call_bacco(self, npoints, kmin, kmax, redshift, cold=True, cosmo=None):
        """
        Call to the BACCO linear emulator given a set cosmological
        parameters.  By default, the P(k) is computed in 1000 points
        from kmin=1e-4 to kmax=198.  To account for the k-cut of BACCO
        (k_max = 50 h/Mpc), a linear extrapolation is implemented for
        the log(P(k)).

        Parameters
        ----------
        npoints : int, optional
            Number of points to sample the power spectrum. Default is 1000.
        kmin : float, optional
            Minimum k value for the linear power spectrum. Default is 1e-4.
        kmax : float, optional
            Maximum k value for the linear power spectrum. Default is 198.
        cold : bool, optional
            Compute the cold matter power spectrum. Default is False.
        cosmo : dict, optional
            Cosmological parameters. Default is None, parameters are fixed and
            read from the paramfile.

        Returns
        -------
        kL : (npoints,) array
            Wavenumber array.
        PL : (npoints,) array
            Linear power spectrum.
        """
        if cosmo is not None:
            ns   = cosmo['ns'] if 'ns' in cosmo else self.ns
            As   = cosmo['As'] if 'As' in cosmo else self.As
            h    = cosmo['h'] if 'h' in cosmo else self.h
            Obh2 = cosmo['Obh2'] if 'Obh2' in cosmo else self.Obh2
            Och2 = cosmo['Och2'] if 'Och2' in cosmo else self.Och2
            Mnu  = cosmo['Mnu'] if 'Mnu' in cosmo else self.Mnu
            tau  = cosmo['tau'] if 'tau' in cosmo else 0.0952
        else:
            ns   = self.ns
            As   = self.As
            h    = self.h
            Obh2 = self.Obh2
            Och2 = self.Och2
            Mnu  = self.Mnu
            tau  = self.tau

        params = {
            'ns'            : ns,
            'A_s'           : As,
            'tau'           : tau,
            'hubble'        : h,
            'omega_baryon'  : Obh2/h/h,
            'omega_cold'    : (Och2 + Obh2)/h/h, # This is Omega_cb!!!
            'neutrino_mass' : Mnu,
            'w0'            : -1,
            'wa'            : 0,
            'expfactor'     : 1/(1+redshift)
        }

        kL = logspace(log10(kmin), log10(kmax), npoints)

        # The emulator only works up to k=50, so split the k vector
        kcut = kL[where(kL <= 50)]
        kext = kL[where(kL > 50)]
        _, PL = self.emulator.get_linear_pk(k=kcut, cold=cold, **params)

        # Extrapolation with power law
        m = math.log(PL[-1] / PL[-2]) / math.log(kcut[-1] / kcut[-2])
        PL_ext = PL[-1] / kcut[-1]**m * kext**m

        return kL, hstack((PL, PL_ext))

#-------------------------------------------------------------------------------

    def call_camb(self, npoints, kmin, kmax, redshift, cold=True, cosmo=None):
        """
        Call to CAMB given the cosmological parameters, to compute
        the linear power spectrum.

        Parameters
        ----------
        npoints : int, optional
            Number of points in the k-grid. Defaults to 1000.
        kmin : float, optional
            Minimal k value in the k-grid. Defaults to 1e-4.
        kmax : float, optional
            Maximal k value in the k-grid. Defaults to 198.
        cold : bool, optional
            If True, compute the cold dark matter power spectrum.
            Defaults to True.
        cosmo : dict, optional
            Dictionary with cosmological parameters. If None, parameters are
            fixed and read from the paramfile. Defaults to None.

        Returns
        -------
        kL : numpy array
            k-grid.
        PL : numpy array
            Linear power spectrum.
        """
        if cosmo is not None:
            ns   = cosmo['ns'] if 'ns' in cosmo else self.ns
            As   = cosmo['As'] if 'As' in cosmo else self.As
            h    = cosmo['h'] if 'h' in cosmo else self.h
            Obh2 = cosmo['Obh2'] if 'Obh2' in cosmo else self.Obh2
            Och2 = cosmo['Och2'] if 'Och2' in cosmo else self.Och2
            tau  = cosmo['tau'] if 'tau' in cosmo else 0.0952
            Mnu  = cosmo['Mnu'] if 'Mnu' in cosmo else self.Mnu
            Ok   = cosmo['Ok'] if 'Ok' in cosmo else 0.
        else:
            ns   = self.ns
            As   = self.As
            h    = self.h
            Obh2 = self.Obh2
            Och2 = self.Och2
            Ok   = self.Ok
            tau  = self.tau
            Mnu  = self.Mnu

        nmassive = 1 if Mnu != 0 else 0
        params = camb.model.CAMBparams()
        params.InitPower.set_params(ns=ns, As=As, r=0.)
        params.set_cosmology(H0 = 100.*h,
                             ombh2 = Obh2,
                             omch2 = Och2,
                             mnu   = Mnu,
                             nnu   = 3.046,
                             num_massive_neutrinos = nmassive,
                             neutrino_hierarchy = 'degenerate',
                             omk   = Ok,
                             tau   = tau)
        params.set_dark_energy(w=-1, wa=0, dark_energy_model='ppf')
        params.set_matter_power(redshifts = [redshift], kmax = kmax)
        results = camb.get_results(params)

        if cold:
            var=8
        else:
            var=7

        kcamb, zcamb, pkcamb = results.get_matter_power_spectrum(
            minkh = kmin, maxkh = kmax, npoints = npoints,
            var1 = var, var2 = var)

        pk_itp = camb.get_matter_power_interpolator(
            params, nonlinear=False, hubble_units=True, k_hunit=True,
            kmax=kmax, zmax=20.0, var1=var, var2=var)

        setattr(self, 'camb_interpolator', pk_itp.P)

        kL, PL = kcamb, pkcamb.ravel()
        return kL, PL

#-------------------------------------------------------------------------------

    def CallEH_NW(self, cosmo=None):
        """
        Computes the smooth matter power spectrum following the
        prescription of Eisenstein & Hu 1998.

        Parameters
        ----------
        cosmo : dict, optional
            Dictionary with cosmological parameters.
            Default: None, parameters are fixed and read from the paramfile

        Returns
        -------
        kL : array_like
            k-grid
        P_EH : array_like
            Smooth linear power spectrum
        """
        if cosmo is not None:
            ns   = cosmo['ns'] if 'ns' in cosmo else self.ns
            As   = cosmo['As'] if 'As' in cosmo else self.As
            h    = cosmo['h'] if 'h' in cosmo else self.h
            Obh2 = cosmo['Obh2'] if 'Obh2' in cosmo else self.Obh2
            Och2 = cosmo['Och2'] if 'Och2' in cosmo else self.Och2
            Mnu  = cosmo['Mnu'] if 'Mnu' in cosmo else self.Mnu
            tau = cosmo['tau'] if 'tau' in cosmo else 0.0952
            Tcmb = cosmo['Tcmb'] if 'Tcmb' in cosmo else self.Tcmb
            PL   = cosmo['PL']
        else:
            ns   = self.ns
            As   = self.As
            h    = self.h
            Obh2 = self.Obh2
            Och2 = self.Och2
            Mnu = self.Mnu
            tau  = self.tau
            Tcmb = self.Tcmb
            PL   = self.PL

        Omh2 = Obh2 + Och2 + Mnu/93.14
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

    def pksmooth_dst(self, plinear, redshift, kmin, cosmo=None, cold=True):
        """
        Returns the dewiggled power spectrum for a given cosmology and redshift.
        The input plinear must be evaluated on the same grid as fastpt (self.kL)

        The de-wiggling is performed by identifying and removing the BAO bump
        in real space by means of a type-II dst transform, then transforming
        back to Fourier space.

        See arXiv:1712.08067, arXiv:1906.02742. Adapted from the baccoemu
        implementation.
        """
        if cosmo is not None:
            ns   = cosmo['ns'] if 'ns' in cosmo else self.ns
            As   = cosmo['As'] if 'As' in cosmo else self.As
            h    = cosmo['h'] if 'h' in cosmo else self.As
            Obh2 = cosmo['Obh2'] if 'Obh2' in cosmo else self.Obh2
            Och2 = cosmo['Och2'] if 'Och2' in cosmo else self.Och2
            Mnu  = cosmo['Mnu'] if 'Mnu' in cosmo else self.Mnu
            w0   = cosmo['w0'] if 'w0' in cosmo else self.w0
            wa   = cosmo['wa'] if 'wa' in cosmo else self.wa
        else:
            ns   = self.ns
            As   = self.As
            h    = self.h
            Obh2 = self.Obh2
            Och2 = self.Och2
            Mnu  = self.Mnu
            w0   = self.w0
            wa   = self.wa

        params = {
            'ns'            : ns,
            'A_s'           : As,
            'hubble'        : h,
            'omega_baryon'  : Obh2/h/h,
            'omega_cold'    : (Och2 + Obh2)/h/h, # This is Omega_cb!!!
            'neutrino_mass' : Mnu,
            'w0'            : w0,
            'wa'            : wa,
            'expfactor'     : 1/(1+redshift)
        }

        # Sample k, P(k) in 2^15 points
        nk = int(2**15)
        kmax = 10
        klin = linspace(kmin, kmax, nk)
        if self.linear == 'bacco':
            _, pklin = self.emulator.get_linear_pk(k=klin, cold=cold, **params)
        elif self.linear == 'camb':
            pklin = self.camb_interpolator(redshift, klin)

        # DST-II log(k *P(k))
        logkpk = log10(klin * pklin)
        dstpk = dst(logkpk, type=2)

        # Split in even and odd indices,
        even = dstpk[0::2] # array with even indexes
        odd  = dstpk[1::2] # array with odd indexes
        i_even = arange(len(even)).astype(int)
        i_odd  = arange(len(odd)).astype(int)
        even_cs = splrep(i_even, even, s=0)
        odd_cs  = splrep(i_odd, odd, s=0)

        # Compute second derivatives and interp separately with cubic splines
        even_2nd_der = splev(i_even, even_cs, der=2, ext=0)
        odd_2nd_der = splev(i_odd, odd_cs, der=2, ext=0)

        # Find i_min and i_max for the even and odd arrays
        # These are optimised for the considered krange [1e-4,10]
        imin_even = i_even[100:300][np.argmax(even_2nd_der[100:300])] - 30
        imax_even = i_even[100:300][np.argmin(even_2nd_der[100:300])] + 70
        imin_odd = i_odd[100:300][np.argmax(odd_2nd_der[100:300])] - 30
        imax_odd = i_odd[100:300][np.argmin(odd_2nd_der[100:300])] + 75

        # Cut out the BAOs
        # mask indices
        i_even_holed = np.concatenate((i_even[:imin_even], i_even[imax_even:]))
        i_odd_holed = np.concatenate((i_odd[:imin_odd], i_odd[imax_odd:]))
        # mask log(k*P(k))
        even_holed = np.concatenate((even[:imin_even], even[imax_even:]))
        odd_holed = np.concatenate((odd[:imin_odd], odd[imax_odd:]))
        # interp the arrays rescaled by a factor (i + 1)^2 using cubic splines
        even_holed_cs = splrep(i_even_holed, even_holed*(i_even_holed+1)**2, s=0)
        odd_holed_cs = splrep(i_odd_holed, odd_holed * (i_odd_holed+1)**2, s=0)

        # Merge the two arrays without the bumps, and without the rescaling
        # factor of (i + 1)^2, and inversely FST
        even_smooth = splev(i_even, even_holed_cs, der=0, ext=0) / (i_even+1)**2
        odd_smooth = splev(i_odd, odd_holed_cs, der=0, ext=0) / (i_odd + 1)**2

        dstkpk_smooth = np.zeros(nk)
        dstkpk_smooth[0::2] = even_smooth
        dstkpk_smooth[1::2] = odd_smooth

        pksmooth = idst(dstkpk_smooth, type=2) / (2 * len(dstkpk_smooth))
        pksmooth = 10**(pksmooth) / klin

        high_k = self.kL[self.kL > 2]
        high_p = plinear[self.kL > 2]
        k_ext = concatenate((klin[klin<=2], high_k))
        p_ext = concatenate((pksmooth[klin<=2], high_p))

        pksmooth_tot = splrep(np.log(k_ext), np.log(p_ext), s=0)
        smooth_itp = np.exp(splev(np.log(self.kL), pksmooth_tot, der=0, ext=0))

        return self.kL, smooth_itp

#-------------------------------------------------------------------------------

    def _gaussianFiltering(self, lamb, cosmo=None):
        """
        Smooth the linear power spectrum using a Gaussian filter

        Parameters
        ----------
        lamb : float
            Width of the Gaussian filter in log10(k)
        cosmo : dict, optional
            Dictionary with cosmological parameters.
            Default: None, parameters are read from the paramfile

        Returns
        -------
        Psmooth : array of floats
            Smoothed power spectrum
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

        PEH = extrapolateFx(self.kL, PEH)
        Psmooth = gaussian_filter1d(PLinear/PEH, lamb/dqlog)*PEH

        return Psmooth[kNumber//2:kNumber//2+kNumber]

#-------------------------------------------------------------------------------

    def IRresum(self, redshift, kind, kmin=1.0001e-4, lamb=0.25, kS=0.2,
                lOsc=102.707, cosmo=None, cold=True):
        """
        Splitting of the linear power spectrum into a smooth and a
        wiggly part, and computation of the damping factors.

        Parameters
        ----------
        redshift : float
            Redshift at which to compute the power spectrum
        kind : str
            Kind of IR resummation ("IR" or "IR-PT")
        kmin : float, optional
            Minimum k value for the linear power spectrum. Default is 1e-4
        lamb : float, optional
            Width of Gaussian filter. Default 0.25
        kS : float, optional
            Scale of separation between large and small scales. Default 0.2
        lOsc : float, optional
            Scale of the BAO. Default 102.707
        cosmo : dict, optional
            Dictionary with cosmological parameters. Default: None, parameters
            are fixed and read from the paramfile
        cold : bool, optional
            Compute the cold matter power spectrum. Default is True

        Returns
        -------
        kL : array of floats
            k-grid
        Pnw : array of floats
            No-wiggle power spectrum
        Pw : array of floats
            Wiggle power spectrum
        Sigma2 : float
        dSigma2 : float
        """
        if cosmo is not None:
            PL = cosmo['PL']
        else:
            PL = self.PL

        # Sigma2 as integral up to 0.2;
        # Uses Simpson integration (no extreme accuracy needed)
        icut = (self.kL <= kS)
        kLcut = self.kL[icut]

        if kind == 'EH':
            Pnw = self._gaussianFiltering(lamb, cosmo=cosmo)
        elif kind == 'DST':
            _, Pnw = self.pksmooth_dst(PL,redshift,kmin,cosmo=cosmo,cold=cold)
        Pw = PL - Pnw

        Pnwcut = Pnw[icut]
        norm = 1./(6.*pi**2)
        Sigma2  = norm*simps(Pnwcut*(1.-spherical_jn(0,kLcut*lOsc)+
                                     2.*spherical_jn(2,kLcut*lOsc)), x=kLcut)
        dSigma2 = norm*simps(3.*Pnwcut*spherical_jn(2,kLcut*lOsc), x=kLcut)

        return self.kL, Pnw, Pw, Sigma2, dSigma2

#-------------------------------------------------------------------------------

    def _muxdamp(self, k, mu, Sigma2, dSigma2, f):
        """
        Computes the full damping factor for IR-resummation in redshift space
        and the (mu^n * exponential damping).

        Parameters
        ----------
        k : float or array of floats, k-grid
        mu : float or array of floats, cosine of the angle between k and l.o.s.
        Sigma2 : float, value of \\Sigma^2
        dSigma2 : float, value of \\delta \\Sigma^2
        f : float, growth rate

        Returns
        -------
        Sig2mu : float or array of floats, \\Sigma^2(\\mu)
        RSDdamp : float or array of floats (can be 2D),
                  \\exp(-k^2*\\Sigma^2(\\mu))
        """
        Sig2mu = Sigma2 + f * mu**2 * (2. * Sigma2 + f *
                                       (Sigma2 + dSigma2 * (mu**2 - 1)))
        RSDdamp = exp(-k**2 * Sig2mu)
        return Sig2mu, RSDdamp

#-------------------------------------------------------------------------------

    def growth_factor(self, redshift, cosmo=None):
        """
        Computes the growth factor D for the redshift specified in
        the cosmo dictionary using the hypergeometric function.
        Assumes flat LCDM.

        Parameters
        ----------
        cosmo : dict, optional
            Must include Omh2, h, z. Default None, uses the input
            cosmology.

        Returns
        -------
        Dz/D0 : float
            Growth factor normalised to z=0.
        D0 : float
            Growth factor at z=0.
        """
        if cosmo is not None:
            Och2 = cosmo['Och2'] if 'Och2' in cosmo else self.Och2
            Obh2 = cosmo['Obh2'] if 'Obh2' in cosmo else self.Obh2
            h = cosmo['h'] if 'h' in cosmo else self.h
        else:
            Och2 = self.Och2
            Obh2 = self.Obh2
            Mnu = self.Mnu
            h = self.h

        Ocb = (Och2 + Obh2)/h/h
        a = 1 / (1 + redshift)
        Dz = a * hyp2f1(1./3, 1., 11./6, (Ocb - 1.) / Ocb * a**3)
        D0 = hyp2f1(1./3, 1., 11./6, (Ocb - 1.) / Ocb)
        return  Dz / D0, D0

#-------------------------------------------------------------------------------

    def growth_rate(self, redshift, cosmo=None):
        """
        Computes the growth rate f for the given redshift using
        the hypergeometric function. Assumes a flat LCDM model.

        Parameters
        ----------
        redshift : float
            The redshift at which to compute the growth rate.
        cosmo : dict, optional
            Cosmological parameters, must include 'Och2', 'Obh2',
            'Mnu', and 'h'. Defaults to the object's attributes if None.

        Returns
        -------
        f : float
            The computed growth rate.
        """
        if cosmo is not None:
            Och2 = cosmo['Och2'] if 'Och2' in cosmo else self.Och2
            Obh2 = cosmo['Obh2'] if 'Obh2' in cosmo else self.Obh2
            Mnu = cosmo['Mnu'] if 'Mnu' in cosmo else self.Mnu
            h = cosmo['h'] if 'h' in cosmo else self.h
        else:
            Och2 = self.Och2
            Obh2 = self.Obh2
            Mnu = self.Mnu
            h = self.h

        Ocb = (Och2 + Obh2)/h/h
        a = 1 / (1 + redshift)
         # Un-normalised growth factor:
        D, D0 = self.growth_factor(redshift, cosmo=cosmo)
        dhyp = hyp2f1(4./3, 2., 17./6, (Ocb - 1.) / Ocb * a**3)
        return 1. + (6./11) * a**4 / (D*D0) * (Ocb - 1.) / Ocb * dhyp
        
#-------------------------------------------------------------------------------

    def _get_growth_functions(self, redshift, f=None, D=None, cosmo=None):
        """
        Private method that returns the growth functions f, D.

        If `cosmo=None`, uses values from the input cosomlogy.
        Otherwise, if `f` and / or `D` are not specified, computes
        the growth functions from the `growth_rate` and `growth_factor`
        methods.

        Parameters
        ----------
        f : float
            Input growth rate. Default `None`
        D : float
            Input growth factor. Default `None`
        cosmo : dictionary
            Input cosmological dictionary. Default `None`

        Returns
        -------
        f, D: float
            Growth functions.
        """
        if cosmo is not None:
            if f is None:
                f = self.growth_rate(redshift, cosmo=cosmo)
            if D is None:
                D, _ = self.growth_factor(redshift, cosmo=cosmo)
        else:
            if f is None:
                f = self.f
            if D is None:
                D = self.Dz
        return f, D

#-------------------------------------------------------------------------------

    def _get_growth_functions_gamma(self, redshift, gamma=0.545, cosmo=None):
        if cosmo is not None:
            Omh2 = cosmo['Omh2'] if 'Omh2' in cosmo else self.Omh2
            h = cosmo['h'] if 'h' in cosmo else self.h
            Om = Omh2/h**2
        else:
            Om = self.Om

        integrand =  lambda zz: -(Om/(Om+(1.-Om)*(1.+zz)**(-3)))**gamma / (1+zz)

        cosmo_ini = {}
        if cosmo is not None:
            cosmo_ini = cosmo
        else:
            cosmo_ini['Omh2'] = self.Omh2
            cosmo_ini['h'] = self.h
        redshift_ini = 1100

        D_lcdm_ini, _ = self.growth_factor(redshift_ini, cosmo=cosmo_ini)

        D0 = D_lcdm_ini / (exp(quad(integrand, 0, redshift_ini)[0]))
        D = D0 * exp(quad(integrand, 0, redshift)[0])
        f = -integrand(redshift) * (1.+redshift)
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
            Obh2 = cosmo['Obh2'] if 'Obh2' in cosmo else self.Obh2
            Och2 = cosmo['Och2'] if 'Och2' in cosmo else self.Och2
            Mnu = cosmo['Mnu'] if 'Mnu' in cosmo else self.Mnu
            h = cosmo['h'] if 'h' in cosmo else self.h
            Om = (Obh2 + Och2 + Mnu/93.14)/h**2
            w0 = cosmo['w0'] if 'w0' in cosmo else self.w0
            Ok = cosmo['Ok'] if 'Ok' in cosmo else self.Ok
        else:
            Om = (self.Obh2 + self.Och2 + self.Mnu/93.14)/(self.h**2)
            w0 = self.w0
            Ok = self.Ok
        return sqrt(Om*(1+z)**3 + (1-Om-Ok)*(1+z)**(-3.*(1.+w0)) + Ok*(1+z)**2)

#-------------------------------------------------------------------------------

    def angular_diam_distance(self, z, cosmo=None):
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
        function = lambda x: 1/self.Hubble_adim(x, cosmo=cosmo)
        res, _  = quad(function, 0, z)
        return res

#-------------------------------------------------------------------------------

    def _set_fiducials_for_AP(self, redshift, cosmo):
        """
        Sest fiducial quantities for Alcock-Paczynski distortions.

        Parameters
        ----------
        redshift: float
            The redshift at which to compute the fiducials.
        cosmo: dict
            The cosmology to use for the computation. If None, the fiducials
            are set to the default cosmology of the class.
        """
        if cosmo is not None:
            Obh2 = cosmo['Obh2'] if 'Obh2' in cosmo else self.Obh2
            Och2 = cosmo['Och2'] if 'Och2' in cosmo else self.Och2
            Mnu = cosmo['Mnu'] if 'Mnu' in cosmo else self.Mnu
            h = cosmo['h'] if 'h' in cosmo else self.h
            Om = (Obh2 + Och2 + Mnu/93.14)/h**2
            w0 = cosmo['w0'] if 'w0' in cosmo else self.w0
            wa = cosmo['wa'] if 'wa' in cosmo else self.wa
            Ok = cosmo['Ok'] if 'Ok' in cosmo else self.Ok
        else:
            Om = (self.Obh2 + self.Och2 + self.Mnu/93.14)/(self.h**2)
            w0 = self.w0
            wa = self.wa
            Ok = self.Ok

        self.Hubble_adim_fid = cosmology.Hubble_adim(redshift, Om, w0, wa)
        if isinstance(redshift, (float, int)):
            self.angular_diam_distance_fid = \
                cosmology.angular_diam_distance(redshift, Om, w0, wa)
        else:
            self.angular_diam_distance_fid = \
                np.asarray([cosmology.angular_diam_distance(iz, Om, w0, wa)
                for iz in redshift])

#-------------------------------------------------------------------------------

    def AP_factors(self, mugrid, z, cosmo, AP_as_nuisance, alpha_par, alpha_perp):
        """
        Computes quantities to apply Alcock-Paczynski distortions.

        Parameters
        ----------
        mugrid: array
            Cosine of the angle between k and the line of sight
        z: float
            Redshift of the observed galaxy sample
        cosmo: dictionary
            Containing values for `'Obh2'`, `'Och2'`, `h` and `'z'`
        AP_as_nuisance: bool
            If True uses specified values for the alphas instead of computing
            them from the cosmology
        alpha_par, alpha_perp: float
            Values to use for the alphas. Only used if `AP_as_nuisance==True`

        Returns
        -------
        AP_factor: array, sqrt(mu^2/alpha_par^2 + (1-mu^2)/alpha_perp^2)
        alpha_par: float, E^{fiducial}(z)/E(z)
        alpha_perp: float, D_A(z)/D_A^F(z)
        AP_amplitude: float, 1 / (alpha_{par} alpha_{perp}^2)
        """
        if cosmo is not None:
            Obh2 = cosmo['Obh2'] if 'Obh2' in cosmo else self.Obh2
            Och2 = cosmo['Och2'] if 'Och2' in cosmo else self.Och2
            Mnu = cosmo['Mnu'] if 'Mnu' in cosmo else self.Mnu
            h = cosmo['h'] if 'h' in cosmo else self.h
            Om = (Obh2 + Och2 + Mnu/93.14)/h**2
            w0 = cosmo['w0'] if 'w0' in cosmo else self.w0
            wa = cosmo['wa'] if 'wa' in cosmo else self.wa
        else:
            Om = (self.Obh2 + self.Och2 + self.Mnu/93.14)/(self.h**2)
            w0 = self.w0
            wa = self.wa

        if AP_as_nuisance == False:
            alpha_par =  self.Hubble_adim_fid / \
                cosmology.Hubble_adim(z, Om, w0, wa)
            alpha_perp = cosmology.angular_diam_distance(z, Om, w0, wa) / \
                self.angular_diam_distance_fid

        AP_amplitude = 1 / (alpha_par * alpha_perp**2)
        AP_factor = sqrt((mugrid/alpha_par)**2 + (1. - mugrid**2)/alpha_perp**2)

        return alpha_par, alpha_perp, AP_factor, AP_amplitude

#-------------------------------------------------------------------------------

    def _apply_AP_distortions(self, k, mu, redshift, cosmo, AP_as_nuisance,
                   alpha_par, alpha_perp):
        """
        Private method that applies the AP distortions to the input k, mu
        arrays. Has an option to be used with arbitrary values of the alphas,
        can be activated by setting `AP_as_nuisance` to True and `alpha_par`,
        `alpha_perp` to the desired values.

        .. math::
        q = k \\left( \\frac{\\mu^2}{\\alpha_{\\parallel}^2} + \\frac{1-\\mu^2}{\\alpha_{\\perp}^2} \\right)^{1/2}

        .. math::
        \\nu = \\frac{\\mu}{\\alpha_{\\parallel}} \\left( \\frac{\\mu^2}{\\alpha_{\\parallel}^2} + \\frac{1-\\mu^2}{\\alpha_{\\perp}^2}} \\right)^{-1/2}

        Arguments
        ---------
        k : array
            Reference values of k
        mu : array
            Reference values of mu
        cosmo : dictionary
            Containing values for `'Obh2'`, `'Och2'`, `h` and `z`.
        AP_as_nuisance : bool
            If True uses specified values for the alphas instead of computing
            them from the cosmology
        alpha_par, alpha_perp : float
            Values to use for the alphas. Only used if `AP_as_nuisance=True`

        Returns
        -------
        q : array
            Distorted (observed) k
        nu : array
            Distorted (observed) mu
        AP_amplitude : float
            Rescaling factor for the power spectrum, 1/(alpha_par alpha_perp^2)
        """
        alpha_par, alpha_perp, AP_factor, AP_amplitude = \
            self.AP_factors(mu, redshift, cosmo, AP_as_nuisance, alpha_par, alpha_perp)

        k_sub = k[:, newaxis]
        nu = mu / (alpha_par * AP_factor)
        q  = k_sub * AP_factor
        return q, nu, AP_amplitude

#-------------------------------------------------------------------------------

    def Kaiser(self, b1, f, mu):
        return b1 + f * mu**2

#-------------------------------------------------------------------------------

    def PowerfuncIR_real(self, redshift, do_redshift_rescaling, D=None,
                         cosmo=None, kind='EH'):
        """
        Computes the leading order power spectrum, the next-to-leading
        order corrections and the EFT counterterm contribution in real
        space. If `self.IRresum=False` all contributions will not be
        IR-resummed.

        Parameters
        ----------
        cosmo : dictionary
            If `None` uses the input cosmology

        Returns
        -------
        Arrays with the various building blocks for the real space
        galaxy power spectrum in the following order:
        `PLO, PNLO, kL**2*PLO, Pb1b2, Pb1g2, Pb2b2, Pb2g2, Pg2g2, Pb1g3, kL**2`
        """
        if cosmo is not None:
            PL = cosmo['PL']
        else:
            PL = self.PL

        if do_redshift_rescaling:
            _, D =  self._get_growth_functions(redshift, f=1, D=D, cosmo=cosmo)
            knw, Pnw, Pw, Sigma2, dSigma2 = self.IRresum(0, kind, cosmo=cosmo)
        else:
            D = 1
            knw, Pnw, Pw, Sigma2, dSigma2 = self.IRresum(redshift, kind, cosmo=cosmo)

        DZ2 = D**2
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

    def _Pgg_kmu_terms(self, cosmo=None, redshift=0, kind='EH', do_AP=False,
                       window_convolution=None):
        """
        Private method that computes the terms for the loop
        corrections at redshift z=0, splits into wiggle and no-wiggle
        and stores them as attributes of the class. It also sets
        interpolators to be used when `self.do_AP == True`

        Paramters
        ---------
        cosmo :  dictionary
            If `None` uses the input cosmology
        """
        if cosmo is not None:
            PL = cosmo['PL']
        else:
            PL = self.PL

        _, self.Pnw, self.Pw, self.Sigma2, self.dSigma2 = \
            self.IRresum(redshift, kind, cosmo=cosmo)

        # Loops on P_L and Pnw
        loop22_L = self.fastpt.Pkmu_22_one_loop_terms(self.kL, PL, C_window=.75)
        loop22_nw = self.fastpt.Pkmu_22_one_loop_terms(self.kL, self.Pnw,
                                                       C_window=.75)
        loop13_L = self.fastpt.Pkmu_13_one_loop_terms(self.kL, PL)
        loop13_nw = self.fastpt.Pkmu_13_one_loop_terms(self.kL, self.Pnw)
        setattr(self, 'loop22_nw', array(loop22_nw))
        setattr(self, 'loop13_nw', array(loop13_nw))

        # Compute wiggle
        loop22_w = array([i - j for i, j in zip(loop22_L, loop22_nw)])
        loop13_w = array([i - j for i, j in zip(loop13_L, loop13_nw)])
        setattr(self, 'loop22_w', loop22_w)
        setattr(self, 'loop13_w', loop13_w)

        # Interpolators
        if (window_convolution is None) & (do_AP == False):
            setattr(self, 'Pnw_int', interp1d(self.kL, self.Pnw))
            setattr(self, 'Pw_int', interp1d(self.kL, self.Pw))

            setattr(self, 'loop22_nw_int', interp1d(self.kL, self.loop22_nw))
            setattr(self, 'loop13_nw_int', interp1d(self.kL, self.loop13_nw))
            setattr(self, 'loop22_w_int', interp1d(self.kL, self.loop22_w))
            setattr(self, 'loop13_w_int', interp1d(self.kL, self.loop13_w))
        
        else:
            setattr(self, 'Pnw_int', interp1d(self.kL, self.Pnw, fill_value='extrapolate'))
            setattr(self, 'Pw_int', interp1d(self.kL, self.Pw, fill_value='extrapolate'))

            setattr(self, 'loop22_nw_int', interp1d(self.kL, self.loop22_nw, fill_value='extrapolate'))
            setattr(self, 'loop13_nw_int', interp1d(self.kL, self.loop13_nw, fill_value='extrapolate'))
            setattr(self, 'loop22_w_int', interp1d(self.kL, self.loop22_w, fill_value='extrapolate'))
            setattr(self, 'loop13_w_int', interp1d(self.kL, self.loop13_w, fill_value='extrapolate'))

#-------------------------------------------------------------------------------

    def PowerfuncIR_RSD(self, redshift, do_redshift_rescaling, f=None, D=None,
                        cosmo=None):
        """
        Computes the redshift space leading order, the next-to-leading order
        corrections and EFT counterterms contributions of the power
        spectrum multipoles in redshift space. If `self.IRresum == False` all
        contributions will not be IR-resummed.

        Parameters
        ----------
        f : float
            Growth rate. If `None`, it's computed from the input cosmology
        D : float
            Growth factor. If `None`, it's computed from the input cosmology
        cosmo :  dictionary
            If `None` uses the input cosmology

        Returns
        -------
        P_0, P_2, P_4 : lists
            Terms to build the multipoles
        """
        if do_redshift_rescaling:
            f, D =  self._get_growth_functions(redshift, f=f, D=D, cosmo=cosmo)
        else:
            f, D =  self._get_growth_functions(redshift, f=f, D=1, cosmo=cosmo)

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
        K = 2.*array([[simps(mu**n * RSDdamp * self.kL[:,newaxis]**2. * Sig2mu *
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

    def P_ell_conv_fixed_cosmo(self, redshift, do_redshift_rescaling, f=None,
                               D=None, cosmo=None):
        """
        Computes power spectrum (multipoles) window function convolution.

        Parameters
        ----------
        redshift : float
            The requested redshift.
        do_redshift_rescaling : bool
            If True, P_L is computed at z=0 and rescaled with the linear growth.
            If False, P_L is computed at `redshift`.
        f : float, optional
            The growth rate. If None, it is computed using the input cosmology
            from the parameter file.
        D : float, optional
            The linear growth factor. If None, it is computed using the input
            cosmology from the parameter file.
        cosmo : dict, optional
            Contains the cosmology. If None, uses the input cosmology from the
            parameter file.

        Returns
        -------
        See PowerfuncIR_RSD()
        """
        P_noconv = self.PowerfuncIR_RSD(redshift, do_redshift_rescaling, f=f,
                                        D=D, cosmo=cosmo)
        P_noconv_block = np.block(
            [[interp1d(self.kL, P_noconv[l][i], kind='cubic')(self.kpW[l])
              for l in range(3)] for i in range(len(P_noconv[0]))]
        ).T
        P_conv = (self.window @ P_noconv_block).T
        return [list(P_ell_c) for P_ell_c in np.split(P_conv, 3, axis=1)]

#-------------------------------------------------------------------------------

    def P_kaiser(self, redshift, do_redshift_rescaling, kgrid=None, f=None,
                 D=None, cosmo=None, AP_as_nuisance=False, alpha_par=1,
                 alpha_perp=1, b1=1, aP=0, Psn=0):
        """
        Computes the Kaiser power spectrum in redshift space

        Parameters
        ----------
        redshift: float, redshift used to evaluate the growth functions
        do_redshift_rescaling: bool, if True P_L is computed at z=0 and rescales
                               with the linear growth, if False P_L is computed
                               at `redshift`
        kgrid: array, the output grid on which the multipoles are evaluated
        f, D: float, growth rate and growth factor. If None they are computed
              internally
        cosmo: dict, contains the cosmology. If None, uses `self.Inputcosmo`
        AP_as_nuisance: bool, if True, values for alpha_par and alpha_perp can
                        be passed, otherwise, they are computed from the
                        Fiducialcosmology and cosmo dictionary
        alpha_par, alpha_perp: AP parameters
        b1: float, bias parameter
        aP: float, shot-noise parameter
        Psn: float, Poisson shot noise

        Returns
        -------
        The power spectrum multipoles ordered as P0,P2,P4
        """
        if do_redshift_rescaling:
            f, D =  self._get_growth_functions(redshift, f=f, D=D, cosmo=cosmo)
        else:
            f, D =  self._get_growth_functions(redshift, f=f, D=1, cosmo=cosmo)

        DZ2 = D*D
        DZ4 = D**4.

        q = kgrid[:, newaxis]
        nu = self.mu
        AP_ampl = 1.

        if self.do_AP:
            q, nu, AP_ampl = self._apply_AP_distortions(
                kgrid, self.mu, redshift, cosmo, AP_as_nuisance,
                alpha_par, alpha_perp)

        # Rescale wiggle and no-wiggle P(k), Sigma and Sigma2
        Pnw_sub = self.Pnw_int(q) * DZ2
        Pw_sub = self.Pw_int(q) * DZ2
        Sigma2 = self.Sigma2 * DZ2 * self.IRres
        dSigma2 = self.dSigma2 * DZ2 * self.IRres

        Sig2mu, RSDdamp = self._muxdamp(q, nu, Sigma2, dSigma2, f)

        # Next-to-leading order, counterterm, noise
        PNLO = self.Kaiser(b1, f, nu)**2 * (Pnw_sub + RSDdamp * Pw_sub *
                                       (1. + q**2 * Sig2mu)) + Psn * (1. + aP)

        Pell = cosmology.multipole_projection(self.mu, PNLO, [0,2,4])

        return Pell*AP_ampl

#-------------------------------------------------------------------------------

    def P_kmu_z(self, redshift, do_redshift_rescaling, kgrid=None, f=None,
                D=None, cosmo=None, AP_as_nuisance=False, alpha_par=1,
                alpha_perp=1, b1=1, b2=0, bG2=0, bG3=0, c0=0, c2=0, c4=0, ck4=0,
                aP=0, e0k2=0, e2k2=0, Psn=0, sigma_z=0, f_out=0, **kwargs):
        """
        Computes the non-linear galaxy power spectrum in redshift
        space. Assumes the self._Pgg_kmu_terms function has already been called
        to set interpolators for the power spectra starting from a linear power
        spectrum. If the attribute `self.IRresum == False`, the output power
        spectrum is not IR-resummed.

        Parameters
        ----------
        redshift : float
            Redshift used to evaluate the growth functions
        do_redshift_rescaling : bool
            If True P_L is computed at z=0 and rescales with the linear growth,
            if False P_L is computed at `redshift`
        kgrid : array
            The output grid on which the multipoles are evaluated
        f, D : float
            Growth rate and growth factor. If None they are computed internally
        cosmo : dict
            Contains the cosmology. If None, uses `self.Inputcosmo`
        AP_as_nuisance : bool
            If True, values for alpha_par and alpha_perp can be passed, else,
            they are computed from the Fiducialcosmology and cosmo dictionary
        alpha_par, alpha_perp : AP parameters
        b1, b2, bG2, bG3 : float
            Bias parameters
        c0, c2, c4, ck4 : float
            EFT counterterms
        aP, e0k2, e2k2 : float
            Shot-noise parameters
        Psn : float
            Poisson shot noise

        Returns
        -------
        Pell: array
            Contains the power spectrum multipoles ordered as P0,P2,P4
        """
        if do_redshift_rescaling:
            f, D =  self._get_growth_functions(redshift, f=f, D=D, cosmo=cosmo)
        else:
            f, D =  self._get_growth_functions(redshift, f=f, D=1, cosmo=cosmo)

        DZ2 = D*D
        DZ4 = D**4.

        q = kgrid[:, newaxis]
        nu = self.mu
        AP_ampl = 1.

        sigma_r = lightspeed_kms * sigma_z / (100*\
                                              cosmology.Hubble_adim(redshift,
                                                                    self.Om,
                                                                    self.w0,
                                                                    self.wa))

        if self.do_AP:
            q, nu, AP_ampl = self._apply_AP_distortions(
                kgrid, self.mu, redshift, cosmo, AP_as_nuisance,
                alpha_par, alpha_perp)

        # Rescale wiggle and no-wiggle P(k), Sigma and Sigma2
        Pnw_sub = self.Pnw_int(q) * DZ2
        Pw_sub = self.Pw_int(q) * DZ2
        Sigma2 = self.Sigma2 * DZ2 * self.IRres
        dSigma2 = self.dSigma2 * DZ2 * self.IRres

        # Rescaling loops
        loop22_nw_sub =  self.loop22_nw_int(q) * DZ4
        loop13_nw_sub =  self.loop13_nw_int(q) * DZ4
        loop22_w =  self.loop22_w_int(q) * DZ4
        loop13_w =  self.loop13_w_int(q) * DZ4

        Sig2mu, RSDdamp = self._muxdamp(q, nu, Sigma2, dSigma2, f)

        # Next-to-leading order, counterterm, noise
        PNLO = self.Kaiser(b1, f, nu)**2 * (Pnw_sub + RSDdamp * Pw_sub *
                                       (1. + q**2 * Sig2mu))
        Pkmu_ctr = (-2. * (c0 + c2 * f * nu**2 + c4 * f * f * nu**4) *
                    q**2 + ck4 * f**4 * nu**4 * self.Kaiser(b1, f, nu)**2 *
                    q**4) * (Pnw_sub + RSDdamp * Pw_sub)
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
        bias13 = array([b1 * self.Kaiser(b1, f, nu), bG3*self.Kaiser(b1, f, nu),
                        bG2 * self.Kaiser(b1, f, nu),
                        nu**2 * f * self.Kaiser(b1, f, nu),
                        nu**2 * f * b1 * self.Kaiser(b1, f, nu),
                        (nu * f)**2 * self.Kaiser(b1, f, nu),
                        nu**4 * f**2 * self.Kaiser(b1, f, nu)])

        Pkmu_22_nw = einsum('ijl,ikl->kl', bias22, loop22_nw_sub)
        Pkmu_22_w  = einsum('ijl,ikl->kl', bias22, loop22_w)
        Pkmu_13_nw = einsum('ijl,ikl->kl', bias13, loop13_nw_sub)
        Pkmu_13_w  = einsum('ijl,ikl->kl', bias13, loop13_w)

        Pkmu_22 = Pkmu_22_nw + RSDdamp * Pkmu_22_w
        Pkmu_13 = Pkmu_13_nw + RSDdamp * Pkmu_13_w
        Pkmu    = np.exp(-(q*nu*sigma_r)**2) * (1 - f_out)**2 * \
            (PNLO + Pkmu_22 + Pkmu_13 + Pkmu_ctr) + Pkmu_noise

        Pell = cosmology.multipole_projection(self.mu, Pkmu, [0,2,4])

        return Pell*AP_ampl


#-------------------------------------------------------------------------------

    def P_ell_conv_varied_cosmo(self, redshift, do_redshift_rescaling,
                                kgrid=None, f=None, D=None, cosmo=None,
                                AP_as_nuisance=False, alpha_par=1, alpha_perp=1,
                                b1=1, b2=0, bG2=0, bG3=0, c0=0, c2=0, c4=0,
                                ck4=0, aP=0, e0k2=0, e2k2=0, Psn=0,
                                sigma_z = 0, f_out = 0, **kwargs):
        """
        Computes power spectrum (multipoles) window function convolution for
        P_kmu_z

        Parameters
        ----------
        redshift : float
            Redshift used to evaluate the growth functions
        do_redshift_rescaling : bool
            If True P_L is computed at z=0 and rescales with the linear growth,
            if False P_L is computed at `redshift`
        kgrid : array
            The output grid on which the multipoles are evaluated
        f, D : float
            Growth rate and growth factor. If None they are computed internally
        cosmo : dict
            Contains the cosmology. Default is None, uses input cosmology from
            the parameter file
        AP_as_nuisance : bool
            If True, values for alpha_par and alpha_perp can be passed, else,
            they are computed from the Fiducialcosmology and cosmo dictionary
        alpha_par, alpha_perp : AP parameters
        b1, b2, bG2, bG3 : float
            Bias parameters
        c0, c2, c4, ck4 : float
            EFT counterterms
        aP, e0k2, e2k2 : float
            Shot-noise parameters
        Psn : float
            Poisson shot noise

        Returns
        -------
        array containing the convolved power spectrum multipoles ordered
        as P0, P2, P4
        """
        P_noconv = self.P_kmu_z(redshift, do_redshift_rescaling, kgrid=kgrid,
                                f=f, D=D, cosmo=cosmo,
                                AP_as_nuisance=AP_as_nuisance,
                                alpha_par=alpha_par, alpha_perp=alpha_perp,
                                b1=b1, b2=b2, bG2=bG2, bG3=bG3, c0=c0, c2=c2,
                                c4=c4, ck4=ck4, aP=aP, e0k2=e0k2, e2k2=e2k2,
                                Psn=Psn, sigma_z=sigma_z, f_out=f_out, **kwargs)

        P_noconv_block = np.block(
            [interp1d(kgrid, P_noconv[l], kind='cubic')(self.kpW[l])
             for l in range(3)]).T
        P_conv = (self.window @ P_noconv_block).T
        return np.split(P_conv, 3, axis=0)

#-------------------------------------------------------------------------------

    def P_kmu_z_marg(self, redshift, do_redshift_rescaling, kgrid=None, f=None,
                     D=None, cosmo=None, AP_as_nuisance=False, alpha_par=1,
                     alpha_perp=1, b1=1, b2=0, bG2=0, Psn=0, sigma_z=0, f_out=0,
                     **kwargs):
        """
        Computes the redshift space power spectrum multipoles, marginalizing over
        the EFT parameters c0, c2, c4, ck4, aP, e0k2, e2k2.

        Parameters
        ----------
        redshift: float, redshift used to evaluate the growth functions
        do_redshift_rescaling: bool, if True P_L is computed at z=0 and rescales
                           with the linear growth, if False P_L is computed
                           at `redshift`
        kgrid: array, the output grid on which the multipoles are evaluated
        f, D: float, growth rate and growth factor. If None they are computed
          internally
        cosmo: dict, contains the cosmology. Default is None, uses input
           cosmology from the parameter file
        AP_as_nuisance: bool, if True, values for alpha_par and alpha_perp can
                    be passed, otherwise, they are computed from the
                    Fiducialcosmology and cosmo dictionary
        alpha_par, alpha_perp: AP parameters
        b1, b2, bG2: float, bias parameters
        Psn: float, Poisson shot noise
        sigma_z: float, redshift uncertainty
        f_out: float, fraction of objects that are not in the sample

        Returns
        -------
        2 arrays: the first one contains the convolved power spectrum multipoles
        ordered as P0, P2, P4 and the second one contains the marginalised nuisance
        parameters.
        """
        if do_redshift_rescaling:
            f, D =  self._get_growth_functions(redshift, f=f, D=D, cosmo=cosmo)
        else:
            f, D =  self._get_growth_functions(redshift, f=f, D=1, cosmo=cosmo)

        DZ2 = D*D
        DZ4 = D**4.

        q = kgrid[:, newaxis]
        nu = self.mu
        AP_ampl = 1.

        sigma_r = lightspeed_kms * sigma_z / (100*\
                                              cosmology.Hubble_adim(redshift,
                                                                    self.Om,
                                                                    self.w0,
                                                                    self.wa))

        if self.do_AP:
            q, nu, AP_ampl = self._apply_AP_distortions(kgrid, self.mu,
                                                        redshift,
                                                        cosmo,
                                                        AP_as_nuisance,
                                                        alpha_par, alpha_perp)

        damping_syst = np.exp(-(q*nu*sigma_r)**2) * (1-f_out)**2

        # Rescale wiggle and no-wiggle P(k), Sigma and Sigma2
        Pnw_sub = self.Pnw_int(q) * DZ2
        Pw_sub = self.Pw_int(q) * DZ2
        Sigma2 = self.Sigma2 * DZ2 * self.IRres
        dSigma2 = self.dSigma2 * DZ2 * self.IRres

        # Rescaling loops
        loop22_nw_sub =  self.loop22_nw_int(q) * DZ4
        loop13_nw_sub =  self.loop13_nw_int(q) * DZ4
        loop22_w =  self.loop22_w_int(q) * DZ4
        loop13_w =  self.loop13_w_int(q) * DZ4

        Sig2mu, RSDdamp = self._muxdamp(q, nu, Sigma2, dSigma2, f)

        # Next-to-leading order, counterterm, noise
        PNLO = self.Kaiser(b1, f, nu)**2 * (Pnw_sub + RSDdamp * Pw_sub *
                                                     (1. + q**2 * Sig2mu))

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
        bias13 = array([b1 * self.Kaiser(b1, f, nu),
                        0. * self.Kaiser(b1, f, nu),
                        bG2 * self.Kaiser(b1, f, nu),
                        nu**2 * f * self.Kaiser(b1, f, nu),
                        nu**2 * f * b1 * self.Kaiser(b1, f, nu),
                        (nu * f)**2 * self.Kaiser(b1, f, nu),
                        nu**4 * f**2 * self.Kaiser(b1, f, nu)])

        Pkmu_22_nw = einsum('ijl,ikl->kl', bias22, loop22_nw_sub)
        Pkmu_22_w  = einsum('ijl,ikl->kl', bias22, loop22_w)
        Pkmu_13_nw = einsum('ijl,ikl->kl', bias13, loop13_nw_sub)
        Pkmu_13_w  = einsum('ijl,ikl->kl', bias13, loop13_w)

        Pkmu_22 = Pkmu_22_nw + RSDdamp * Pkmu_22_w
        Pkmu_13 = Pkmu_13_nw + RSDdamp * Pkmu_13_w
        Pkmu    = damping_syst * (PNLO + Pkmu_22 + Pkmu_13)

        Pkmu_bG3 = damping_syst * (self.Kaiser(b1, f, nu) * loop13_nw_sub[1] +
                                   RSDdamp*self.Kaiser(b1, f, nu) * loop13_w[1])
        Pkmu_ctr = damping_syst * (q**2 * (Pnw_sub + RSDdamp * Pw_sub))

        Pell   = cosmology.multipole_projection(self.mu, Pkmu, [0,2,4])

        PG3ell = cosmology.multipole_projection(self.mu, Pkmu_bG3, [0,2,4])
        Pc0    = cosmology.multipole_projection(self.mu, -2.* Pkmu_ctr, [0,2,4])
        Pc2    = cosmology.multipole_projection(
            self.mu, -2.* f * nu**2 * Pkmu_ctr, [0,2,4])
        Pc4    = cosmology.multipole_projection(
            self.mu, -2.* f**2 * nu**4 * Pkmu_ctr, [0,2,4])
        Pck4   = cosmology.multipole_projection(
            self.mu,
            f**4 * nu**4 * self.Kaiser(b1, f, nu)**2 * q**2 *Pkmu_ctr,
            [0,2,4])

        Pa = cosmology.multipole_projection(self.mu, np.full(q.shape, Psn), [0,2,4])
        Pe0k2  = cosmology.multipole_projection(self.mu, Psn*q**2, [0,2,4])
        Pe2k2  = cosmology.multipole_projection(
            self.mu, Psn*q**2 * nu**2, [0,2,4])
        Pml = concatenate((PG3ell, Pc0, Pc2, Pc4, Pck4, Pa, Pe0k2, Pe2k2))

        return AP_ampl*Pell, AP_ampl*Pml

#-------------------------------------------------------------------------------

    def P_kmu_z_marg_scaledep(self, redshift, do_redshift_rescaling, kgrid=None, f=None,
                     D=None, cosmo=None, AP_as_nuisance=False, alpha_par=1,
                     alpha_perp=1, b1=1, b2=0, bG2=0, Psn=0, sigma_z=0, f_out=0,
                     **kwargs):

        if do_redshift_rescaling:
            f, D =  self._get_growth_functions(redshift, f=f, D=D, cosmo=cosmo)
        else:
            f, D =  self._get_growth_functions(redshift, f=f, D=1, cosmo=cosmo)

        DZ2 = D*D
        DZ4 = D**4.

        q = kgrid[:, newaxis]
        nu = self.mu
        AP_ampl = 1.

        sigma_r = lightspeed_kms * sigma_z / (100*\
                                              cosmology.Hubble_adim(redshift,
                                                                    self.Om,
                                                                    self.w0,
                                                                    self.wa))

        if self.do_AP:
            q, nu, AP_ampl = self._apply_AP_distortions(kgrid, self.mu,
                                                        redshift,
                                                        cosmo,
                                                        AP_as_nuisance,
                                                        alpha_par, alpha_perp)

        damping_syst = np.exp(-(q*nu*sigma_r)**2) * (1-f_out)**2

        # Rescale wiggle and no-wiggle P(k), Sigma and Sigma2
        Pnw_sub = self.Pnw_int(q) * DZ2
        Pw_sub = self.Pw_int(q) * DZ2

        Sigma2 = self.Sigma2 * DZ2 * self.IRres
        dSigma2 = self.dSigma2 * DZ2 * self.IRres

        # Rescaling loops
        loop22_nw_sub =  self.loop22_nw_int(q) * DZ4
        loop13_nw_sub =  self.loop13_nw_int(q) * DZ4
        loop22_w =  self.loop22_w_int(q) * DZ4
        loop13_w =  self.loop13_w_int(q) * DZ4

        # Setup of f dimension in case of scale depent growth
        if self.scale_dependent_growth:
            f = f[:,newaxis]

        Sig2mu, RSDdamp = self._muxdamp(q, nu, Sigma2, dSigma2, f)

        # Next-to-leading order, counterterm, noise
        PNLO = self.Kaiser(b1, f, nu)**2 * (Pnw_sub + RSDdamp * Pw_sub *
                                                     (1. + q**2 * Sig2mu))

        # Biases
        bias22 = array([b1**2 * nu**0 * f**0, b1 * b2 * nu**0 * f**0, b1 * bG2 * nu**0 * f**0,
                        b2**2 * nu**0 * f**0, b2 * bG2 * nu**0 * f**0, bG2**2 * nu**0 * f**0,
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
        bias13 = array([b1 * self.Kaiser(b1, f, nu),
                        0. * self.Kaiser(b1, f, nu),
                        bG2 * self.Kaiser(b1, f, nu),
                        nu**2 * f * self.Kaiser(b1, f, nu),
                        nu**2 * f * b1 * self.Kaiser(b1, f, nu),
                        (nu * f)**2 * self.Kaiser(b1, f, nu),
                        nu**4 * f**2 * self.Kaiser(b1, f, nu)])

        # Le dimensioni di 'ijl,ikl->kl' corrispondono a
        # i : elementi della bias expansion
        # j : boh
        # k : wavenumbers (self.kPE)
        # l : mu
        # Use correct einsum
        if self.scale_dependent_growth:
            repl = 'ikl,ikl->kl'
        else:
            repl = 'ijl,ikl->kl'

        Pkmu_22_nw = einsum(repl, bias22, loop22_nw_sub)
        Pkmu_22_w  = einsum(repl, bias22, loop22_w)
        Pkmu_13_nw = einsum(repl, bias13, loop13_nw_sub)
        Pkmu_13_w  = einsum(repl, bias13, loop13_w)

        Pkmu_22 = Pkmu_22_nw + RSDdamp * Pkmu_22_w
        Pkmu_13 = Pkmu_13_nw + RSDdamp * Pkmu_13_w
        Pkmu    = damping_syst * (PNLO + Pkmu_22 + Pkmu_13)

        Pkmu_bG3 = damping_syst * (self.Kaiser(b1, f, nu) * loop13_nw_sub[1] +
                                   RSDdamp*self.Kaiser(b1, f, nu) * loop13_w[1])
        Pkmu_ctr = damping_syst * (q**2 * (Pnw_sub + RSDdamp * Pw_sub))

        Pell   = cosmology.multipole_projection(self.mu, Pkmu, [0,2,4])

        PG3ell = cosmology.multipole_projection(self.mu, Pkmu_bG3, [0,2,4])
        Pc0    = cosmology.multipole_projection(self.mu, -2.* Pkmu_ctr, [0,2,4])
        Pc2    = cosmology.multipole_projection(
            self.mu, -2.* f * nu**2 * Pkmu_ctr, [0,2,4])
        Pc4    = cosmology.multipole_projection(
            self.mu, -2.* f**2 * nu**4 * Pkmu_ctr, [0,2,4])
        Pck4   = cosmology.multipole_projection(
            self.mu,
            f**4 * nu**4 * self.Kaiser(b1, f, nu)**2 * q**2 *Pkmu_ctr,
            [0,2,4])

        Pa = cosmology.multipole_projection(self.mu, np.full(q.shape, Psn), [0,2,4])
        Pe0k2  = cosmology.multipole_projection(self.mu, Psn*q**2, [0,2,4])
        Pe2k2  = cosmology.multipole_projection(
            self.mu, Psn*q**2 * nu**2, [0,2,4])
        Pml = concatenate((PG3ell, Pc0, Pc2, Pc4, Pck4, Pa, Pe0k2, Pe2k2))

        return AP_ampl*Pell, AP_ampl*Pml
    
#-------------------------------------------------------------------------------

    def Pgg_BAO_l(self, redshift, kgrid=None, b1=1, alpha_par=1, alpha_perp=1,
                  Sigma_par=0, Sigma_perp=0, Sigma_rec=0, Sigma_s=0, f=0, B=1,
                  Psn=0, **kwargs):
        """
        Computes the galaxy power spectrum multipoles given the full
        set of Physical and shape parameters

        Parameters
        ----------
        redshift : float
            redshift of the sample
        b1 : float
            linear bias
        alpha_par, alpha_perp : float
            AP parameters
        Sigma_par, Sigma_perp : float
            BAO damping parameters
        Sigma_rec : float
            Sigma of reconstruction, fixed to fiducial value of recon for rec-iso, 0 otherwise
        Sigma_s : float
            NL velocity dispersion (used to model FoG)
        f : float
            growth rate
        B : float
            free amplitude parameter, also includes rs_fid/rs

        Returns
        -------
        Pgg_BAO_l(k) : array, multipoles (2,len(k)) of the galaxy power spectrum
        """
        if self.do_redshift_rescaling:
            _,D =  self._get_growth_functions(redshift,f=None,D=None,cosmo=None)
        else:
            D = 1

        DZ2 = D*D

        beta = f/b1

        elles = np.array([0,2,4])
        mus = np.linspace(-1,1,101)
        kref = 0.2
        norm = self.Pnw_int(kref) #43.42091852 # norm, value of Pk at k=rref

        # Setting up reconstruction template
        if Sigma_rec != 0:
            Sk_array = np.exp(-self.kL**2 * Sigma_rec**2/2.)
        else:
            Sk_array = np.zeros(len(self.kL))
        Sk = interp1d(self.kL, Sk_array)

        #--adding RSD
        sigma_v2 = lambda mu: (1.-mu**2) * Sigma_perp**2 + mu**2 * Sigma_par**2
        Kaiser   = lambda mu,k: (1. + mu**2 * beta * (1. - Sk(k)))**2
        FoG      = lambda mu,k: 1./(1.+k**2 * mu**2 * Sigma_s**2/2.)

        P_mu_k = lambda mu,k: B**2 * Kaiser(mu,k) * FoG(mu,k) * DZ2 * (
            self.Pnw_int(k) + self.Pw_int(k)*np.exp(-k**2 * sigma_v2(mu) / 2.))

        # Adding AP (This part can be rewritten to use a common function,
        # here the binning is preserved)
        F = alpha_par/alpha_perp
        Alpha_resc = (alpha_perp**2 * alpha_par)

        q  = lambda mu,k : k/alpha_perp * (1 + mu**2 * (1/F**2-1))**(0.5)
        nu = lambda mu : mu/F * (1 + mu**2 * (1/F**2-1))**(-0.5)

        # NOTE: B reabsorbes the rescaling, no need to put the rs rescaling
        P_nu_q = lambda mu,k: P_mu_k(nu(mu), q(mu,k))

        P_nu_q_disc = P_nu_q(mus[:,None], kgrid[None,:])

        P_BAO_l = array([simps((2*l+1.)/2.*legendre(l)(mus)[:,None]*P_nu_q_disc,
                               mus, axis=0) for l in elles])

        P_BAO_l[0] += Psn

        P_BAO_l /= Alpha_resc

        #adding Broad-band
        bb_l = np.zeros((len(elles),len(kgrid)))
        for l in range(0,3):
            for i in range(-3,3):
                str_ali = 'a'+str(elles[l])+str(i+3)
                if str_ali in kwargs.keys():
                    bb_l[l] += kwargs[str_ali] * kgrid**i * norm * kref**(-i)

        return P_BAO_l + bb_l
    
#-------------------------------------------------------------------------------

    def Pgg_BAO_l_marg(self, redshift, kgrid=None, b1=1, alpha_par=1, alpha_perp=1,
                  Sigma_par=0, Sigma_perp=0, Sigma_rec=0, Sigma_s=0, f=0, B=1,
                  Psn=0, **kwargs):  
        """
        Computes the galaxy power spectrum in redshift space, adding
        BAO template and Broad-Band template for each multipole.

        Parameters
        ----------
        redshift: float, redshift used to evaluate the growth functions
        kgrid: array, the output grid on which the multipoles are evaluated
        b1: float, bias parameter
        alpha_par, alpha_perp: float, AP parameters
        Sigma_par, Sigma_perp, Sigma_rec, Sigma_s: float, dispersion parameters
        f: float, growth rate
        B: float, shot-noise parameter
        Psn: float, Poisson shot noise

        Returns
        -------
        P_BAO_l: array containing the power spectrum multipoles ordered as P0,P2,P4
        P_bb_l: array containing the Broad-band power spectrum multipoles ordered as
            P0,P2,P4, with each multipole further divided into 7 coefficients
            corresponding to k^i, i=-3,...,3
        """
        if self.do_redshift_rescaling:
            _,D =  self._get_growth_functions(redshift,f=None,D=None,cosmo=None)
        else:
            D = 1

        DZ2 = D*D

        beta = f/b1

        elles = np.array([0,2,4])
        mus = np.linspace(-1,1,101)
        kref = 0.2
        norm = self.Pnw_int(kref) #43.42091852 # norm, value of Pk at k=rref

        # Setting up reconstruction template
        if Sigma_rec != 0:
            Sk_array = np.exp(-self.kL**2 * Sigma_rec**2/2.)
        else:
            Sk_array = np.zeros(len(self.kL))
        Sk = interp1d(self.kL, Sk_array)

        #--adding RSD
        sigma_v2 = lambda mu: (1.-mu**2) * Sigma_perp**2 + mu**2 * Sigma_par**2
        Kaiser   = lambda mu,k: (1. + mu**2 * beta * (1. - Sk(k)))**2
        FoG      = lambda mu,k: 1./(1.+k**2 * mu**2 * Sigma_s**2/2.)

        P_mu_k = lambda mu,k: B**2 * Kaiser(mu,k) * FoG(mu,k) * DZ2 * (
            self.Pnw_int(k) + self.Pw_int(k)*np.exp(-k**2 * sigma_v2(mu) / 2.))

        # Adding AP (This part can be rewritten to use a common function,
        # here the binning is preserved)
        F = alpha_par/alpha_perp
        Alpha_resc = (alpha_perp**2 * alpha_par)

        q  = lambda mu,k : k/alpha_perp * (1 + mu**2 * (1/F**2-1))**(0.5)
        nu = lambda mu : mu/F * (1 + mu**2 * (1/F**2-1))**(-0.5)

        # NOTE: B reabsorbes the rescaling, no need to put the rs rescaling
        P_nu_q = lambda mu,k: P_mu_k(nu(mu), q(mu,k))

        P_nu_q_disc = P_nu_q(mus[:,None], kgrid[None,:])

        P_BAO_l = array([simps((2*l+1.)/2.*legendre(l)(mus)[:,None]*P_nu_q_disc,
                               mus, axis=0) for l in elles])

        P_BAO_l[0] += Psn

        P_BAO_l /= Alpha_resc

        #adding Broad-band
        
        bb_ell = []
        zero_array = np.zeros(len(kgrid))

        for l in range(0,3):
            for i in range(-3, 3):
                arrays = [zero_array.copy(), zero_array.copy(), zero_array.copy()]
                arrays[l] = kgrid**i * norm * kref**(-i)
                bb_ell.append(arrays)

        P_bb_l = np.concatenate(bb_ell)
        
        return P_BAO_l, P_bb_l

#-------------------------------------------------------------------------------

    def tns_model(self, redshift, do_redshift_rescaling, b1=1, sigma_v=0,
                  f=None, D=None, cosmo=None):
        """
        Returns the TNS model for the anisotropic galaxy power spectrum.
        The FoG damping is implemented with Lorentzian.

        Parameters
        ----------
        b1 : float, optional
            Linear bias. Defaults to 1.
        sigma_v : float, optional
            Velocity dispersion. Defaults to 0.
        f : float, optional
            Growth rate.
        D : float, optional
            Growth factor.
        cosmo : dict, optional
            Cosmology dictionary.

        Returns
        -------
        ABsum*D_fog : array
            Anisotropic galaxy power spectrum.
        """
        if cosmo is not None:
            PL = cosmo['PL']
        else:
            PL = self.PL

        if do_redshift_rescaling:
            f, D =  self._get_growth_functions(redshift, f=f, D=D, cosmo=cosmo)
        else:
            f, D =  self._get_growth_functions(redshift, f=f, D=1, cosmo=cosmo)

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

        # Leading order, delta-delta, delta-theta, theta-theta
        PLO = self.Kaiser(b1, f, mu)**2 * PL[:,newaxis]

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
        """
        Computes the galaxy power spectrum in real space given the full set of
        bias parameters, counterterm, and stochastic parameters. If the bias
        parameters are not specified, the real space matter power spectrum is
        computed.

        Parameters
        ----------
        b1 : float, optional
            Linear bias. Defaults to 1.
        b2, bG2, bG3 : float, optional
            Higher-order biases. Defaults to 0.
        c0 : float, optional
            Effective sound-speed. Defaults to 0.
        aP : float, optional
            Constant correction to Poisson shot-noise. Defaults to 0.
        ek2 : float, optional
            k**2 stochastic term. Defaults to 0.
        Psn : float, optional
            Poisson shot-noise. Defaults to 0.
        loop : bool, optional
            If True, computes the 1-loop power spectrum. If False, computes the
            linear model. Defaults to True.

        Returns
        -------
        P_gg(k) : array
            Galaxy power spectrum.
        """
        return ((1. - float(loop))*b1*b1*self.Pk['LO'] +
                float(loop) * (b1*b1*self.Pk['NLO'] + b1*b2*self.Pk['b1b2'] +
                               b1*bG2*self.Pk['b1bG2'] + b2*b2*self.Pk['b2b2'] +
                               b2*bG2*self.Pk['b2bG2'] +
                               bG2*bG2*self.Pk['bG2bG2'] +
                               b1*bG3*self.Pk['b1bG3'] + ek2*self.Pk['k2'] -
                               2.*c0*self.Pk['k2LO']) + (1.+aP)*Psn)

#-------------------------------------------------------------------------------

    def Pgm_real(self, b1=1, b2=0, bG2=0, bG3=0, c0=0, ek2=0, loop=True):
        """
        Computes the galaxy-matter power spectrum in real space using
        the specified bias parameters, counterterm, and stochastic
        parameters.

        Parameters
        ----------
        b1 : float
            Linear bias.
        b2, bG2, bG3 : float
            Higher-order biases.
        c0 : float
            Effective sound-speed.
        ek2 : float
            k**2 stochastic term.
        loop : bool
            If True, computes the 1-loop power spectrum. If False, computes the linear model.

        Returns
        -------
        P_gm(k) : array
            Cross galaxy-matter power spectrum.
        """
        return ((1. - float(loop)) * b1 * self.Pk['LO'] +
                float(loop) * (b1 * self.Pk['NLO'] +
                               0.5 * (b2 * self.Pk['b1b2'] +
                                      bG2*self.Pk['b1bG2'] +
                                      b1*bG3*self.Pk['b1bG3']) +
                               ek2*self.Pk['k2'] - 2. * c0 * self.Pk['k2LO']))

#-------------------------------------------------------------------------------

    def Pmm_real(self, c0=0, loop=True):
        """
        Computes the matter power spectrum in real space given the
        effective sound-speed

        Parameters
        ----------
        c0 : float, optional
            Effective sound-speed. Defaults to 0.
        loop : bool, optional
            If True, computes the 1-loop power spectrum. If False, computes the linear model.
            Defaults to True.

        Returns
        -------
        P_mm(k) : array
            The real space matter power spectrum
        """
        return ((1. - float(loop)) * self.Pk['LO'] +
                float(loop) * (self.Pk['NLO'] - 2. * c0 * self.Pk['k2LO']))

#-------------------------------------------------------------------------------

    def Pgg_l(self, l, f=None, **kwargs):
        """
        Computes the galaxy power spectrum multipoles given the full
        set of bias parameters, counterterm amplitudes, and stochastic
        parameters.  By default, the growth rate is computed from the
        cosmology, but a different value can be specified.

        Parameters
        ----------
        l : int
            Multipole degree
        b1 : float
            Linear bias
        b2, bG2, bG3 : float
            Higher-order biases
        c0, c2, c4, ck4 : float
            EFT counterterms
        aP : float
            Constant deviation from Poisson shot-noise
        e0k2, e2k2 : float
            k**2 corrections to the shot-noise
        Psn : float
            Poisson shot-noise
        f : None or float
            Growth rate

        Returns
        -------
        P_gg,l(k) : array
            Specified multipole of the galaxy power spectrum
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

    def Bispfunc_RSD(self, redshift, k1, k2, k3, D=None, cosmo=None,
                     do_redshift_rescaling=True):
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
        if do_redshift_rescaling:
            _, D = self._get_growth_functions(redshift, f=0, D=D, cosmo=cosmo)
        else:
            D = 1

        if cosmo == None:
            PL = D**2 * self.PL
        else:
            PL = D**2 * cosmo['PL']

        if self.do_AP:
            B0 = self.Bl_terms_AP(k1, k2, k3, self.Is0, PL)

            l2 = self.Bl_terms_AP(k1, k2, k3, self.Is2, PL)
            l4 = self.Bl_terms_AP(k1, k2, k3, self.Is4, PL)

        else:
            B0 = self.Bl_terms(0, k1, k2, k3, PL)

            l2 = self.Bl_terms(2, k1, k2, k3, PL)
            l4 = self.Bl_terms(4, k1, k2, k3, PL)

        B2 = 2.5 * (3. * l2 - B0) / (2.*pi)
        B4 = 1.125 * (35. * l4 - 30. * l2 + 3. * B0) / (2.*pi)

        B0 /= (2.*pi)
        B0 = np.insert(B0, len(B0), ones(len(k1)), axis=0)

        return B0, B2, B4

#-------------------------------------------------------------------------------

    def Bl_terms(self, a, k1, k2, k3, PL):
        """Bl terms

        Computes the building block for the tree-level redshift space
        bispectrum multipoles

        Parameters
        ----------
        `a`: int, power of mu1
        `k1`, `k2`, `k3`: float or array, sides of Fourier triangles

        Returns
        -------
        Array with the building blocks for the tree-level bispectrum
        multipoles in the following order
        """

        PkL = InterpolatedUnivariateSpline(self.kL, PL, k=3)

        def muij(K):
            ki, kj, kl = K
            return 0.5*(kl**2 - ki**2 - kj**2)/(ki*kj)

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

        mu12 = muij([k1, k2, k3])
        mu23 = muij([k2, k3, k1])
        mu31 = muij([k3, k1, k2])

        F_12 = bispfunc.F_2(k1, k2, k3)
        F_23 = bispfunc.F_2(k2, k3, k1)
        F_31 = bispfunc.F_2(k3, k1, k2)
        G_12 = bispfunc.G_2(k1, k2, k3)
        G_23 = bispfunc.G_2(k2, k3, k1)
        G_31 = bispfunc.G_2(k3, k1, k2)

        S_12 = mu12**2 - 1.
        S_23 = mu23**2 - 1.
        S_31 = mu31**2 - 1.

        P12 = PkL(k1) * PkL(k2)
        P23 = PkL(k2) * PkL(k3)
        P31 = PkL(k3) * PkL(k1)

        K = [k1, k2, k3]

        x = [(x, y) for x in [0,1,2,3,4,5,6]
             for y in ['00', '01', '02', '03', '04',
                       '10', '11', '12', '13', '14',
                       '20', '21', '22', '23',
                       '30', '31', '32', '34',
                       '40', '41', '43']]

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

        return array((Bb13, Bb12b2, Bb12bG2, Bfb12, Bfb13, Bf2b12, Bfb1b2,
                      Bfb1bG2, Bf2b1, Bf3b1, Bf2b2, Bf2bG2, Bf3, Bf4, Bb12a1,
                      Bfb1a1, Bf2a1))

#-------------------------------------------------------------------------------

    def Bl_terms_AP(self, k1, k2, k3, Is, PL):

        ###array of angle integrals from the corresponding dictionary
        I0, I0mu = Is['I0'], Is['I0mu']
        I2, I2mu, tI2 = Is['I2'], Is['I2mu'], Is['tI2']
        I11, I11mu, tI11 = Is['I11'], Is['I11mu'], Is['tI11']
        I13, I13mu, tI13 = Is['I13'], Is['I13mu'], Is['tI13']
        I112, I112mu, tI112 = Is['I112'], Is['I112mu'], Is['tI112']
        I22, I22mu, tI22 = Is['I22'], Is['I22mu'], Is['tI22']
        I114, I114mu, tI114 = Is['I114'], Is['I114mu'], Is['tI114']
        I123, I123mu, tI123 = Is['I123'], Is['I123mu'], Is['tI123']
        I222, I222mu, tI222 = Is['I222'], Is['I222mu'], Is['tI222']
        I134, I134mu, tI134 = Is['I134'], Is['I134mu'], Is['tI134']
        I4, I4mu, tI4 = Is['I4'], Is['I4mu'], Is['tI4']


        ###functions of k1, k2, k3 that we compute at initialisation
        K = self.K
        F = self.F
        G = self.G
        S = self.S
        dF = self.dF
        dG = self.dG
        dS = self.dS
        KK = self.KK
        dKK = self.dKK

        PkL = InterpolatedUnivariateSpline(self.kL, PL, k=3)
        P12 = PkL(k1) * PkL(k2)
        P23 = PkL(k2) * PkL(k3)
        P31 = PkL(k3) * PkL(k1)

        ###derivative of the power spectrum###
        P1 = PkL(k1)
        P2 = PkL(k2)
        P3 = PkL(k3)
        dP1_dlnk1 = k1*misc.derivative(PkL, k1, dx=1e-6)
        dP2_dlnk2 = k2*misc.derivative(PkL, k2, dx=1e-6)
        dP3_dlnk3 = k3*misc.derivative(PkL, k3, dx=1e-6)

        P = [P1, P2, P3]
        dP = [dP1_dlnk1, dP2_dlnk2, dP3_dlnk3]
        PP = [P12, P31, P23]
        dPP = [(dP1_dlnk1*P2, P1*dP2_dlnk2, zeros(len(P1))),
               (dP1_dlnk1*P3, zeros(len(P1)), dP3_dlnk3*P1),
               (zeros(len(P1)), dP2_dlnk2*P3, dP3_dlnk3*P2)]
        FP = [K['F_12']*P12, K['F_31']*P31, K['F_23']*P23]
        GP = [K['G_12']*P12, K['G_31']*P31, K['G_23']*P23]
        SP = [K['S_12']*P12, K['S_31']*P31, K['S_23']*P23]


        ###b13###
        Bb13_0 = zeros(len(P1))
        Bb13_perp = zeros(len(P1))
        Bb13_dif = zeros(len(P1))

        Bb12b2_0 = zeros(len(P1))
        Bb12b2_perp = zeros(len(P1))
        Bb12b2_dif = zeros(len(P1))

        Bb12bG2_0 = zeros(len(P1))
        Bb12bG2_perp = zeros(len(P1))
        Bb12bG2_dif = zeros(len(P1))

        Bfb12_0 = zeros(len(P1))
        Bfb12_perp = zeros(len(P1))
        Bfb12_dif = zeros(len(P1))
        permfb12 = [(0,1,2), (2,0,1), (1,2,0)]

        Bfb13_0 = zeros(len(P1))
        Bfb13_perp = zeros(len(P1))
        Bfb13_dif = zeros(len(P1))
        permfb13 = [(0, 1), (2, 1), (2,0)]

        Bf2b12_0 = zeros(len(P1))
        Bf2b12_perp = zeros(len(P1))
        Bf2b12_dif = zeros(len(P1))
        permf2b12 = [(0, 2, 5, 0), (2, 1, 3, 0), (4, 1, 1, 2)]

        Bfb1b2_0 = zeros(len(P1))
        Bfb1b2_perp = zeros(len(P1))
        Bfb1b2_dif = zeros(len(P1))
        permfb1b2 = [(0, 1), (0, 2), (2, 1)]

        Bfb1bG2_0 = zeros(len(P1))
        Bfb1bG2_perp = zeros(len(P1))
        Bfb1bG2_dif = zeros(len(P1))
        permfb1bG2 = [(0, 1), (0, 2), (2, 1)]

        Bf2b1_0  = zeros(len(P1))
        Bf2b1_perp = zeros(len(P1))
        Bf2b1_dif = zeros(len(P1))
        permf2b1 = [(0, 1, 2), (1, 0, 2), (2, 0, 1)]

        Bf3b1_0 = zeros(len(P1))
        Bf3b1_perp = zeros(len(P1))
        Bf3b1_dif = zeros(len(P1))
        permf3b1 = [(0, 2, 1, 4), (2, 0, 1, 1), (2, 5, 0, 3)]

        Bf2b2_0 = zeros(len(P1))
        Bf2b2_perp = zeros(len(P1))
        Bf2b2_dif = zeros(len(P1))

        Bf2bG2_0 = zeros(len(P1))
        Bf2bG2_perp = zeros(len(P1))
        Bf2bG2_dif = zeros(len(P1))

        Bf3_0 = zeros(len(P1))
        Bf3_perp = zeros(len(P1))
        Bf3_dif = zeros(len(P1))

        Bf4_0 = zeros(len(P1))
        Bf4_perp = zeros(len(P1))
        Bf4_dif = zeros(len(P1))
        permf4 = [(4, 2), (1, 0), (3, 5)]

        Bb12a1_0 = zeros(len(P1))
        Bb12a1_perp = zeros(len(P1))
        Bb12a1_dif = zeros(len(P1))

        Bfb1a1_0 = zeros(len(P1))
        Bfb1a1_perp = zeros(len(P1))
        Bfb1a1_dif = zeros(len(P1))

        Bf2a1_0 = zeros(len(P1))
        Bf2a1_perp = zeros(len(P1))
        Bf2a1_dif = zeros(len(P1))

        for i in range(3):
            Bb13_0 += FP[i]*I0[0]
            Bb13_perp += (sum(dF[i], axis=0)*PP[i]+\
                          F[i]*sum(dPP[i], axis=0))*I0[0]
            Bb12b2_0 += 0.5*PP[i]*I0[0]
            Bb12b2_perp += 0.5*sum(dPP[i], axis=0)*I0[0]
            Bb12bG2_0 += SP[i]*I0[0]
            Bb12bG2_perp += (sum(dS[i], axis=0)*PP[i]+\
                             S[i]*sum(dPP[i], axis=0))*I0[0]
            Bfb12_0 += (FP[i]*( I2[permfb12[i][0]]+I2[permfb12[i][1]] ) +
                        GP[i]* I2[permfb12[i][2]])
            Bfb12_perp += ((sum(dF[i], axis=0)*PP[i]+F[i]*sum(dPP[i], axis=0))*
                           (I2[permfb12[i][0]]+I2[permfb12[i][1]])+
                           (sum(dG[i], axis=0)*PP[i]+G[i]*sum(dPP[i], axis=0))*
                           I2[permfb12[i][2]])
            Bfb12_dif += (FP[i]*( tI2[permfb12[i][0]]+tI2[permfb12[i][1]]) +
                          GP[i]* tI2[permfb12[i][2]])
            Bfb13_0 += -0.5*((I11[permfb13[i][0]]*KK[int(2*i)]+\
                              I11[permfb13[i][1]]*KK[int(2*i+1)])*PP[i])
            Bfb13_perp += -0.5*((I11[permfb13[i][0]]*\
                                 (sum(dKK[int(2*i)], axis=0)*\
                                  PP[i]+KK[int(2*i)]*sum(dPP[i], axis=0))+\
                                 I11[permfb13[i][1]]*\
                                 (sum(dKK[int(2*i+1)], axis=0)*PP[i]+\
                                  KK[int(2*i+1)]*sum(dPP[i], axis=0))))
            Bfb13_dif += -0.5*((tI11[permfb13[i][0]]*KK[int(2*i)]+\
                                tI11[permfb13[i][1]]*KK[int(2*i+1)])*PP[i])
            Bf2b12_0 += -0.5*(((I13[permf2b12[i][0]]+2.*I112[permf2b12[i][1]])*\
                               KK[int(2*i)]+\
                               (I13[permf2b12[i][2]]+2.*I112[permf2b12[i][3]])*\
                               KK[int(2*i+1)])*PP[i])
            Bf2b12_perp += -0.5*(((I13[permf2b12[i][0]]+\
                                   2.*I112[permf2b12[i][1]])*\
                                  (sum(dKK[int(2*i)], axis=0)*\
                                   PP[i]+KK[int(2*i)]*sum(dPP[i], axis=0))+\
                                  (I13[permf2b12[i][2]]+\
                                   2.*I112[permf2b12[i][3]])*\
                                  (sum(dKK[int(2*i+1)], axis=0)*PP[i]+\
                                   KK[int(2*i+1)]*sum(dPP[i], axis=0))))
            Bf2b12_dif += -0.5*(((tI13[permf2b12[i][0]]+\
                                  2.*tI112[permf2b12[i][1]])*KK[int(2*i)]+\
                                 (tI13[permf2b12[i][2]]+\
                                  2.*tI112[permf2b12[i][3]])*KK[int(2*i+1)])*PP[i])
            Bfb1b2_0 += 0.5*(I2[permfb1b2[i][0]]+I2[permfb1b2[i][1]])*PP[i]
            Bfb1b2_perp += 0.5*((I2[permfb1b2[i][0]]+
                                 I2[permfb1b2[i][1]]))*sum(dPP[i], axis=0)
            Bfb1b2_dif += 0.5*(tI2[permfb1b2[i][0]]+tI2[permfb1b2[i][1]])*PP[i]
            Bfb1bG2_0 += (I2[permfb1bG2[i][0]]+I2[permfb1bG2[i][1]])*SP[i]
            Bfb1bG2_perp += ((I2[permfb1bG2[i][0]]+I2[permfb1bG2[i][1]]))*\
                (sum(dS[i], axis=0)*PP[i]+S[i]*sum(dPP[i], axis=0))
            Bfb1bG2_dif += (tI2[permfb1bG2[i][0]]+tI2[permfb1bG2[i][1]])*SP[i]
            Bf2b1_0 += (I22[permf2b1[i][0]]*FP[i] +
                        (I22[permf2b1[i][1]] + I22[permf2b1[i][2]]) * GP[i])
            Bf2b1_perp += (I22[permf2b1[i][0]]*\
                           (sum(dF[i], axis=0)*PP[i]+F[i]*sum(dPP[i], axis=0))+\
                           (I22[permf2b1[i][1]] + I22[permf2b1[i][2]])*
                           (sum(dG[i], axis=0)*PP[i]+G[i]*sum(dPP[i], axis=0)))
            Bf2b1_dif += (tI22[permf2b1[i][0]]*FP[i] +
                          (tI22[permf2b1[i][1]] + tI22[permf2b1[i][2]]) * GP[i])
            Bf3b1_0 += -0.5*((I114[permf3b1[i][0]]+\
                              2.*I123[permf3b1[i][1]])*KK[int(2*i)]+
                             (I114[permf3b1[i][2]]+2.*I123[permf3b1[i][3]])*\
                             KK[int(2*i)+1])*PP[i]
            Bf3b1_perp += -0.5*((I114[permf3b1[i][0]]+2.*I123[permf3b1[i][1]])*\
                                (sum(dKK[int(2*i)], axis=0)*PP[i] +
                                 KK[int(2*i)]*sum(dPP[i], axis=0)) +
                                (I114[permf3b1[i][2]]+2.*I123[permf3b1[i][3]])*\
                                (sum(dKK[int(2*i+1)], axis=0)*PP[i] +
                                 KK[int(2*i+1)]*sum(dPP[i], axis=0)))
            Bf3b1_dif += -0.5*((tI114[permf3b1[i][0]]+\
                                2.*tI123[permf3b1[i][1]])*KK[int(2*i)]+\
                               (tI114[permf3b1[i][2]]+\
                                2.*tI123[permf3b1[i][3]])*KK[int(2*i)+1])*PP[i]
            Bf2b2_0 += 0.5 * I22[i] * PP[i]
            Bf2b2_perp += 0.5 * I22[i] * sum(dPP[i], axis=0)
            Bf2b2_dif += 0.5 * tI22[i] * PP[i]
            Bf2bG2_0 += I22[i] * SP[i]
            Bf2bG2_perp += I22[i]*(sum(dS[i], axis=0)*PP[i]+\
                                   S[i]*sum(dPP[i], axis=0))
            Bf2bG2_dif += tI22[i] * SP[i]
            Bf3_0 += GP[i]*I222[0]
            Bf3_perp += (sum(dG[i], axis=0)*PP[i]+\
                         G[i]*sum(dPP[i], axis=0))*I222[0]
            Bf3_dif += GP[i]*tI222[0]
            Bf4_0 += -0.5*(I134[permf4[i][0]]*KK[int(2*i)] +\
                           I134[permf4[i][1]]*KK[int(2*i+1)]) * PP[i]
            Bf4_perp += -0.5*(I134[permf4[i][0]]*\
                              (sum(dKK[int(2*i)], axis=0)*PP[i]+\
                               KK[int(2*i)]*sum(dPP[i], axis=0))+
                              I134[permf4[i][1]]*\
                              (sum(dKK[int(2*i+1)], axis=0)*PP[i]+\
                               KK[int(2*i+1)]*sum(dPP[i], axis=0)))
            Bf4_dif += -0.5*(tI134[permf4[i][0]]*KK[int(2*i)]+
                             tI134[permf4[i][1]]*KK[int(2*i+1)])*PP[i]
            Bb12a1_0 += 0.5 * P[i]*I0[0]
            Bb12a1_perp += 0.5 * dP[i]*I0[0]
            Bb12a1_dif += 0.5 * I0mu[i] * dP[i]
            Bfb1a1_0 += I2[i] * P[i]
            Bfb1a1_perp += I2[i] * dP[i]
            Bfb1a1_dif += (I2mu[i][i] * dP[i] + tI2[i] * P[i])
            Bf2a1_0 += 0.5 * I4[i] * P[i]
            Bf2a1_perp += 0.5 * I4[i] * dP[i]
            Bf2a1_dif += 0.5 * (I4mu[i][i] * dP[i] + tI4[i] * P[i])

            for j in range(3):
                Bb13_dif += (dF[i][j]*PP[i]+F[i]*dPP[i][j])*I0mu[j]
                Bb12b2_dif += 0.5*dPP[i][j]*I0mu[j]
                Bb12bG2_dif += (dS[i][j]*PP[i]+S[i]*dPP[i][j])*I0mu[j]
                Bfb12_dif += ((dF[i][j]*PP[i]+F[i]*dPP[i][j])*\
                              (I2mu[permfb12[i][0]][j]+I2mu[permfb12[i][1]][j])+\
                              (dG[i][j]*PP[i]+G[i]*dPP[i][j])*\
                              I2mu[permfb12[i][2]][j])
                Bfb13_dif += -0.5*((I11mu[permfb13[i][0]][j]*\
                                    (dKK[int(2*i)][j]*PP[i]+KK[int(2*i)]*\
                                     dPP[i][j])+\
                                    I11mu[permfb13[i][1]][j]*\
                                    (dKK[int(2*i+1)][j]*PP[i]+\
                                     KK[int(2*i+1)]*dPP[i][j])))
                Bf2b12_dif += -0.5*(((I13mu[permf2b12[i][0]][j]+\
                                      2.*I112mu[permf2b12[i][1]][j])*\
                                     (dKK[int(2*i)][j]*PP[i]+KK[int(2*i)]*dPP[i][j])+\
                                     (I13mu[permf2b12[i][2]][j]+\
                                      2.*I112mu[permf2b12[i][3]][j])*\
                                     (dKK[int(2*i+1)][j]*PP[i]+\
                                      KK[int(2*i+1)]*dPP[i][j])))
                Bfb1b2_dif += 0.5*((I2mu[permfb1b2[i][0]][i]+
                                    I2mu[permfb1b2[i][1]][j]))*dPP[i][j]
                Bfb1bG2_dif += ((I2mu[permfb1bG2[i][0]][j]+
                                 I2mu[permfb1bG2[i][1]][j]))*\
                                 (dS[i][j]*PP[i]+S[i]*dPP[i][j])
                Bf2b1_dif += (I22mu[permf2b1[i][0]][j]*\
                              (dF[i][j]*PP[i]+F[i]*dPP[i][j])+\
                              (I22mu[permf2b1[i][1]][j]+I22mu[permf2b1[i][2]][j])*\
                              (dG[i][j]*PP[i]+G[i]*dPP[i][j]))
                Bf3b1_dif += -0.5*((I114mu[permf3b1[i][0]][j]+\
                                    2.*I123mu[permf3b1[i][1]][j])*\
                                   (dKK[int(2*i)][j]*PP[i] +\
                                    KK[int(2*i)]*dPP[i][j]) +\
                                   (I114mu[permf3b1[i][2]][j]+\
                                    2.*I123mu[permf3b1[i][3]][j])*\
                                   (dKK[int(2*i+1)][j]*PP[i]+\
                                    KK[int(2*i+1)]*dPP[i][j]))
                Bf2b2_dif += 0.5 * I22mu[i][j] * dPP[i][j]
                Bf2bG2_dif += I22mu[i][j]*(dS[i][j]*PP[i]+\
                                           S[i]*dPP[i][j])
                Bf3_dif += I222mu[j] * (dG[i][j]*PP[i]+G[i]*dPP[i][j])
                Bf4_dif += -0.5*(I134mu[permf4[i][0]][j]*
                                 (dKK[int(2*i)][j]*PP[i]+KK[int(2*i)]*dPP[i][j])+\
                                 I134mu[permf4[i][1]][j]*
                                 (dKK[int(2*i+1)][j]*PP[i]+KK[int(2*i+1)]*dPP[i][j]))

        return array((Bb13_0, Bb12b2_0, Bb12bG2_0, Bfb12_0, Bfb13_0, Bf2b12_0,
                      Bfb1b2_0, Bfb1bG2_0, Bf2b1_0, Bf3b1_0, Bf2b2_0, Bf2bG2_0,
                      Bf3_0, Bf4_0, Bb12a1_0, Bfb1a1_0, Bf2a1_0, Bb13_perp,
                      Bb12b2_perp, Bb12bG2_perp, Bfb12_perp, Bfb13_perp,
                      Bf2b12_perp, Bfb1b2_perp, Bfb1bG2_perp, Bf2b1_perp,
                      Bf3b1_perp, Bf2b2_perp, Bf2bG2_perp, Bf3_perp, Bf4_perp,
                      Bb12a1_perp, Bfb1a1_perp, Bf2a1_perp, Bb13_dif, Bb12b2_dif,
                      Bb12bG2_dif, Bfb12_dif, Bfb13_dif, Bf2b12_dif, Bfb1b2_dif,
                      Bfb1bG2_dif, Bf2b1_dif, Bf3b1_dif, Bf2b2_dif, Bf2bG2_dif,
                      Bf3_dif, Bf4_dif, Bb12a1_dif, Bfb1a1_dif, Bf2a1_dif))

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
