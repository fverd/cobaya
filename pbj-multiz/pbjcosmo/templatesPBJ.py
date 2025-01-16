import numpy as np
import math
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import simps
from scipy.special import legendre

class PBJtemplates:

    def Bggg_eff(self, PLO = None):
        """
        Computes bispectrum templates by evaluating all bispectrum contributions
        in real space on the sorted effective momenta
        """
        if PLO is None:
            PLO = self.Pk['LO']

        mu12 = 0.5*(self.kB3E**2-self.kB1E**2-self.kB2E**2)/(self.kB1E*self.kB2E)
        mu23 = 0.5*(self.kB1E**2-self.kB2E**2-self.kB3E**2)/(self.kB2E*self.kB3E)
        mu31 = 0.5*(self.kB2E**2-self.kB3E**2-self.kB1E**2)/(self.kB3E*self.kB1E)

        F2_12 = 5./7 + 0.5*mu12*(self.kB1E/self.kB2E+self.kB2E/self.kB1E) + 2.*mu12**2/7
        F2_23 = 5./7 + 0.5*mu23*(self.kB2E/self.kB3E+self.kB3E/self.kB2E) + 2.*mu23**2/7
        F2_31 = 5./7 + 0.5*mu31*(self.kB3E/self.kB1E+self.kB1E/self.kB3E) + 2.*mu31**2/7

        S_12 = mu12**2 - 1.
        S_23 = mu23**2 - 1.
        S_31 = mu31**2 - 1.

        PkL = InterpolatedUnivariateSpline(self.kL, PLO, k=3)
        Pk1 = PkL(self.kB1E)
        Pk2 = PkL(self.kB2E)
        Pk3 = PkL(self.kB3E)

        return (2.*(F2_12*Pk1*Pk2 + F2_23*Pk2*Pk3 + F2_31*Pk3*Pk1),
                (Pk1*Pk2 + Pk2*Pk3 + Pk3*Pk1),
                2.*(S_12*Pk1*Pk2 + S_23*Pk2*Pk3 + S_31*Pk3*Pk1),
                (Pk1+Pk2+Pk3),
                np.ones(self.N_T))

#-------------------------------------------------------------------------------

    def Blgg_eff(self):
        """
        Computes bispectrum templates by evaluating all bispectrum contributions
        in redshift space on the sorted effective momenta
        """

        B0, B2, B4 = self.Bispfunc_RSD(self.z, self.kB1E, self.kB2E, self.kB3E,
                                       do_redshift_rescaling=self.do_redshift_rescaling)

        return (B0, B2, B4)

#-------------------------------------------------------------------------------

    def varPk(self, b1=1, b2=0, bG2=0, bG3=0, c0=0, aP=0, ek2=0, Psn=0):
        """
        Computes the variance for the galaxy power spectrum in real space using
        the effective expansion approach

        b1:           float, linear bias
        b2, bG2, bG3: float, higher-order biases
        c0:           float, effective sound-speed
        aP:           float, constant correction to Poisson shot-noise
        ek2:          float, k**2 stochastic term
        Psn:          float, Poisson shot-noise
        """
        Pgg =((b1*b1*self.Pk['NLO'] + b1*b2*self.Pk['b1b2'] +
               b1*bG2*self.Pk['b1bG2'] + b2*b2*self.Pk['b2b2'] +
               b2*bG2*self.Pk['b2bG2'] + bG2*bG2*self.Pk['bG2bG2'] +
               b1*bG3*self.Pk['b1bG3'] + ek2*self.Pk['k2']  -
               2.*c0*self.Pk['k2LO']) + (1.+aP)*Psn)

        Pk   = InterpolatedUnivariateSpline(self.kL, Pgg, k=3)
        D1Pk = Pk.derivative(n=1)
        D2Pk = Pk.derivative(n=2)

        kE  = self.kPE
        kE2 = self.kPE2

        return 2.*(Pk(kE)**2+(D1Pk(kE)**2+Pk(kE)*D2Pk(kE))*(kE2-kE**2))/self.Nk

#-------------------------------------------------------------------------------

    def varBk(self, b1=1, b2=0, bG2=0, bG3=0, c0=0, aP=0, ek2=0, Psn=0):
        """
        Computes the variance for the galaxy bispectrum in real space using the
        effective expansion approach

        b1:           float, linear bias
        b2, bG2, bG3: float, higher-order biases
        c0:           float, effective sound-speed
        aP:           float, constant correction to Poisson shot-noise
        ek2:          float, k**2 stochastic term
        Psn:          float, Poisson shot-noise
        """
        Pgg =((b1*b1*self.Pk['NLO'] + b1*b2*self.Pk['b1b2'] +
              b1*bG2*self.Pk['b1bG2'] + b2*b2*self.Pk['b2b2'] +
              b2*bG2*self.Pk['b2bG2'] + bG2*bG2*self.Pk['bG2bG2'] +
              b1*bG3*self.Pk['b1bG3'] + ek2*self.Pk['k2']  -
              2.*c0*self.Pk['k2LO']) + (1.+aP)*Psn)

        Pk = InterpolatedUnivariateSpline(self.kL, Pgg, k=3)
        D1Pk = Pk.derivative(n=1)
        D2Pk = Pk.derivative(n=2)

        q1, q2, q3, q12, q22, q32, q1q2, q2q3, q3q1 = self.RebinnedPtEffB.T[:9]

        q1 *= self.kf; q2 *= self.kf; q3 *= self.kf
        q12 *= self.kf**2; q22 *= self.kf**2; q32 *= self.kf**2; q1q2 *=self.kf**2; q2q3 *= self.kf**2; q3q1 *= self.kf**2

        v0 = Pk(q1)*Pk(q2)*Pk(q3)
        v1 = (Pk(q1)*(D1Pk(q2)*D1Pk(q3)*(q2q3 - q2*q3) +
                      0.5*Pk(q2)*D2Pk(q3)*(q32-q3*q3) +
                      0.5*Pk(q3)*D2Pk(q2)*(q22-q2*q2)))
        v2 = (Pk(q2)*(D1Pk(q3)*D1Pk(q1)*(q3q1 - q3*q1) +
                      0.5*Pk(q3)*D2Pk(q1)*(q12-q1*q1) +
                      0.5*Pk(q1)*D2Pk(q3)*(q32-q3*q3)))
        v3 = (Pk(q3)*(D1Pk(q1)*D1Pk(q2)*(q1q2 - q1*q2) +
                      0.5*Pk(q1)*D2Pk(q2)*(q22-q2*q2) +
                      0.5*Pk(q2)*D2Pk(q1)*(q12-q1*q1)))

        return 6.*(2.*np.pi)**3*(v0+v1+v2+v3)/self.NTk/(self.kf**3)

#-------------------------------------------------------------------------------

    def Pgg_template(self, b1=1, b2=0, bG2=0, bG3=0, c0=0, aP=0, ek2=0, Psn=0):
        """
        Computes the total template for the galaxy power spectrum in real space

        b1:           float, linear bias
        b2, bG2, bG3: float, higher-order biases
        c0:           float, effective sound-speed
        aP:           float, constant correction to Poisson shot-noise
        ek2:          float, k**2 stochastic term
        Psn:          float, Poisson shot-noise
        """
        return ((b1*b1*self.P_ell_Templ['Pk']['NLO'] +
                 b1*b2*self.P_ell_Templ['Pk']['b1b2'] +
                 b1*bG2*self.P_ell_Templ['Pk']['b1bG2'] +
                 b2*b2*self.P_ell_Templ['Pk']['b2b2'] +
                 b2*bG2*self.P_ell_Templ['Pk']['b2bG2'] +
                 bG2*bG2*self.P_ell_Templ['Pk']['bG2bG2'] +
                 b1*bG3*self.P_ell_Templ['Pk']['b1bG3'] +
                 ek2*self.P_ell_Templ['Pk']['k2']  -
                 2.*c0*self.P_ell_Templ['Pk']['k2LO']) + (1.+aP)*Psn)

#-------------------------------------------------------------------------------

    def Bggg_template(self, b1=1, b2=0, bG2=0, a1=0, a2=0, Psn=0):
        """
        Computes the total template for the galaxy bispectrum in real space

        b1:      float, linear bias
        b2, bG2: float, higher-order biases
        a1, a2:  float, deviations from Poisson shot-noise
        Psn:     float, Poisson shot-noise
        """
        return (b1*b1*b1*self.BTempl['b13'] +
                b1*b1*b2*self.BTempl['b12b2'] +
                b1*b1*bG2*self.BTempl['b12bG2'] +
                b1*b1*(1.+a1)*Psn*self.BTempl['b12a1'] + (1.+a2)*Psn*Psn)

#-------------------------------------------------------------------------------

    def Plgg_template(self, l, b1=1, b2=0, bG2=0, bG3=0, c0=0, c2=0, c4=0,
                      ck4=0, aP=0, e0k2=0, e2k2=0, Psn=0, f=None):
        """
        Computes the total template for the galaxy power spectrum multipoles in
        redshift space. By default, the growth rate is computed from the
        cosmology, but a different value can be specified. However, infrared
        resummation is computed using the f from the cosmology.

        l:               int, multipole degree
        b1:              float, linear bias
        b2, bG2, bG3:    float, higher-order biases
        c0, c2, c4, ck4: float, EFT counterterms
        aP:              float, constant deviation from Poisson shot-noise
        e0k2, e2k2:      float, k**2 corrections to the shot-noise
        Psn:             float, Poisson shot-noise
        f:               None or float, growth rate
        """
        if f == None: f = self.f

        sl = 'P'+str(l)

        Plk = self.P_ell_Templ[sl]
        Pdet = (b1*b1*Plk['b1b1'] + f*b1*Plk['fb1'] + f*f*Plk['f2'] +
                b1*b2*Plk['b1b2'] + b1*bG2*Plk['b1bG2'] + b2*b2*Plk['b2b2'] +
                b2*bG2*Plk['b2bG2'] + bG2*bG2*Plk['bG2bG2'] + f*b2*Plk['fb2'] +
                f*bG2*Plk['fbG2'] + (f*b1)**2*Plk['f2b12'] +
                f*b1*b1*Plk['fb12'] + f*b1*b2*Plk['fb1b2'] +
                f*b1*bG2*Plk['fb1bG2'] + f*f*b1*Plk['f2b1'] +
                f*f*b2*Plk['f2b2'] + f*f*bG2*Plk['f2bG2'] + f**4*Plk['f4'] +
                f**3*Plk['f3'] + f**3*b1*Plk['f3b1'] + b1*bG3*Plk['b1bG3'] +
                f*bG3*Plk['fbG3'] - 2.*c0*Plk['c0'] - 2.*f*c2*Plk['fc2'] -
                2.*f*f*c4*Plk['f2c4'] + f**4*b1**2*ck4*Plk['f4b12ck4'] +
                2.*f**5*b1*ck4*Plk['f5b1ck4'] + f**6*ck4*Plk['f6ck4'])

        Pstoch = ((l*l/8. - 0.75*l + 1.) * ((1. + aP)*Plk['aP'] +
                                            e0k2 * Plk['e0k2']) * Psn +
                  (-l*l/4. + l) * e2k2 * Plk['e2k2'] * Psn)

        return Pdet+Pstoch

#-------------------------------------------------------------------------------

    def varP00gg(self, b1=1, b2=0, bG2=0, bG3=0, c0=0, c2=0, c4=0, ck4=0, aP=0,
                 e0k2=0, e2k2=0, Psn=0, f=None):
        """
        Computes the Gaussian variance of the monopole of the power spectrum
        given the full set of bias and stochastic parameters

        b1:              float, linear bias
        b2, bG2, bG3:    float, higher-order biases
        c0, c2, c4, ck4: float, EFT counterterms
        aP:              float, constant deviation from Poisson shot-noise
        e0k2, e2k2:      float, k**2 corrections to the shot-noise
        Psn:             float, Poisson shot-noise
        f:               None or float, growth rate
        """
        P0 = self.Plgg_template(0, b1=b1, b2=b2, bG2=bG2, bG3=bG3, c0=c0, c2=c2,
                                c4=c4, ck4=ck4, aP=aP, e0k2=e0k2, e2k2=e2k2,
                                Psn=Psn, f=f)
        P2 = self.Plgg_template(2, b1=b1, b2=b2, bG2=bG2, bG3=bG3, c0=c0, c2=c2,
                                c4=c4, ck4=ck4, aP=aP, e0k2=e0k2, e2k2=e2k2,
                                Psn=Psn, f=f)
        P4 = self.Plgg_template(4, b1=b1, b2=b2, bG2=bG2, bG3=bG3, c0=c0, c2=c2,
                                c4=c4, ck4=ck4, aP=aP, e0k2=e0k2, e2k2=e2k2,
                                Psn=Psn, f=f)

        return 2.*(P0**2 + 0.2*P2**2 + P4**2/9.)/self.Nk

#-------------------------------------------------------------------------------

    def varP22gg(self, b1=1, b2=0, bG2=0, bG3=0, c0=0, c2=0, c4=0, ck4=0, aP=0,
                 e0k2=0, e2k2=0, Psn=0, f=None):
        """
        Computes the Gaussian variance of the quadrupole of the power spectrum
        given the full set of bias and stochastic parameters

        b1:              float, linear bias
        b2, bG2, bG3:    float, higher-order biases
        c0, c2, c4, ck4: float, EFT counterterms
        aP:              float, constant deviation from Poisson shot-noise
        e0k2, e2k2:      float, k**2 corrections to the shot-noise
        Psn:             float, Poisson shot-noise
        f:               None or float, growth rate
        """
        P0 = self.Plgg_template(0, b1=b1, b2=b2, bG2=bG2, bG3=bG3, c0=c0, c2=c2,
                                c4=c4, ck4=ck4, aP=aP, e0k2=e0k2, e2k2=e2k2,
                                Psn=Psn, f=f)
        P2 = self.Plgg_template(2, b1=b1, b2=b2, bG2=bG2, bG3=bG3, c0=c0, c2=c2,
                                c4=c4, ck4=ck4, aP=aP, e0k2=e0k2, e2k2=e2k2,
                                Psn=Psn, f=f)
        P4 = self.Plgg_template(4, b1=b1, b2=b2, bG2=bG2, bG3=bG3, c0=c0, c2=c2,
                                c4=c4, ck4=ck4, aP=aP, e0k2=e0k2, e2k2=e2k2,
                                Psn=Psn, f=f)

        return 10.*(P0**2 + 4.*P0*(P2+P4)/7. + 6.*P2**2/7. + 24.*P2*P4/77. + 1789.*P4**2/9009)/self.Nk

#-------------------------------------------------------------------------------

    def varP44gg(self, b1=1, b2=0, bG2=0, bG3=0, c0=0, c2=0, c4=0, ck4=0, aP=0,
                 e0k2=0, e2k2=0, Psn=0, f=None):
        """
        Computes the Gaussian variance of the hexadecapole of the power spectrum
        given the full set of bias and stochastic parameters

        b1:              float, linear bias
        b2, bG2, bG3:    float, higher-order biases
        c0, c2, c4, ck4: float, EFT counterterms
        aP:              float, constant deviation from Poisson shot-noise
        e0k2, e2k2:      float, k**2 corrections to the shot-noise
        Psn:             float, Poisson shot-noise
        f:               None or float, growth rate
        """
        P0 = self.Plgg_template(0, b1=b1, b2=b2, bG2=bG2, bG3=bG3, c0=c0, c2=c2,
                                c4=c4, ck4=ck4, aP=aP, e0k2=e0k2, e2k2=e2k2,
                                Psn=Psn, f=f)
        P2 = self.Plgg_template(2, b1=b1, b2=b2, bG2=bG2, bG3=bG3, c0=c0, c2=c2,
                                c4=c4, ck4=ck4, aP=aP, e0k2=e0k2, e2k2=e2k2,
                                Psn=Psn, f=f)
        P4 = self.Plgg_template(4, b1=b1, b2=b2, bG2=bG2, bG3=bG3, c0=c0, c2=c2,
                                c4=c4, ck4=ck4, aP=aP, e0k2=e0k2, e2k2=e2k2,
                                Psn=Psn, f=f)

        return 162.*(P0**2/9. + 40.*P0*P2/693. +36.*P0*P4/1001. + 1789.*P2**2/45045. + 40.*P2*P4/1001. + 529.*P4**2/17017)/self.Nk

#-------------------------------------------------------------------------------

    def cov_ll_gg(self, ls, b1=1, b2=0, bG2=0, bG3=0, c0=0, c2=0, c4=0, ck4=0,
                  aP=0, e0k2=0, e2k2=0, Psn=0, f=None):
        """
        Computes the Gaussian covariance, including cross-correlations, of the
        specified multipoles of the power spectrum given the full set of bias
        and stochastic parameters

        ls:              int, degrees of the considered multipoles
        b1:              float, linear bias
        b2, bG2, bG3:    float, higher-order biases
        c0, c2, c4, ck4: float, EFT counterterms
        aP:              float, constant deviation from Poisson shot-noise
        e0k2, e2k2:      float, k**2 corrections to the shot-noise
        Psn:             float, Poisson shot-noise
        f:               None or float, growth rate
        """
        P0 = self.Plgg_template(0, b1=b1, b2=b2, bG2=bG2, bG3=bG3, c0=c0, c2=c2,
                                c4=c4, ck4=ck4, aP=aP, e0k2=e0k2, e2k2=e2k2,
                                Psn=Psn, f=f)
        P2 = self.Plgg_template(2, b1=b1, b2=b2, bG2=bG2, bG3=bG3, c0=c0, c2=c2,
                                c4=c4, ck4=ck4, aP=aP, e0k2=e0k2, e2k2=e2k2,
                                Psn=Psn, f=f)
        P4 = self.Plgg_template(4, b1=b1, b2=b2, bG2=bG2, bG3=bG3, c0=c0, c2=c2,
                                c4=c4, ck4=ck4, aP=aP, e0k2=e0k2, e2k2=e2k2,
                                Psn=Psn, f=f)

        blocks = [[None for j in ls] for i in ls]

        li = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        lj = [0, 1, 2, 0, 1, 2, 0, 1, 2]

        for l1 in ls:
            for l2 in ls:
                C = [0,0,0,0,0,0,0,0,0]
                I, J = int(l1/2), int(l2/2)
                for i in range(9):
                    la, lb = 2*li[i], 2*lj[i]

                    lmin = max(abs(l1-la), abs(l2-lb))
                    lmax = min(l1+la, l2+lb)

                    for l in range(lmin, lmax+1):
                        C[i] += 2.*(2.*l+1)*wigner_3j(l1, la, l, 0, 0, 0)**2 * wigner_3j(l2, lb, l, 0, 0, 0)**2

                vP = np.array((2.*l1+1.)*(2.*l2+1.)*(C[0]*P0*P0 + (C[1]+C[3])*P0*P2 + (C[2]+C[6])*P0*P4 + C[4]*P2*P2 + (C[5]+C[7])*P2*P4 + C[8]*P4*P4), dtype=np.float32)/self.Nk

                blocks[I][J] = np.diag(vP)

        Blocks = np.array(blocks)
        return Blocks
