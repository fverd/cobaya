import numpy as np
from scipy.signal import fftconvolve
from math import pi
try:    import fastpt.FASTPT_simple as fpt
except:    import FASTPT_simple as fpt

class FASTPTPlus(fpt.FASTPT):
    """FASTPTPlus

    Inherits from fastpt.FASTPT. Implements methods to compute the
    loop corrections for the redshift space galaxy power spectrum.
    """

    def Pkmu_22_one_loop_terms(self, K, P, P_window=None, C_window=None):
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
        `(P_dd_13 + P_dd_22),Pb1b2,(Pb1g2mc + Pb1g2pr),Pb2b2,Pb2g2,Pg2g2,Pb1g3`
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

        # return np.array([(P_dd_13 + P_dd_22), Pb1b2, Pb1g2mc,
        #               Pb2b2, Pb2g2, Pg2g2, Pb1g3])
        return np.array([(P_dd_13 + P_dd_22), Pb1b2, (Pb1g2mc + Pb1g2pr),
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
        n   = np.arange(-N+1,N )
        dL  = np.log(k[1])-np.log(k[0])
        s   = n*dL
        cut = 7
        high_s = s[s > cut]
        low_s  = s[s < -cut]
        mid_high_s = s[(s <= cut)  & (s > 0)]
        mid_low_s  = s[(s >= -cut) & (s < 0)]

        Z=lambda r : (12./r**2+10.+100.*r**2-42.*r**4+3./r**3*(r**2-1.)**3*(7*r**2+2.)*np.log((r+1.)/np.absolute(r-1.)))*r
        Z_low=lambda r : (352/5.+96/.5/r**2-160/21./r**4-526/105./r**6+236/35./r**8)*r
        Z_high=lambda r: (928/5.*r**2-4512/35.*r**4+416/21.*r**6+356/105.*r**8)*r

        f_mid_low  = Z(np.exp(-mid_low_s))
        f_mid_high = Z(np.exp(-mid_high_s))
        f_high     = Z_high(np.exp(-high_s))
        f_low      = Z_low(np.exp(-low_s))

        f = np.hstack((f_low,f_mid_low,80,f_mid_high,f_high))
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
        K: array of floats, Fourier wavenumbers, np.log-spaced
        P: array of floats, power spectrum

        Returns
        -------
        Pb1g3_bar: array, P_b1bG3 contribution to the galaxy power spectrum
        """

        N   = k.size
        n   = np.arange(-N+1,N)
        dL  = np.log(k[1])-np.log(k[0])
        s   = n * dL
        cut = 7
        high_s     = s[s>cut]
        low_s      = s[s<-cut]
        mid_high_s = s[(s<=cut)  & (s>0)]
        mid_low_s  = s[(s>=-cut) & (s<0)]

        Zb1g3      = lambda r: r*(-12./r**2+44.+44.*r**2-12*r**4+6./r**3*(r**2-1.)**4*np.log((r+1.)/np.absolute(r-1.)))
        Zb1g3_low  = lambda r: r*(512/5.-1536/35./r**2+512/105./r**4+512/1155./r**6 +512/5005./r**8)
        Zb1g3_high = lambda r: r*(512/5.*r**2-1536/35.*r**4+512/105.*r**6+512/1155.*r**8)

        fb1g3_mid_low  = Zb1g3(np.exp(-mid_low_s))
        fb1g3_mid_high = Zb1g3(np.exp(-mid_high_s))
        fb1g3_high     = Zb1g3_high(np.exp(-high_s))
        fb1g3_low      = Zb1g3_low(np.exp(-low_s))

        fb1g3 = np.hstack((fb1g3_low,fb1g3_mid_low,64,fb1g3_mid_high,fb1g3_high))
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
        n = np.arange(-N+1,N)
        dL = np.log(k[1])-np.log(k[0])
        s = n*dL
        cut = 7
        high_s = s[s>cut]
        low_s  = s[s<-cut]
        mid_high_s = s[(s<=cut) & (s>0)]
        mid_low_s = s[(s>=-cut) & (s<0)]

        Z13tt = lambda r: -1.5*(-24./r**2 + 52. - 8.*r**2 + 12.*r**4 - 6.*(r**2-1.)**3*(r**2+2.)/r**3*np.log((r+1.)/np.absolute(r-1.))) * r
        Z13tt_low = lambda r: r*( -672./5. + 3744./35./r**2 - 608./35./r**4 - 160./77./r**6 - 2976./5005./r**8)
        Z13tt_high = lambda r: r*(-96./5.*r**2 - 288./7.*r**4 + 352./35.*r**6 + 544./385.*r**8)

        f13tt_mid_low = Z13tt(np.exp(-mid_low_s))
        f13tt_mid_high= Z13tt(np.exp(-mid_high_s))
        f13tt_high    = Z13tt_high(np.exp(-high_s))
        f13tt_low     = Z13tt_low(np.exp(-low_s))

        f13tt = np.hstack((f13tt_low,f13tt_mid_low,-48,f13tt_mid_high,f13tt_high))
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
        n = np.arange(-N+1,N)
        dL = np.log(k[1])-np.log(k[0])
        s = n*dL
        cut = 7
        high_s = s[s>cut]
        low_s  = s[s<-cut]
        mid_high_s = s[(s<=cut) & (s>0)]
        mid_low_s = s[(s>=-cut) & (s<0)]

        Z13 = lambda r: ( 36. + 96.*r**2 - 36.*r**4 + 18.*(r**2-1.)**3/r*np.log((r+1.)/np.absolute(r-1.)) )*r
        Z13_low = lambda r: r*( 576./5. - 576./35./r**2 - 64./35./r**4 - 192./385./r**6 - 192./1001./r**8)
        Z13_high = lambda r: r*(192.*r**2 - 576./5.*r**4 + 576./35.*r**6 + 64./35.*r**8)

        f13_mid_low = Z13(np.exp(-mid_low_s))
        f13_mid_high= Z13(np.exp(-mid_high_s))
        f13_high    = Z13_high(np.exp(-high_s))
        f13_low     = Z13_low(np.exp(-low_s))

        f13 = np.hstack((f13_low,f13_mid_low,96,f13_mid_high,f13_high))
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
        n = np.arange(-N+1,N)
        dL = np.log(k[1])-np.log(k[0])
        s = n*dL
        cut = 7
        high_s = s[s>cut]
        low_s  = s[s<-cut]
        mid_high_s = s[(s<=cut) & (s>0)]
        mid_low_s = s[(s>=-cut) & (s<0)]

        Z13 = lambda r: ( 36./r**2 + 12. + 252.*r**2 - 108.*r**4 + 18.*(r**2-1.)**3*(1.+3.*r**2)/r**3 * np.log((r+1.)/np.absolute(r-1.)))*r
        Z13_low = lambda r: r*( 768./5. + 2304./35./r**2 - 768./35./r**4 - 256./77./r**6 - 768./715./r**8)
        Z13_high = lambda r: r*(2304./5.*r**2 - 2304./7.*r**4 + 256./5.*r**6 + 2304./385.*r**8)

        f13_mid_low = Z13(np.exp(-mid_low_s))
        f13_mid_high= Z13(np.exp(-mid_high_s))
        f13_high    = Z13_high(np.exp(-high_s))
        f13_low     = Z13_low(np.exp(-low_s))

        f13 = np.hstack((f13_low,f13_mid_low,192,f13_mid_high,f13_high))
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
