from pandas import read_csv
import numpy as np
import pickle
from .theoryPBJ import *
from .templatesPBJ import *
from .likelihoodPBJ import *
from .binningPBJ import *
from collections import OrderedDict
from scipy.special import legendre
from scipy.interpolate import InterpolatedUnivariateSpline
import sys
from .fptplus import FASTPTPlus
from .tools import bispectrum_functions as bispfunc
from .tools import param_handler as parhandler
from .tools import prior_handler as priorhandler
from .tools import data_handler as datahandler

if sys.version_info[0] == 2:
    range = xrange

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

class Pbj(PBJtheory, PBJtemplates, PBJlikelihood, PBJsampler):

    def __init__(self, Dict):
        """
        Minimal initialization requires the definition of a dictionary including
        only the cosmological parameters and the redshift.
        Additionally, an external power spectrum can be specified, with a file
        containing only two columns, k and PL(k). However, when an external file
        is available, the cosmology and the redshift must be anyway specified.
        A boolean flag for infrared resummation may be specified (default True).
        If the cosmology is not specified (you want to run chains to constrain
        the cosmology), some quantities will be initialized, but theoretical
        predictions will not be computed in the initialization
        """
        # Starting with minimal initialization
        self.Dict = Dict

        # theory
        try: self.linear = Dict['theory']['linear']
        except: self.linear = 'camb'
        print("\033[1;32m[info] \033[00m"+\
              "The linear power spectrum will be computed with "+self.linear)
        # Load the linear emulator
        if "baccoemu" in sys.modules:
            self.emulator = baccoemu.Matter_powerspectrum(linear=True,
                                                          smeared_bao=False,
                                                          nonlinear_boost=False,
                                                          baryonic_boost=False,
                                                          compute_sigma8=False,
                                                          verbose=False)

        try: self.IRres      = Dict['theory']['IRresum']
        except: self.IRres   = True
        try: self.IRresum_kind = Dict['theory']['IRresum_kind']
        except: self.IRresum_kind = 'EH'
        print("\033[1;32m[info] \033[00m"+"Infrared resummation: "+\
              str(self.IRres)+", kind: "+self.IRresum_kind)

        try: self.do_redshift_rescaling = Dict['theory']['do_redshift_rescaling']
        except: self.do_redshift_rescaling = True

        # AP
        try: self.do_AP = Dict['AP']['do_AP']
        except: self.do_AP = False
        print("\033[1;32m[info] \033[00m"+"Alcock-Paczynski distortions: "+\
              str(self.do_AP))

        # misc
        try: self.flat = Dict['FlatCosmology']
        except: self.flat = True

        self.mu = linspace(-1, 1, 101).reshape(1, 101)

        # Convolution
        if "window" in self.Dict:
            if self.Dict["window"]["convolve"] is not None:
                with open(self.Dict["window"]["file"], "rb") as infile:
                    self.w_dict = pickle.load(infile)
        else:
            self.Dict["window"] = {"convolve": None}

        if 'redshift_bins' in Dict:
            self.z_bins = np.asarray(Dict['redshift_bins'])

        # inputcosmo
        self.Inputcosmo = Dict['input_cosmology']
        if self.Inputcosmo != None:
            for entry in self.Inputcosmo.keys():
                setattr(self, entry, self.Inputcosmo[entry])
            if 'Mnu' not in self.Inputcosmo: self.Mnu = 0.
            if 'w0' not in self.Inputcosmo: self.w0 = -1.
            if 'wa' not in self.Inputcosmo: self.wa = 0.
            if 'tau' not in self.Inputcosmo: self.tau = 0.0952
            if 'Tcmb' not in self.Inputcosmo: self.Tcmb = 2.7255

            self.Omh2 = self.Obh2 + self.Och2 + self.Mnu/93.14
            self.Om = self.Omh2/self.h**2

            if self.flat:
                self.Ok = 0
                self.OL = 1. - self.Om
            else:
                self.Ok = 1. - self.Om - self.OL

            try:
                Pinput = np.loadtxt(self.Inputcosmo['inputfile'])
                print("\033[1;32m[info] \033[00m"+"Linear power spectrum loaded from: "+\
                      self.Inputcosmo['inputfile'])
                kL, PL = np.array(Pinput).T
            except:
                if self.do_redshift_rescaling: #SONO INVERTITI QUA?
                    kL, PL = self.linear_power_spectrum(linear='bacco', redshift=0)
                else:
                    kL, PL = self.linear_power_spectrum(linear='bacco',
                                                        redshift=self.z)

            self.kL = kL
            self.PL = PL


            self.Dz, _ = self.growth_factor(self.z)
            self.f  = self.growth_rate(self.z)

            self.fastpt = FASTPTPlus(self.kL, -2, param_mat=None, low_extrap=-6,
                                     high_extrap=5, n_pad=1000)

            self.tns = tns(self.kL, -2, param_mat=None, to_do=['RSD'],
                           low_extrap=-6, high_extrap=5, n_pad=1000)

            self.Pktags = ['LO', 'NLO', 'k2LO', 'b1b2', 'b1bG2', 'b2b2',
                           'b2bG2', 'bG2bG2', 'b1bG3', 'k2']

            self.Bktags = ['b13', 'b12b2', 'b12bG2', 'b12a1']

            self.Pltags = ['b1b1', 'fb1', 'f2', 'b1b2', 'b1bG2', 'b2b2',
                           'b2bG2', 'bG2bG2', 'fb2', 'fbG2', 'f2b12', 'fb12',
                           'fb1b2', 'fb1bG2', 'f2b1', 'f2b2', 'f2bG2', 'f4',
                           'f3', 'f3b1', 'b1bG3', 'fbG3', 'c0', 'fc2', 'f2c4',
                           'f4b12ck4','f5b1ck4','f6ck4','e0k2', 'e2k2', 'aP']

            if self.do_AP:
                self.Bltags = ['Bb13_0', 'Bb12b2_0', 'Bb12bG2_0', 'Bfb12_0',
                               'Bfb13_0', 'Bf2b12_0', 'Bfb1b2_0', 'Bfb1bG2_0',
                               'Bf2b1_0', 'Bf3b1_0', 'Bf2b2_0', 'Bf2bG2_0',
                               'Bf3_0', 'Bf4_0', 'Bb12a1_0', 'Bfb1a1_0', 'Bf2a1_0',
                               'Bb13_perp', 'Bb12b2_perp', 'Bb12bG2_perp',
                               'Bfb12_perp', 'Bfb13_perp', 'Bf2b12_perp',
                               'Bfb1b2_perp', 'Bfb1bG2_perp', 'Bf2b1_perp',
                               'Bf3b1_perp', 'Bf2b2_perp', 'Bf2bG2_perp',
                               'Bf3_perp', 'Bf4_perp', 'Bb12a1_perp', 'Bfb1a1_perp',
                               'Bf2a1_perp', 'Bb13_dif', 'Bb12b2_dif', 'Bb12bG2_dif',
                               'Bfb12_dif', 'Bfb13_dif', 'Bf2b12_dif', 'Bfb1b2_dif',
                               'Bfb1bG2_dif', 'Bf2b1_dif', 'Bf3b1_dif', 'Bf2b2_dif',
                               'Bf2bG2_dif', 'Bf3_dif', 'Bf4_dif', 'Bb12a1_dif',
                               'Bfb1a1_dif', 'Bf2a1_dif']
            else:
                self.Bltags = ['b13', 'b12b2', 'b12bG2', 'fb12', 'fb13', 'f2b12',
                               'fb1b2', 'fb1bG2', 'f2b1', 'f3b1', 'f2b2', 'f2bG2',
                               'f3', 'f4', 'b12a1', 'fb1a1', 'f2a1']


            P_REAL = self.PowerfuncIR_real(self.z, self.do_redshift_rescaling,
                                           kind=self.IRresum_kind)
            self.Pk = OrderedDict(zip(self.Pktags,P_REAL))

            if self.do_redshift_rescaling:
                self._Pgg_kmu_terms(kind=self.IRresum_kind, do_AP=self.do_AP,
                                    window_convolution=self.Dict["window"]["convolve"])
            else:
                self._Pgg_kmu_terms(redshift=self.z, kind=self.IRresum_kind,
                                    do_AP=self.do_AP,
                                    window_convolution=self.Dict["window"]["convolve"])

            P0_RSD, P2_RSD, P4_RSD = self.PowerfuncIR_RSD(
                self.z, self.do_redshift_rescaling)
            P0k = OrderedDict(zip(self.Pltags, P0_RSD))
            P2k = OrderedDict(zip(self.Pltags, P2_RSD))
            P4k = OrderedDict(zip(self.Pltags, P4_RSD))

            self.P_ell_k = {0: P0k, 2: P2k, 4: P4k}

        else:
            self.kL = np.logspace(-4, np.log10(198.), num=1000, endpoint=True)
            self.fastpt = FASTPTPlus(self.kL, -2, low_extrap=-6, high_extrap=5,
                                     n_pad=10000)

        if self.do_AP==True:
            self.AP_as_nuisance = Dict.get('AP', {}).get('AP_as_nuisance', False)

            if self.AP_as_nuisance==False:
                try:
                    self.FiducialCosmo = Dict['AP']['fiducial_cosmology']
                    if hasattr(self, 'z_bins'):
                        self._set_fiducials_for_AP(self.z_bins, self.FiducialCosmo)
                    else:
                        self._set_fiducials_for_AP(self.z, self.FiducialCosmo)
                except:
                    raise KeyError("do_AP = True, but FiducialCosmo dictionary is not specified in parameter file")

#-------------------------------------------------------------------------------

    def initialise_grid(self):
        '''
        Try to initialize box and grid properties, then prepare for binning and
        compute templates for all correlation functions with the chosen binning
        scheme.
        '''

        self.Boxsize  = self.Dict['boxsize']
        # binning
        self.dk       = self.Dict['binning']['grid']['dk']
        self.cf       = self.Dict['binning']['grid']['cf']
        self.nbinsP   = self.Dict['binning']['grid']['nbinsP']
        self.nbinsB   = self.Dict['binning']['grid']['nbinsB']

        try: self.InputTheory = self.Dict['input_theory_templates']
        except: self.InputTheory = {'Pk': None}

        self.SNLeaking = self.Dict.get('binning', {}).get('SNLeaking', False)

        self.kf = 2.*np.pi/self.Boxsize
        print("\033[1;32m[info] \033[00m"+"Boxsize: " + str(self.Boxsize) + \
              ", grid properties: " + str(self.Dict['binning']['grid']))

        self.binning = self.Dict.get('binning', {}).get('type', 'Effective')
        if self.Dict["window"]["convolve"] is not None:
            print("\033[1;32m[info] \033[00m"+"Using binning from window.")
        else:
            print("\033[1;32m[info] \033[00m"+"Using "+self.binning+" binning.")

        NPmax = (self.nbinsP-0.5)*self.dk + self.cf
        if NPmax > 512.5:
            self.nbinsP = int((NPmax - self.cf)/self.dk + 0.5)
            self.NPmax = (self.nbinsP-0.5)*self.dk + self.cf
            print("\033[1;31m[WARNING] \033[00m"+\
                  "Power spectrum exceeding maximum grid size.\nSetting nbinsP = " + \
                  str(self.nbinsP))
        else: self.NPmax = NPmax

        # Convolution
        if self.Dict["window"]["convolve"] is not None:
            # Check that k and k' of the window mixing matrix work fine
            self.kWindow = np.arange(self.cf, self.NPmax, self.dk) * self.kf
            if not np.all(((self.w_dict['k1'] - self.kWindow) / self.kWindow) < 1e-5):
                raise ValueError("Window k range does not match binning k range.")
            if self.Dict["window"]["grid"]["epsilonkpConv"] is not None:
                self.Dict["window"]["grid"]["kpmaxConv"] = (
                    self.NPmax + self.Dict["window"]["grid"]["epsilonkpConv"]
                )
            elif self.Dict["window"]["grid"]["kpmaxConv"] < self.NPmax:
                raise ValueError("Maximum convolution k' must be larger than maximum binning k = {}.".format(self.NPmax))
            # Trim window mixing matrix to k'_max
            self.kpW = list()
            for lp in self.w_dict['k2']:
                if self.w_dict['k2'][lp][0] <= 1e-4: print("\033[1;31m[WARNING] \033[00m"+\
                                                           "Array of k' of the mixing matrix (for multipole " + lp + ") cut at k'=10^(-4) h/Mc")
                mask = (self.w_dict['k2'][lp] < self.Dict["window"]["grid"]["kpmaxConv"])
                self.kpW.append(self.w_dict['k2'][lp][mask])
                for l in (0, 2, 4):
                    self.w_dict['wc'][l, lp] = self.w_dict['wc'][l, lp][:, mask]
            # Stack mixing matrix
            self.window = np.block(
                [[self.w_dict['wc'][l, lp] for lp in self.w_dict['k2'].keys()] for l in (0, 2, 4)]
            )
            print("\033[1;32m[info] \033[00m"+"Loaded window mixing matrix.")

        NBmax = (self.nbinsB-0.5)*self.dk + self.cf
        if NBmax > 64.5:
            self.nbinsB = int((NBmax -self.cf)/self.dk + 0.5)
            self.NBmax = (self.nbinsB-0.5)*self.dk + self.cf
            print("\033[1;31m[WARNING] \033[00m"+\
                  "Bispectrum exceeding maximum grid size.\nSetting nbinsB = " + \
                  str(self.nbinsB))
        else: self.NBmax = NBmax

        power_binning_scheme = BinningScheme(self.cf, self.dk, self.nbinsP,
                                             right_open=True)
        bisp_binning_scheme = BinningScheme(self.cf, self.dk, self.nbinsB,
                                           right_open=True)

        power_linear_bins = Bins.linear_bins(power_binning_scheme)

        self.kP = np.array(power_linear_bins.centers())

        self.real_space_power_binner = PowerBinner.new(
            power_linear_bins, self.binning, "real", None)
        self.redshift_space_power_binner = PowerBinner.new(
            power_linear_bins, self.binning, "redshift", multipoles=[0, 2, 4])
        self.bispectrum_binner = BispectrumBinner.new(bisp_binning_scheme,
                                                      open_triangles=True)

        self.Ptemplates = []
        for pk in self.Pk.values():
            F = InterpolatedUnivariateSpline(self.kL, pk, k=3)
            self.Ptemplates.append(
                self.real_space_power_binner.bin_function(F, x_scale=self.kf))
        self.Ptemplates.append(np.ones(self.nbinsP))

        self.P0templates = []
        self.P2templates = []
        self.P4templates = []

        if self.Dict["window"]["convolve"] is not None and self.Dict["likelihood"]["cosmology"] == "fixed":
            self.P0templates, self.P2templates, self.P4templates = (
                self.P_ell_conv_fixed_cosmo(self.z, self.do_redshift_rescaling)
            )
        else:
            for F0, F2, F4 in zip(self.P_ell_k[0].values(),
                                  self.P_ell_k[2].values(),
                                  self.P_ell_k[4].values()):
                iF0k = InterpolatedUnivariateSpline(self.kL, F0, k=3)
                iF2k = InterpolatedUnivariateSpline(self.kL, F2, k=3)
                iF4k = InterpolatedUnivariateSpline(self.kL, F4, k=3)

                iFkmu = lambda q, mu: iF0k(q) + (1.5*mu*mu - 0.5) * iF2k(q) + (35 * mu **4 - 30 * mu**2 + 3)/8. * iF4k(q)

                ieff0, ieff2, ieff4 = self.redshift_space_power_binner.bin_function(
                    iFkmu, x_scale=self.kf)
                self.P0templates.append(ieff0)
                self.P2templates.append(ieff2)
                self.P4templates.append(ieff4)

        self.N_T = len(self.bispectrum_binner.bins.bins)
        self.kB1E = np.array([t.sup for t in
                              self.bispectrum_binner.effective_triangles])*self.kf
        self.kB2E = np.array([t.med for t in
                              self.bispectrum_binner.effective_triangles])*self.kf
        self.kB3E = np.array([t.inf for t in
                              self.bispectrum_binner.effective_triangles])*self.kf
        self.NTk = self.bispectrum_binner.counts

        tri_bins = np.array([[b.bin1.center(), b.bin2.center(), b.bin3.center()]
                             for b in self.bispectrum_binner.bins.bins]).T
        self.kB1, self.kB2, self.kB3 = tri_bins

        real_avg_binner = PowerBinner.new(power_linear_bins, "average", "real")
        self.kPE = real_avg_binner.bin_function(lambda q: q, x_scale=self.kf)
        self.kPE2 = real_avg_binner.bin_function(lambda q: q**2, x_scale=self.kf)

        # Check that k and k' of the P window mixing matrix work fine
        if self.Dict["window"]["convolve"] is not None:
            if not np.all(
                    ((self.w_dict['k1'] - self.kPE) / self.kPE) < 1e-5):
                raise ValueError("Window k range does not match binning k range.")

        k2_0, k2_2, k2_4 = self.redshift_space_power_binner.bin_function(
            lambda q, mu: q**2, x_scale=self.kf)
        mu2k2_0, mu2k2_2, mu2k2_4 = self.redshift_space_power_binner.bin_function(
            lambda q, mu: q**2*mu**2, x_scale=self.kf)
        sn_0, sn_2, sn_4 = self.redshift_space_power_binner.bin_function(
            lambda q, mu: 1.+q*mu*0, x_scale=self.kf)

        if self.SNLeaking:
            self.sn_rsd_p0 = [k2_0, mu2k2_0, sn_0]
            self.sn_rsd_p2 = [k2_2, mu2k2_2, sn_2]
            self.sn_rsd_p4 = [k2_4, mu2k2_4, sn_4]
        else:
            self.sn_rsd_p0 = [k2_0, sn_0]
            self.sn_rsd_p2 = [mu2k2_2]
            self.sn_rsd_p4 = []

        for sn in self.sn_rsd_p0: self.P0templates.append(sn)
        for sn in self.sn_rsd_p2: self.P2templates.append(sn)
        for sn in self.sn_rsd_p4: self.P4templates.append(sn)

        self.K, self.F, self.G, self.S, self.dF, self.dG, self.dS, self.KK, self.dKK = \
            bispfunc.Bl_templates_k(self.kB1E, self.kB2E, self.kB3E)
        self.Is0 = bispfunc.Bl_templates_angles(0, self.kB1E, self.kB2E, self.kB3E)
        self.Is2 = bispfunc.Bl_templates_angles(2, self.kB1E, self.kB2E, self.kB3E)
        self.Is4 = bispfunc.Bl_templates_angles(4, self.kB1E, self.kB2E, self.kB3E)

        if self.binning == 'Effective':
            self.Btemplates = self.Bggg_eff()
            self.B0templates, self.B2templates, self.B4templates = self.Blgg_eff()

        elif self.binning == 'Expansion':
            print("\033[1;31m[WARNING] \033[00m"+"Expansion binning implemented only for power spectrum. Bispectrum will be evaluated with the Effective method, or will be uploaded from input theory files.")
            self.Btemplates = self.Bggg_eff()
            self.B0templates, self.B2templates, self.B4templates = self.Blgg_eff()

        elif self.binning == 'Average':
            print("\033[1;31m[WARNING] \033[00m"+"Full bin-average implemented only for power spectrum. All others will be evaluated with the Effective method, or will be uploaded from input theory files.")
            self.Btemplates = self.Bggg_eff()
            self.B0templates, self.B2templates, self.B4templates = self.Blgg_eff()

        for k in self.InputTheory:
            if (self.InputTheory[k] != None) & (self.binning == 'Average'):
                if k[0] == 'P':
                    ONES = np.ones((self.nbinsP,1))
                elif k[0] == 'B' and k[1] != 'A':
                    ONES = np.ones((self.N_T,1))

                if k == 'Pk':
                    self.Ptemplates  = np.hstack((np.array(
                        read_csv(self.InputTheory[k], sep=r"\s+",
                                 header=None)),ONES)).T
                elif k == 'Bk':
                    self.Btemplates  = np.hstack((np.array(
                        read_csv(self.InputTheory[k], sep=r"\s+",
                                 header=None)),ONES)).T
                elif k == 'P0':
                    self.P0templates = (np.array(read_csv(
                        self.InputTheory[k], sep=r"\s+", header=None))).T
                elif k == 'P2':
                    self.P2templates = (np.array(read_csv(
                        self.InputTheory[k], sep=r"\s+", header=None))).T
                elif k == 'P4':
                    self.P4templates = (np.array(read_csv(
                        self.InputTheory[k], sep=r"\s+", header=None))).T
                elif k == 'B0':
                    self.B0templates = np.hstack((np.array(
                        read_csv(self.InputTheory[k], sep=r"\s+",
                                 header=None)),ONES)).T
                elif k == 'B2':
                    self.B2templates = np.hstack((np.array(
                        read_csv(self.InputTheory[k], sep=r"\s+",
                                 header=None)),ONES)).T
                elif k == 'B4':
                    self.B4templates = np.hstack((np.array(
                        read_csv(self.InputTheory[k], sep=r"\s+",
                                 header=None)),ONES)).T

                print("\033[1;32m[info] \033[00m"+"Theory files for",k," loaded")

            elif (self.InputTheory[k] != None) & (self.binning == 'Expansion'):
                if k[0] == 'B' and k[1] != 'A':
                    ONES = np.ones((self.N_T,1))

                if k == 'Bk':
                    self.Btemplates  = np.hstack((np.array(
                        read_csv(self.InputTheory[k], sep=r"\s+",
                                 header=None)),ONES)).T
                elif k == 'B0':
                    self.B0templates = np.hstack((np.array(
                        read_csv(self.InputTheory[k], sep=r"\s+",
                                 header=None)),ONES)).T
                elif k == 'B2':
                    self.B2templates = np.hstack((np.array(
                        read_csv(self.InputTheory[k], sep=r"\s+",
                                 header=None)),ONES)).T
                elif k == 'B4':
                    self.B4templates = np.hstack((np.array(
                        read_csv(self.InputTheory[k], sep=r"\s+",
                                 header=None)),ONES)).T
                print("\033[1;32m[info] \033[00m"+"Theory files for",k," loaded")

        PkTempl = OrderedDict(zip(self.Pktags+['1'], self.Ptemplates))
        P0Templ = OrderedDict(zip(self.Pltags,       self.P0templates))
        P2Templ = OrderedDict(zip(self.Pltags,       self.P2templates))
        P4Templ = OrderedDict(zip(self.Pltags,       self.P4templates))

        BkTempl = OrderedDict(zip(self.Bktags+['1'], self.Btemplates))
        B0Templ = OrderedDict(zip(self.Bltags+['1'], self.B0templates))
        B2Templ = OrderedDict(zip(self.Bltags,       self.B2templates))
        B4Templ = OrderedDict(zip(self.Bltags,       self.B4templates))

        self.P_ell_Templ = {'Pk': PkTempl, 'P0': P0Templ, 'P2': P2Templ,
                            'P4': P4Templ, 'PX': PkTempl, 'Pm': PkTempl}
        self.B_ell_Templ = {'Bk': BkTempl, 'B0': B0Templ, 'B2': B2Templ,
                            'B4': B4Templ}

#-------------------------------------------------------------------------------
# Try to upload data vectors and the covariance matrix

    def initialise_data(self):
        """
        Loads the data files and covariance
        """
        self.Obs        = self.Dict['likelihood']['observables']
        print("\033[1;32m[info] \033[00m"+"Observables: "+str(self.Obs))

        if 'Pk' in self.Obs:
            self.do_preal = True
        if 'Bk' in self.Obs:
            self.do_breal = True
        if any(map(lambda v: v in self.Obs, ['P0', 'P2', 'P4', 'Q0'])):
            self.do_pell = True
        if any(map(lambda v: v in self.Obs, ['BAO_P0', 'BAO_P2', 'BAO_P4'])):
            self.do_pell_bao = True
        if any(map(lambda v: v in self.Obs, ['B0', 'B2', 'B4'])):
            self.do_bell = True


        self.covtype    = self.Dict['covariance']['type']
        self.covfile    = self.Dict['covariance']['file']
        if isinstance(self.covfile, list) and len(self.covfile)==1:
            self.covfile = self.covfile[0]

        print("\033[1;32m[info] \033[00m"+"Using "+self.covtype)

        self.datafiles  = self.Dict['data_files']
        self.SN         = self.Dict['SN']
        if isinstance(self.SN, list) and len(self.SN)==1:
            self.SN = self.SN[0]

        self.Pi         = self.Dict['PiConvention'] \
            if 'PiConvention' in self.Dict else 'CMB'
        if self.Pi == 'CMB':
            self.Twopi3 = 1.
        elif self.Pi == 'LSS':
            self.Twopi3 = (2.*np.pi)**3

        self.NSets = self.Dict['NSets'] if 'Nsets' in self.Dict else None
        self.MeanBool = self.Dict['Mean'] if 'Mean' in self.Dict else None

        # data
        MeanSets, self.NSets, self.DataDict = datahandler.load_data(self.datafiles,
                                                                    self.Obs,
                                                                    self.Twopi3,
                                                                    self.nbinsP,
                                                                    self.NSets,
                                                                    self.MeanBool)

        # Shot noise
        if isinstance(self.SN, float):
            self.Psn  = self.SN
            self.Psn2 = self.SN**2
            self.Psnarr  = np.full(self.NSets, self.Psn)
            self.Psn2arr = np.full(self.NSets, self.Psn2)

        elif isinstance(self.SN, list):
            if len(self.SN) != len(self.z_bins):
                raise ValueError("Number of redshift bins and SN values must match")
            self.Psn  = np.asarray(self.SN)
            self.Psn2 = np.asarray(self.SN)**2
            self.Psnarr  = np.repeat(self.Psn, self.NSets).reshape(len(self.Psn),
                                                                   self.NSets)
            self.Psn2arr = np.repeat(self.Psn2, self.NSets).reshape(len(self.Psn),
                                                                    self.NSets)

        elif isinstance(self.SN, str):
            SN = np.array(read_csv(self.SN, sep=r"\s+", header=None))
            SN2 = SN*SN
            self.Psn  = SN.mean()
            self.Psn2 = SN2.mean()
            self.Psnarr  = np.squeeze(SN)
            self.Psn2arr = np.squeeze(SN2)

        else:
            self.Psn  = 0
            self.Psn2 = 0
            self.Psnarr  = np.zeros(self.NSets)
            self.Psn2arr = np.zeros(self.NSets)

        self.Psn     *= self.Twopi3
        self.Psn2    *= self.Twopi3**2
        self.Psnarr  *= self.Twopi3
        self.Psn2arr *= self.Twopi3**2

        # covariance
        if isinstance(self.covfile, list):
            if len(self.covfile) != len(self.z_bins):
                raise ValueError("Number of redshift bins and covariance files must match")

        self.Cov = datahandler.load_covariance(self.covfile, self.Obs,
                                               self.Twopi3, self.nbinsP,
                                               self.N_T, MeanSets,
                                               covtype = self.covtype)

#-------------------------------------------------------------------------------
# Initialise likelihood

    def initialise_likelihood(self):

        # Parameters
        if 'priors' in self.Dict['likelihood']:
            self.prior_dictionary = parhandler.read_file(
                self.Dict['likelihood']['priors'])
        else:
            self.prior_dictionary = parhandler.read_file("config_default/priors.yaml")

        # Create parameter dictionary and set initial points
        self.initial_param_dict = parhandler.parse_parameters(self.prior_dictionary)

        # Update initial_param_dict with derived parameters
        parhandler.compute_derived_parameters(self.prior_dictionary,
                                              self.initial_param_dict)

        # Extract list of strings for varied parameters
        self.varied_params = [key for key, value in self.prior_dictionary.items()
                              if value.get('type') == 'varied']

        PBJlikelihood.__init__(self, self.initial_param_dict)

        # Update prior_dictionary to include functions for the prior distributions
        priorhandler.add_prior_distribution_to_dict(self.varied_params,
                                                    self.prior_dictionary)

        self.alphas_dict = parhandler.construct_alphas_dict(self.do_AP)

        # Analytic marginalisation
        self.do_analytic_marg = self.Dict['likelihood'].get('do_analytic_marg', False)
        self.do_jeffreys_priors = self.Dict['likelihood'].get('do_jeffreys_priors', False)

        if self.do_analytic_marg or self.do_jeffreys_priors:
            # Check which params are included, and that priors have been set
            # correctly
            if self.Dict['likelihood']['model'] == 'model_BAO_analytic_marg':
                marg_params_all = ['a00', 'a01', 'a02', 'a03', 'a04', 'a05',
                                   'a20', 'a21', 'a22', 'a23', 'a24', 'a25',
                                   'a40', 'a41', 'a42', 'a43', 'a44', 'a45']
            else:
                marg_params_all = ['bG3', 'c0', 'c2', 'c4', 'ck4', 'aP', 'e0k2', 'e2k2']

            self.index_marg_param = []
            for key in marg_params_all:
                # # This takes care of params in the list that are set to 0
                if key in self.initial_param_dict.keys():
                    self.index_marg_param.append(marg_params_all.index(key))

                # This checks Gaussian priors have been imposed
                if self.initial_param_dict[key]['prior'] != 'gaussian':
                    raise ValueError(f"Analytic marginalisation requires priors for '{marg_params}' to be 'gaussian'. Found: '{self.initial_param_dict[key]['prior']}'")

            marg_params = [marg_params_all[i] for i in self.index_marg_param] #check for full shape

            # Then compute quantities to be added to the chi2
            self.prior_vector = np.array(
                [self.initial_param_dict[key]['prior_params']['mean']
                 for key in marg_params])

            if self.prior_vector.ndim == 1:
                prior_cov_mat = np.diag([
                    self.initial_param_dict[key]['prior_params']['sigma']**2
                    for key in marg_params])

                self.prior_cov_inv = np.linalg.inv(prior_cov_mat)

                self.prior_term_F0 = np.einsum(
                    'i,ij,j->', self.prior_vector, self.prior_cov_inv, self.prior_vector)
                self.prior_term_F1i = np.einsum('ij,j->i',
                                                self.prior_cov_inv, self.prior_vector)
            elif self.prior_vector.ndim == 2:
                prior_cov_mat = [np.diag([
                    self.initial_param_dict[key]['prior_params']['sigma'][i]**2
                    for key in marg_params]) for i in range(self.prior_vector.shape[1])]

                self.prior_cov_inv = np.linalg.inv(prior_cov_mat)

                self.prior_term_F0 = np.einsum('ik,kij,jk->k', self.prior_vector,
                                               self.prior_cov_inv, self.prior_vector)
                self.prior_term_F1i = np.einsum('kij,jk->ki',
                                                self.prior_cov_inv, self.prior_vector)


        # Sampler
        self.sampler = self.Dict['likelihood']['sampler']

        if self.sampler == 'emcee' and 'check_convergence' in self.Dict['likelihood']:
            self.CheckConvergence = self.Dict['likelihood']['check_convergence']
        else:
            self.CheckConvergence = False

        # Likelihood function
        self.likeF    = self.Dict['likelihood']['type']
        print("\033[1;32m[info] \033[00m"+"Using "+self.likeF+" likelihood.")

        if self.likeF != 'Gaussian':
            self.Nmocks = self.Dict['covariance']['NMocks']

        if self.likeF == 'Gaussian':
            self.lnlike = self.lnlike_gaussian
        elif self.likeF == 'Hartlap':
            self.lnlike = self.lnlike_hartlap
        elif self.likeF == 'Sellentin':
            self.lnlike = self.lnlike_sellentin
        else:
            raise NotImplementedError(f"Likelihood type {self.likeF} not supported.")


        # Automatic model likelihood setting
        if 'model' in self.Dict['likelihood'] and isinstance(self.Dict['likelihood']['model'], str):
            try:
                self.model_function = getattr(self,
                                              self.Dict['likelihood']['model'], None)
            except:
                raise NotImplementedError(r"Likelihood model function not implemented.")
        else:
            if self.Dict['likelihood']['cosmology'] == 'fixed':
                if self.do_AP or self.do_analytic_marg or self.do_jeffreys_priors:
                    self.model_function = self.model_fixed_cosmology_AP
                else:
                    self.model_function = self.model_fixed_cosmology

            elif self.Dict['likelihood']['cosmology'] == 'varied':
                if self.do_analytic_marg or self.do_jeffreys_priors:
                    self.model_function = self.model_varied_cosmology_analytic_marg
                else:
                    self.model_function = self.model_varied_cosmology

            if hasattr(self, 'z_bins'):
                self.model_function = self.model_varied_cosmology_analytic_marg_multiz

        print("\033[1;32m[info] \033[00m"+\
              f"Using {self.model_function.__name__} as likelihood model function")


        self.Resume = self.Dict['likelihood'].get('resume', False)

        if self.sampler == 'Metropolis':
            try:
                self.ParameterCov = self.Dict['ParameterCov']
                self.ParCov = np.loadtxt(self.ParameterCov)
            except:
                self.ParCov = np.eye(len(self.varied_params))

        # Output files
        try:
            self.kmax_string = self.Dict['output']['kmax_string']
            self.comment = self.Dict['output']['comment']
            self.outfolder = self.Dict['output']['folder']
        except:
            self.kmax_string = False
            self.comment = ''
            self.outfolder = './'

        if not os.path.exists(self.outfolder):
            # The folder does not exist, so create it
            os.makedirs(self.outfolder)
            print(f"Folder '{self.outfolder}' was created.")

#-------------------------------------------------------------------------------

    def initialise_full(self):
        '''
        Full initialization routine

        Calls three methods of the pbj class: initialise_grid(),
        initialise_data(), initialise_likelihood()
        '''
        self.initialise_grid()
        self.initialise_data()
        self.initialise_likelihood()
