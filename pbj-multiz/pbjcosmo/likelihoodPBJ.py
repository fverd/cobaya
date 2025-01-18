# This Python file uses the following encoding: utf-8
from collections import OrderedDict
from emcee import EnsembleSampler, moves
import getdist.plots
from math import sqrt, pi, e
import matplotlib.pyplot as plt
from multiprocessing import Pool
from nautilus import Prior, Sampler
from numpy import inf, log, isfinite, outer, array, hstack, vstack, ix_, \
    concatenate, einsum, diag, sum, c_, empty, all, abs, savetxt, arange, zeros
import numpy as np
from numpy.linalg import inv
from numpy.random import randn, uniform
import os
from pandas import read_csv
from scipy.integrate import simps
from scipy.linalg import block_diag
from scipy.special import legendre
import scipy.stats
from scipy.interpolate import InterpolatedUnivariateSpline
import sys
from typing import Optional, Union, List
import yaml
from .tools import param_handler as parhandler
from .tools import prior_handler as priorhandler
#try:
#    import pymultinest as pmn
#except ImportWarning as err:
#    print("\033[1;31m[WARNING]: \033[00m"+"PyMultinest not installed."+err)

# try:
#     import pocomc
# except ImportWarning as err:
#     print("\033[1;31m[WARNING]: \033[00m"+"pocomc not installed."+err)

# try:
#     import ultranest
# except ImportWarning as err:
#     print("\033[1;31m[WARNING]: \033[00m"+"ultranest not installed."+err)

plt.rc('text', usetex=True)
plt.rc('font', size=11)
plt.rc('xtick', direction='in')
plt.rc('ytick', direction='in')
os.environ["OMP_NUM_THREADS"] = '1'

class PBJlikelihood:

    def __init__(self, initial_param_dict):
        self.full_param_dict = {key: initial_param_dict.get(key)['value']
                                for key in initial_param_dict.keys()}

#-------------------------------------------------------------------------------

    def model_fixed_cosmology(self, params):
        # Update full param dictionary with new param values
        self.full_param_dict.update({key: value for key, value in
                                    zip(self.varied_params, params)})
        # Update derived parameters
        parhandler.update_derived_parameters(self.prior_dictionary,
                                             self.full_param_dict)

        # Create nuisance list with new values
        alpha = self.create_nuisance_list(**self.full_param_dict)

        return self.Chi2_fixCov_fixCosmo(alpha)


    def model_fixed_cosmology_AP(self, params):
        # Update full param dictionary with new param values
        self.full_param_dict.update({key: value for key, value in
                                    zip(self.varied_params, params)})
        # Update derived parameters
        parhandler.update_derived_parameters(self.prior_dictionary,
                                             self.full_param_dict)
        theory_dict = {}
        marg_dict = {}
        if hasattr(self, 'do_pell'):
            if self.do_analytic_marg or self.do_jeffreys_priors:
                Pell, Pell_marg = self.P_kmu_z_marg(self.z, self.do_redshift_rescaling,
                                                    AP_as_nuisance=self.do_AP,
                                                    kgrid=self.kPE, Psn=self.Psn,
                                                    **self.full_param_dict)

                theory_dict['P0'] = Pell[0][self.IdxP]
                theory_dict['P2'] = Pell[1][self.IdxP]
                theory_dict['P4'] = Pell[2][self.IdxP4]

                marg_dict['P0'] = np.asarray([Pell_marg[0+3*j,self.IdxP]
                                              for j in self.index_marg_param])
                marg_dict['P2'] = np.asarray([Pell_marg[1+3*j,self.IdxP]
                                              for j in self.index_marg_param])
                marg_dict['P4'] = np.asarray([Pell_marg[2+3*j,self.IdxP4]
                                              for j in self.index_marg_param])

            else:
                Pell_eff = self.P_kmu_z(self.z, self.do_redshift_rescaling,
                                        AP_as_nuisance=self.do_AP, kgrid=self.kPE,
                                        Psn=self.Psn, **self.full_param_dict)

                theory_dict['P0'] = Pell_eff[0][self.IdxP]
                theory_dict['P2'] = Pell_eff[1][self.IdxP]
                theory_dict['P4'] = Pell_eff[2][self.IdxP4]

        if hasattr(self, 'do_bell'):
            if self.do_analytic_marg:
                raise NotImplementedError("AM bispectrum not implemented yet")

            bias_list = [self.alphas_dict[obs](self.Psn,
                                               self.SNLeaking,
                                               **self.full_param_dict)
                         for obs in ['B0','B2','B4']]
            bispfunc = self.Bispfunc_RSD(
                self.z, self.kB1E, self.kB2E, self.kB3E,
                do_redshift_rescaling=self.do_redshift_rescaling)

            theory_dict['B0'] = einsum('i,ij->j',
                                       bias_list[0], bispfunc[0])[self.IdxB]

            if 'B2' in self.Obs:
                theory_dict['B2'] = einsum('i,ij->j',
                                           bias_list[1], bispfunc[1])[self.IdxB2]
            if 'B4' in self.Obs:
                theory_dict['B4'] = einsum('i,ij->j',
                                           bias_list[2], bispfunc[2])[self.IdxB4]

        TheoryVec = np.concatenate([theory_dict[obs] for obs in self.Obs])

        if self.do_analytic_marg or self.do_jeffreys_priors:
            MargVec = np.hstack([marg_dict[obs] for obs in self.Obs])
            deltaTD = TheoryVec - np.concatenate(self.CutDataVecs, axis=1)

            # F0 = np.einsum('ri, ij, rj -> r', deltaTD, self.invCov, deltaTD) +\
            #     self.prior_term_F0
            # F1i = -np.einsum('ri, ij, kj -> r', MargVec, self.invCov, deltaTD) +\
            #     self.prior_term_F1i
            # F2ij = np.einsum('ri, ij, mj -> rm', MargVec, self.invCov, MargVec) +\
            #     self.prior_cov_inv

            # chi2 = F0 + np.log(np.linalg.det(F2ij)) - \
            #     np.einsum('i, ij, j ->', F1i, np.linalg.inv(F2ij), F1i)
            if self.do_analytic_marg:
                chi2 = self.compute_chi2_marg(deltaTD, MargVec, self.invCov,
                                              self.prior_term_F0, self.prior_term_F1i,
                                              self.prior_cov_inv)
            elif self.do_jeffreys_priors:
                chi2 = self.compute_chi2_jeffreys(deltaTD, MargVec, self.invCov,
                                                  self.prior_term_F0, self.prior_term_F1i,
                                                  self.prior_cov_inv, self.prior_vector)
        else:
            chi2r_TD = einsum('rj,j->r', self.Mat_DCov, TheoryVec)
            chi2r_TT = einsum('i,ij,j', TheoryVec, self.invCov, TheoryVec)
            chi2 = chi2r_TT - 2.*chi2r_TD + self.Mat_DD

        return chi2

    def model_varied_cosmology(self, params):
        # Update full param dictionary with new param values
        self.full_param_dict.update({key: value for key, value in
                                     zip(self.varied_params, params)})
        # Update derived parameters
        parhandler.update_derived_parameters(self.prior_dictionary,
                                             self.full_param_dict)
        if self.do_redshift_rescaling:
            redshift=0
        else:
            redshift=self.z

        _, PL = self.linear_power_spectrum(linear=self.linear,
                                           redshift=redshift,
                                           cosmo=self.full_param_dict)
        self.full_param_dict['PL'] = PL

        # if self.do_AP:
        #     alpha_par =  self.Hubble_adim_fid / \
        #         self.Hubble_adim(self.z, cosmo=self.full_param_dict)
        #     alpha_perp = self.angular_diam_distance(
        #         self.z, cosmo=self.full_param_dict) / self.angular_diam_distance_fid
        #     self.full_param_dict['alpha_par'] = alpha_par
        #     self.full_param_dict['alpha_perp'] = alpha_perp

        if hasattr(self, 'do_pell' or 'do_bell'):
            f, D = self._get_growth_functions(self.z, cosmo=self.full_param_dict)
            self.full_param_dict['f'] = f
            self.full_param_dict['D'] = D

        theory_dict = {}
        if hasattr(self, 'do_preal'):
            Funcs = self.PowerfuncIR_real(self.z, self.do_redshift_rescaling,
                                          cosmo = self.full_param_dict,
                                          kind = self.IRresum_kind)

            Ptempl = []
            for f in Funcs:
                iFk = InterpolatedUnivariateSpline(self.kL, f, k=3)
                Ptempl.append(
                    self.real_space_power_binner.bin_function(iFk, x_scale=self.kf))
            Ptempl.append(np.ones(self.nbinsP))

            alphas = parhandler.pk_alphas(self.Psn, 0, self.full_param_dict)
            Pgg = einsum('i,ij->j', alphas[1:], Ptempl[1:])
            theory_dict['Pk'] = Pgg[self.IdxP]

        if hasattr(self, 'do_pell'):
            self._Pgg_kmu_terms(cosmo=self.full_param_dict, redshift=redshift,
                                kind=self.IRresum_kind, do_AP=self.do_AP,
                                window_convolution= self.Dict["window"]["convolve"])
            if self.Dict["window"]["convolve"] is not None:
                Pell_eff = self.P_ell_conv_varied_cosmo(
                    self.z, self.do_redshift_rescaling,
                    cosmo=self.full_param_dict, Psn=self.Psn, kgrid=self.kL,
                    **self.full_param_dict)

            else:
                Pell_eff = self.P_kmu_z(self.z, self.do_redshift_rescaling,
                                        cosmo=self.full_param_dict, Psn=self.Psn,
                                        kgrid=self.kPE, **self.full_param_dict)


            theory_dict['P0'] = Pell_eff[0][self.IdxP]
            theory_dict['P2'] = Pell_eff[1][self.IdxP]
            theory_dict['P4'] = Pell_eff[2][self.IdxP4]
        # if 'Q0' in self.Obs:
        #     Qcut0 = self.get_Q0(Pell_eff[0], Pell_eff[1], Pell_eff[2])
        #     theory_dict['Q0'] = Qcut0[self.Idx['Q0']]

        if hasattr(self, 'do_breal'):
            alphas = parhandler.bk_alphas(self.Psn, 0, **self.full_param_dict)
            Btempl = self.Bggg_eff(PLO = PL)
            Bggg = einsum('i,ij->j', alphas, Btempl)
            theory_dict['Bk'] = Bggg[self.IdxB]

        if hasattr(self, 'do_bell'):
            # !!!Make this select which multipoles to compute for speed
            bias_list = [self.alphas_dict[obs](self.Psn,
                                               self.SNLeaking,
                                               **self.full_param_dict)
                         for obs in ['B0','B2','B4']]
            bispfunc = self.Bispfunc_RSD(
                self.z, self.kB1E, self.kB2E, self.kB3E,
                cosmo=self.full_param_dict, D=D,
                do_redshift_rescaling=self.do_redshift_rescaling)

            theory_dict['B0'] = einsum('i,ij->j',
                                       bias_list[0], bispfunc[0])[self.IdxB]

            if 'B2' in self.Obs:
                theory_dict['B2'] = einsum('i,ij->j',
                                           bias_list[1], bispfunc[1])[self.IdxB2]
            if 'B4' in self.Obs:
                theory_dict['B4'] = einsum('i,ij->j',
                                           bias_list[2], bispfunc[2])[self.IdxB4]

        TheoryVec = np.concatenate([theory_dict[obs] for obs in self.Obs])

        chi2r_TD = einsum('rj,j->r', self.Mat_DCov, TheoryVec)
        chi2r_TT = einsum('i,ij,j', TheoryVec, self.invCov, TheoryVec)

        return chi2r_TT - 2.*chi2r_TD + self.Mat_DD


    def model_varied_cosmology_analytic_marg(self, params):
        # Update full param dictionary with new param values
        self.full_param_dict.update({key: value for key, value in
                                     zip(self.varied_params, params)})
        # Update derived parameters
        parhandler.update_derived_parameters(self.prior_dictionary,
                                             self.full_param_dict)
        if self.do_redshift_rescaling:
            redshift=0
        else:
            redshift=self.z

        _, PL = self.linear_power_spectrum(linear=self.linear,
                                           redshift=redshift,
                                           cosmo=self.full_param_dict)
        self.full_param_dict['PL'] = PL

        # if self.do_AP:
        #     alpha_par =  self.Hubble_adim_fid / \
        #         self.Hubble_adim(self.z, cosmo=self.full_param_dict)
        #     alpha_perp = self.angular_diam_distance(
        #         self.z, cosmo=self.full_param_dict) / self.angular_diam_distance_fid
        #     self.full_param_dict['alpha_par'] = alpha_par
        #     self.full_param_dict['alpha_perp'] = alpha_perp

        if hasattr(self, 'do_pell' or 'do_bell'):
            f, D = self._get_growth_functions(self.z, cosmo=self.full_param_dict)
            self.full_param_dict['f'] = f
            self.full_param_dict['D'] = D

        theory_dict = {}
        marg_dict   = {}

        if hasattr(self, 'do_pell'):
            self._Pgg_kmu_terms(cosmo=self.full_param_dict, redshift=redshift,
                                kind=self.IRresum_kind)
            Pell, Pell_marg = self.P_kmu_z_marg(self.z, self.do_redshift_rescaling,
                                                cosmo=self.full_param_dict, Psn=self.Psn,
                                                kgrid=self.kPE, **self.full_param_dict)

            theory_dict['P0'] = Pell[0][self.IdxP]
            theory_dict['P2'] = Pell[1][self.IdxP]
            theory_dict['P4'] = Pell[2][self.IdxP4]

            marg_dict['P0'] = np.asarray([Pell_marg[0+3*j,self.IdxP]
                                          for j in self.index_marg_param])
            marg_dict['P2'] = np.asarray([Pell_marg[1+3*j,self.IdxP]
                                          for j in self.index_marg_param])
            marg_dict['P4'] = np.asarray([Pell_marg[2+3*j,self.IdxP4]
                                          for j in self.index_marg_param])

        if hasattr(self, 'do_bell'):
            raise NotImplementedError("AM bispectrum not implemented yet")

        TheoryVec = np.concatenate([theory_dict[obs] for obs in self.Obs])
        MargVec = np.hstack([marg_dict[obs] for obs in self.Obs])

        deltaTD = TheoryVec - np.concatenate(self.CutDataVecs, axis=1)

        # F0 = np.einsum('ri, ij, rj -> r', deltaTD, self.invCov, deltaTD) +\
        #     self.prior_term_F0
        # F1i = -np.einsum('ri, ij, kj -> r', MargVec, self.invCov, deltaTD) +\
        #     self.prior_term_F1i
        # F2ij = np.einsum('ri, ij, mj -> rm', MargVec, self.invCov, MargVec) +\
        #     self.prior_cov_inv

        # chi2 = F0 + np.log(np.linalg.det(F2ij)) - \
        #     np.einsum('i, ij, j ->', F1i, np.linalg.inv(F2ij), F1i)
        if self.do_analytic_marg:
            chi2 = self.compute_chi2_marg(deltaTD, MargVec, self.invCov,
                                          self.prior_term_F0, self.prior_term_F1i,
                                          self.prior_cov_inv)
        elif self.do_jeffreys_priors:
            chi2 = self.compute_chi2_jeffreys(deltaTD, MargVec, self.invCov,
                                              self.prior_term_F0, self.prior_term_F1i,
                                              self.prior_cov_inv, self.prior_vector)

        return chi2


    def model_varied_cosmology_analytic_marg_multiz(self, params):
        # Update full param dictionary with new param values
        self.full_param_dict.update({key: value for key, value in
                                     zip(self.varied_params, params)})
        # Update derived parameters
        parhandler.update_derived_parameters(self.prior_dictionary,
                                             self.full_param_dict)

        _, PL = self.linear_power_spectrum(linear=self.linear,
                                           redshift=0,
                                           cosmo=self.full_param_dict)
        self.full_param_dict['PL'] = PL

        if self.do_AP:
            alpha_par =  self.Hubble_adim_fid / \
                self.Hubble_adim(self.z_bins, cosmo=self.full_param_dict)
            alpha_perp = np.asarray(
                [self.angular_diam_distance(iz, cosmo=self.full_param_dict)
                 for iz in self.z_bins]) / self.angular_diam_distance_fid
            self.full_param_dict['alpha_par'] = list(alpha_par)
            self.full_param_dict['alpha_perp'] = list(alpha_perp)

        if hasattr(self, 'do_pell' or 'do_bell'):
            f, D = self._get_growth_functions(self.z_bins, cosmo=self.full_param_dict)
            self.full_param_dict['f'] = list(f)
            self.full_param_dict['D'] = list(D)

        theory_dict = {k: {} for k,i in enumerate(self.z_bins)}
        marg_dict   = {k: {} for k,i in enumerate(self.z_bins)}
        chi2 = 0

        if hasattr(self, 'do_pell'):
            self._Pgg_kmu_terms(cosmo=self.full_param_dict, redshift=0,
                                kind=self.IRresum_kind)
        
        for ii, iz in enumerate(self.z_bins):
            zparams = {k: v[ii] if isinstance(v, list)
                       else v for k, v in self.full_param_dict.items()}
            Pell, Pell_marg = self.P_kmu_z_marg(
                iz, True, AP_as_nuisance=True,
                cosmo=self.full_param_dict, Psn=self.Psn[ii],
                kgrid=self.kPE, **zparams)

            theory_dict[ii]['P0'] = Pell[0][self.IdxP[ii]]
            theory_dict[ii]['P2'] = Pell[1][self.IdxP[ii]]
            theory_dict[ii]['P4'] = Pell[2][self.IdxP4[ii]]

            marg_dict[ii]['P0'] = np.asarray([Pell_marg[0+3*j,self.IdxP[ii]]
                                              for j in self.index_marg_param])
            marg_dict[ii]['P2'] = np.asarray([Pell_marg[1+3*j,self.IdxP[ii]]
                                              for j in self.index_marg_param])
            marg_dict[ii]['P4'] = np.asarray([Pell_marg[2+3*j,self.IdxP4[ii]]
                                              for j in self.index_marg_param])

            TheoryVec = np.concatenate([theory_dict[ii][obs] for obs in self.Obs])
            MargVec = np.hstack([marg_dict[ii][obs] for obs in self.Obs])

            deltaTD = np.atleast_2d(TheoryVec - np.concatenate(self.CutDataVecs[ii]))

            # F0 = np.einsum('ri, ij, rj -> r', deltaTD, self.invCov[ii], deltaTD) +\
            #     self.prior_term_F0[ii]
            # F1i = -np.einsum('ri, ij, kj -> r', MargVec, self.invCov[ii], deltaTD) +\
            #     self.prior_term_F1i[ii]
            # F2ij = np.einsum('ri, ij, mj -> rm', MargVec, self.invCov[ii], MargVec) +\
            #     self.prior_cov_inv[ii]

            # chi2_iz = F0 + np.log(np.linalg.det(F2ij)) - \
            #     np.einsum('i, ij, j ->', F1i, np.linalg.inv(F2ij), F1i)

            if self.do_analytic_marg:
                chi2_iz = self.compute_chi2_marg(deltaTD, MargVec,
                                                 self.invCov[ii],
                                                 self.prior_term_F0[ii],
                                                 self.prior_term_F1i[ii],
                                                 self.prior_cov_inv[ii])
            elif self.do_jeffreys_priors:
                chi2_iz = self.compute_chi2_jeffreys(deltaTD, MargVec,
                                                     self.invCov[ii],
                                                     self.prior_term_F0[ii],
                                                     self.prior_term_F1i[ii],
                                                     self.prior_cov_inv[ii],
                                                     self.prior_vector[ii])
            chi2 += chi2_iz

        if self.store_theorydict:
            self.theorydict= theory_dict
            print('Storing theorydict, althought it makes not much sense with analytical marginalization')
        return chi2

    def model_BAO(self, params):
        # Update full param dictionary with new param values
        self.full_param_dict.update({key: value for key, value in
                                     zip(self.varied_params, params)})
        # Update derived parameters
        parhandler.update_derived_parameters(self.prior_dictionary,
                                             self.full_param_dict)

        Pell_eff = self.Pgg_BAO_l(self.z, kgrid=self.kPE, Psn=self.Psn,
                                  **self.full_param_dict)
        theory_dict = {'BAO_P0': Pell_eff[0][self.IdxP],
                       'BAO_P2': Pell_eff[1][self.IdxP],
                       'BAO_P4': Pell_eff[2][self.IdxP]}

        TheoryVec = np.concatenate([theory_dict[obs] for obs in self.Obs])

        chi2r_TD = einsum('rj,j->r', self.Mat_DCov, TheoryVec)
        chi2r_TT = einsum('i,ij,j', TheoryVec, self.invCov, TheoryVec)

        return chi2r_TT - 2.*chi2r_TD + self.Mat_DD

    def model_BAO_analytic_marg(self, params):
        # Update full param dictionary with new param values
        self.full_param_dict.update({key: value for key, value in
                                     zip(self.varied_params, params)})
        # Update derived parameters
        parhandler.update_derived_parameters(self.prior_dictionary,
                                             self.full_param_dict)

        Pell_eff, Pell_marg = self.Pgg_BAO_l_marg(self.z, kgrid=self.kPE, Psn=self.Psn,
                                  **self.full_param_dict)
        theory_dict = {'BAO_P0': Pell_eff[0][self.IdxP],
                       'BAO_P2': Pell_eff[1][self.IdxP],
                       'BAO_P4': Pell_eff[2][self.IdxP]}
        marg_dict = {}
        marg_dict['BAO_P0'] = np.asarray([Pell_marg[0+3*j,self.IdxP]
                                          for j in self.index_marg_param])
        marg_dict['BAO_P2'] = np.asarray([Pell_marg[1+3*j,self.IdxP]
                                          for j in self.index_marg_param])
        marg_dict['BAO_P4'] = np.asarray([Pell_marg[2+3*j,self.IdxP]
                                          for j in self.index_marg_param])

        TheoryVec = np.concatenate([theory_dict[obs] for obs in self.Obs])
        MargVec = np.hstack([marg_dict[obs] for obs in self.Obs])

        deltaTD = TheoryVec - np.concatenate(self.CutDataVecs, axis=1)

        # F0 = np.einsum('ri, ij, rj -> r', deltaTD, self.invCov, deltaTD) +\
        #     self.prior_term_F0
        # F1i = -np.einsum('ri, ij, kj -> r', MargVec, self.invCov, deltaTD) +\
        #     self.prior_term_F1i
        # F2ij = np.einsum('ri, ij, mj -> rm', MargVec, self.invCov, MargVec) +\
        #     self.prior_cov_inv

        # chi2 = F0 + np.log(np.linalg.det(F2ij)) - \
        #     np.einsum('i, ij, j ->', F1i, np.linalg.inv(F2ij), F1i)
        if self.do_analytic_marg:
            chi2 = self.compute_chi2_marg(deltaTD, MargVec, self.invCov,
                                          self.prior_term_F0, self.prior_term_F1i,
                                          self.prior_cov_inv)
        elif self.do_jeffreys_priors:
            chi2 = self.compute_chi2_jeffreys(deltaTD, MargVec, self.invCov,
                                              self.prior_term_F0, self.prior_term_F1i,
                                              self.prior_cov_inv, self.prior_vector)
        return chi2

#-------------------------------------------------------------------------------

    def lnprior(self, params):
        """
        Computes the value of the log-prior given particular values of the
        parameters

        Arguments:
            params (list):
        """
        lp = 1.
        for pstr,pval in zip(self.varied_params, params):
            lp *= self.prior_dictionary[pstr]['distr'](pval)
        if lp: return log(lp)

        return -inf

#-------------------------------------------------------------------------------

    def prior_nested(self, quantile_cube: list) -> list:
        """
        Constructs array of priors for nested sampling, transforming from the
        unit hypercube to non-unit uniform / Gaussian priors, as specified in
        the prior_dictionary

        Arguments:
            quantile_cube: parameters

        Returns:
            transformed_pars: transformed parameters
        """
        transformed_pars = np.empty_like(quantile_cube)
        for i in range(len(self.varied_params)):
            transformed_pars[i] = priorhandler.prior_transform(self.prior_dictionary,
                                                               self.varied_params[i],
                                                               quantile_cube[i])

        return transformed_pars

#-------------------------------------------------------------------------------

    def lnprob(self, params: list) -> float:
        """
        Computes the log-posterior and the chi-square given particular values
        of the parameters, for each dataset

        Arguments:
            params: parameters

        Returns:
            lp+lnLike, chi2: logprior+loglikelihood, chisquare.
                If lp is infinite, returns -inf, inf
        """
        lp = self.lnprior(params)
        if isfinite(lp):
            lnLike, chi2 = self.lnlike(params)
            return lp + lnLike, chi2

        return -inf, inf

#-------------------------------------------------------------------------------

    def lnprob_nested(self, cube):
        """
        Computes the log-posterior for the PyMultiNest sampler

        Arguments:
            cube: parameters

        Returns:
            lnLike: loglikelihood
        """
        params = cube
        lnLike, chi2 = self.lnlike(params)

        return lnLike

#-------------------------------------------------------------------------------

    def lnlike_nautilus(self, param_dict):
        """
        Computes the log-likelihood for the nautilus sampler

        Arguments:
            params: parameters

        Returns:
            lnLike: loglikelihood
        """
        def compress_dict(expanded_dict):
            result = {}
            for key, value in expanded_dict.items():
                if '_z' in key:
                    main_key, sub_key = key.split('_z', 1)
                    if main_key not in result:
                        result[main_key] = []
                    result[main_key].append(value)
                else:
                    result[key] = value
            return result

        param_dict = compress_dict(param_dict)

        params = [param_dict[key] for key in self.varied_params]
        lnLike, chi2 = self.lnlike(params)
        return lnLike

#-------------------------------------------------------------------------------

    def lnlike_pocomc(self, params):
        """
        Log-probability function for the pocoMC sampler

        Arguments:
            params: parameters

        Returns:
            lnLike: loglikelihood
        """
        lnLike, chi2 = self.lnlike(params)
        return lnLike

#-------------------------------------------------------------------------------

    def lnlike_gaussian(self, params):
        """
        Computes the total log-likelihood and the total chi-square with a
        Gaussian likelihood function

        Arguments:
            params: parameters

        Returns:
            lnL, -2*lnL: loglikelihood and chisquare
        """

        chi2r = self.model_function(params)
        lnL = -0.5*einsum('i->', chi2r)
        return lnL, -2.*lnL

#-------------------------------------------------------------------------------

    def lnlike_hartlap(self, params):
        """
        Computes the total log-likelihood and the total chi-square with a
        Gaussian likelihood function corrected by the Hartlap correction

        Arguments:
            params: parameters

        Returns:
            lnL, sum(chi2r): loglikelihood and chisquare
        """
        chi2r = self.model_function(params)
        lnLr = -0.5 * ((self.Nmocks - self.Npts - 2.)/(self.Nmocks - 1.))*chi2r
        lnL = sum(lnLr)

        return lnL, sum(chi2r)

#-------------------------------------------------------------------------------

    def lnlike_sellentin(self, params):
        """
        Computes the total log-likelihood and the total chi-square with the
        likelihood function from Sellentin & Heavens 2016

        Arguments:
            params: parameters

        Returns:
            lnL, sum(chi2r): linlikelihood and chisquare
        """
        chi2r = self.model_function(params)
        lnL = -0.5 * self.Nmocks * einsum('i->', log(1 + chi2r/(self.Nmocks-1)))

        return lnL, einsum('i->',chi2r)

#-------------------------------------------------------------------------------

    def compute_chi2_marg(self, delta_theorydata, p_marg, invcov,
                          prior_term_F0, prior_term_F1i, prior_invcov):
        
        F0 = np.einsum('ri, ij, rj -> r', delta_theorydata, invcov,
                       delta_theorydata) + prior_term_F0
        F1i = -np.einsum('ri, ij, kj -> r', p_marg, invcov, delta_theorydata) +\
            prior_term_F1i
        F2ij = np.einsum('ri, ij, mj -> rm', p_marg, invcov, p_marg) +\
            prior_invcov

        chi2 = F0 + np.log(np.linalg.det(F2ij)) - np.einsum('i, ij, j ->', F1i,
                                                            np.linalg.inv(F2ij), F1i)
        return chi2

#-------------------------------------------------------------------------------

    def compute_chi2_jeffreys(self, delta_theorydata, p_marg, invcov,
                              prior_term_F0, prior_term_F1i, prior_invcov, prior_vec):
        F1i = -np.einsum('ri, ij, kj -> r', p_marg, invcov, delta_theorydata) +\
            prior_term_F1i
        F2ij = np.einsum('ri, ij, mj -> rm', p_marg, invcov, p_marg) +\
            prior_invcov
        nuis = np.einsum('i, ij -> j', F1i, np.linalg.inv(F2ij))
        chi2_prior = np.dot(nuis - prior_vec, np.dot(nuis - prior_vec, prior_invcov))
        delta_tot = delta_theorydata + np.einsum('r, ri -> i', nuis, p_marg)

        chi2 = np.einsum('ri, ij, rj -> ', delta_tot, invcov, delta_tot) + chi2_prior

        return chi2

#-------------------------------------------------------------------------------

    def Chi2_fixCov_fixCosmo(self, alpha):
        """
        Support function to compute the log-likelihood when cosmological
        parameters are fixed

        Arguments:
            alpha: parameters

        Returns:
            chi2r_TT-2*chi2r_TD+chi2r_DD: chisquare
        """
        chi2r_DD = self.Mat_DD
        chi2r_TD = einsum('i,ri', alpha, self.Mat_TD)
        chi2r_TT = einsum('i,j,ij', alpha, alpha, self.Mat_TT)

        return chi2r_TT - 2*chi2r_TD + chi2r_DD

#-------------------------------------------------------------------------------

    def CutVectors(self,
                   NmaxP: Union[float, List[float]] = 0,
                   NminP: Union[float, List[float]] = 0,
                   NmaxB: Union[float, List[float]] = 0,
                   NmaxB2: Optional[Union[float, List[float]]] = None,
                   NmaxP4: Optional[Union[float, List[float]]] = None,
                   NmaxB4: Optional[Union[float, List[float]]] = None):
        """
        Given values of NmaxP and NmaxB, trims the datavectors, the theory
        templates and the covariance matrix, then calls the function to combine
        all quantities together for a fast evaluation of the likelihood function

        Arguments:
            NmaxP: max value of kmaxP in units of the fundamental frequency
            NmaxB: max value of kmaxB in units of the fundamental frequency
            NmaxB2: max value of kmaxB2 in units of the fundamental frequency
            NmaxP4: max value of kmaxP4 in units of the fundamental frequency
            NmaxB4: max value of kmaxB4 in units of the fundamental frequency

        Returns:
            sets the following as attributes: IdxP, IdxP4, IdxB, IdxB2.
            Sets CutTheoryVecs, CutDataVecs, Npts, IdxTot
        """
        def to_list(x, length=1):
            return [x] * length if isinstance(x, (int, float)) else x

        def create_bool_mask(k, Nmax, Nmin=None):
            if Nmin is not None:
                return [(k <= Nmax[i]) & (k >= Nmin[i]) for i in range(len(Nmax))]
            return [(k <= Nmax[i]) for i in range(len(Nmax))]


        scalar_inputs = isinstance(NmaxP, (int, float))
        num_bins = len(to_list(NmaxP))

        # Convert all inputs to lists
        NmaxP, NminP, NmaxB = map(lambda x: to_list(x, num_bins), [NmaxP, NminP, NmaxB])
        NmaxB2 = to_list(NmaxB2, num_bins) if NmaxB2 is not None else None
        NmaxP4 = to_list(NmaxP4, num_bins) if NmaxP4 is not None else None
        NmaxB4 = to_list(NmaxB4, num_bins) if NmaxB4 is not None else None

        # Compute indices
        self.IdxP = create_bool_mask(self.kP, NmaxP, NminP)
        self.IdxB = create_bool_mask(self.kB1, NmaxB)
        self.IdxP4 = self.IdxP if NmaxP4 is None else create_bool_mask(self.kP, NmaxP4,
                                                                       NminP)
        self.IdxB2 = self.IdxB if NmaxB2 is None else create_bool_mask(self.kB1, NmaxB2)
        self.IdxB4 = self.IdxB if NmaxB4 is None else create_bool_mask(self.kB1, NmaxB4)
        for i in range(num_bins):
            self.IdxB4[i][4] = False

        idx_dict = {'Pk': self.IdxP, 'P0': self.IdxP, 'P2': self.IdxP, 'P4': self.IdxP4,
                    'Bk': self.IdxB, 'B0': self.IdxB, 'B2': self.IdxB2, 'B4': self.IdxB4,
                    'P0_BAO': self.IdxP, 'P2_BAO': self.IdxP, 'P4_BAO': self.IdxP4}

        def select_template(obs):
            if obs[0] == 'P':
                templ = self.P_ell_Templ[obs]
            elif obs[0] == 'B' and O[1] != 'A':
                templ = self.B_ell_Templ[obs]
            return templ

        template_dict = {obs: array(list(select_template(obs).values())) for obs in self.Obs}

        # Initialize vectors and masks
        self.CutDataVecs = [[] for _ in range(num_bins)]
        self.CutTheoryVecs = [[] for _ in range(num_bins)]
        IdxTot = [[] for _ in range(num_bins)]
        self.invCov = [[] for _ in range(num_bins)]

        if not scalar_inputs:
            for i in range(num_bins):
                self.CutDataVecs[i] = [self.DataDict[obs][i][idx_dict[obs][i]]
                                       for obs in self.Obs]
                IdxTot[i] = hstack([idx_dict[obs][i] for obs in self.Obs])
                cut_cov = self.Cov[i][ix_(IdxTot[i].astype(bool),
                                          IdxTot[i].astype(bool))]
                self.invCov[i] = inv(cut_cov)

        else:
            self.CutDataVecs = [self.DataDict[obs][:,idx_dict[obs][0]] for obs in self.Obs]
            self.CutTheoryVecs = [template_dict[obs][:,idx_dict[obs][0]] for obs in self.Obs]
            IdxTot = hstack([idx_dict[obs][0] for obs in self.Obs])
            cut_cov = self.Cov[ix_(IdxTot.astype(bool),
                                   IdxTot.astype(bool))]
            self.invCov = inv(cut_cov)
            self.IdxP = self.IdxP[0]
            self.IdxB = self.IdxB[0]
            self.IdxP4 = self.IdxP4[0]
            self.IdxB2 = self.IdxB2[0]
            self.IdxB4 = self.IdxB4[0]

            if self.Inputcosmo is not None:
                self.CombineVectors()

        self.Npts = sum(IdxTot)

#-------------------------------------------------------------------------------

    def CombineVectors(self):
        """ Combines all quantities together for a fast evaluation of the
        likelihood function.

        Sets the following as attributes of the class:
            Mat_DD, Mat_TD, Mat_TT
        """
        DATA   = hstack((self.CutDataVecs))
        self.Mat_DCov = einsum('ri,ij->rj', DATA, self.invCov)
        self.Mat_DD = einsum('ri,ij,rj->r',  DATA,   self.invCov, DATA)


        if self.CutTheoryVecs != []:
            THEORY = block_diag(*self.CutTheoryVecs)
            self.Mat_TD = einsum('ri,ij,tj->rt', DATA,   self.invCov, THEORY)
            self.Mat_TT = einsum('ti,ij,sj->ts', THEORY, self.invCov, THEORY)

#-------------------------------------------------------------------------------

    def create_nuisance_list(self, **pars):
        flat_list = np.concatenate([self.alphas_dict[obs](self.Psn,
                                                          self.SNLeaking,
                                                          **pars)
                                    for obs in self.Obs])
        return flat_list

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

class PBJsampler:

    def __init__(self):
        self.ndim = len(self.varied_params)

#-------------------------------------------------------------------------------

    def run_sampler(self, **kwargs):
        print("\033[1;32m[info] \033[00m"+"Sampling with "+self.sampler)
        if self.sampler == 'emcee':
            return self.PBJemcee(**kwargs)
        elif self.sampler == 'Metropolis-Hastings':
            return self.PBJMetropolis(**kwargs)
        elif self.sampler == 'multinest':
            return self.PBJmultinest(**kwargs)
        elif self.sampler == 'pocomc':
            return self.PBJpocomc(**kwargs)
        elif self.sampler == 'nautilus':
            return self.PBJnautilus(**kwargs)
        elif self.sampler == 'ultranest':
            return self.PBJultranest(**kwargs)
        else:
            print("\033[1;31m[WARNING]: \033[00m"+"Sampler not implemented")

#-------------------------------------------------------------------------------

    def _prepare_sampler(self, NmaxP, NminP, NmaxB, NmaxB2, NmaxP4, NmaxB4,
                         savelog):
        """
        Support function for the initialization of samplers. Sets the
        nmax, the filenames, and cuts the datavectors according to
        NmaxX for observable X in the self.Obs list.
        """

        self._set_nmax(NmaxP, NmaxB, NmaxB2, NmaxP4, NmaxB4)
        self.set_file_names()

        if savelog:
            logfile = os.path.join(self.outfolder, self.logfile)
            sys.stdout = open(logfile, 'w')
        self._print_nmax()

        self.CutVectors(NmaxP=NmaxP, NminP=NminP, NmaxB=NmaxB, NmaxB2=NmaxB2,
                        NmaxP4=NmaxP4, NmaxB4=NmaxB4)
        setattr(self, 'ndim', len(self.varied_params))

#-------------------------------------------------------------------------------

    def _set_nmax(self, NmaxP, NmaxB, NmaxB2, NmaxP4, NmaxB4):
        """Sets nmax_dict for observable X as attributes
        """

        maxbins = {'P': ((arange(self.nbinsP)+0.5)*self.dk+self.cf),
                   'B': ((arange(self.nbinsB)+0.5)*self.dk+self.cf)}

        if not NmaxP4:
            NmaxP4 = NmaxP

        Nmax = {'Pk': NmaxP, 'P0': NmaxP, 'P2': NmaxP, 'P4': NmaxP4,
                'P0_BAO': NmaxP, 'P2_BAO': NmaxP, 'P4_BAO': NmaxP4,
                'Bk': NmaxB, 'B0': NmaxB, 'B2': NmaxB2, 'B4': NmaxB4}

        self.nmax_dict = {}

        for o in Nmax.keys():
            if isinstance(Nmax[o], (float,int)):
                self.nmax_dict[o] = [maxbins[o[0]][abs(maxbins[o[0]]-Nmax[o]).argmin()]]
            elif isinstance(Nmax[o], list):
                self.nmax_dict[o] = [maxbins[o[0]][abs(maxbins[o[0]]-nmax).argmin()]
                                     for nmax in Nmax[o]]

#-------------------------------------------------------------------------------

    def _print_nmax(self):
        """Print kmax values to log / stdout
        """
        def format_string(nmax):
            if len(nmax)>1:
                formatted_values = ", ".join(["{0:.4f} h/Mpc".format(inmax*self.kf)
                                              for inmax in nmax])
            else:
                formatted_values = "{0:.4f} h/Mpc".format(nmax[0] * self.kf)
            return formatted_values

        for o in self.Obs:
            if self.nmax_dict[o] != None:
                print("Closest available kmax for ", o, ": ",
                      format_string(self.nmax_dict[o]))

#-------------------------------------------------------------------------------

    def PBJemcee(self, NmaxP=0, NmaxB=0, NmaxB2=None, NmaxP4=None, NmaxB4=None,
                 NminP=0,
                 nwalker: Optional[int] = 100,
                 nsteps: Optional[int] = 50000,
                 progress: Optional[bool] = True,
                 slow: Optional[bool] = False,
                 savelog: Optional[bool] = True,
                 savetau: Optional[bool] = True):
        """
        Implementation of the affine invariant sampler of emcee.

        Arguments:
            NmaxP: kmaxP in units of fundamental frequency
            NmaxB: kmaxB in units of fundamental frequency
            NmaxP4: kmaxP4 in units of fundamental frequency
            NmaxB2: kmaxB2 in units of fundamental frequency
            NmaxB4: kmaxB4 in units of fundamental frequency
            nwalker: number of walkers in the sampler
            nsteps: max number of iterations in the MCMC
            savelog: if True the log of the run is saved to file
            savetau: if True saves the autocorrelation times for each parameter
        """

        self._prepare_sampler(NmaxP, NminP, NmaxB, NmaxB2, NmaxP4, NmaxB4, savelog)

        if not self.CheckConvergence:
            savetau = False

        start_point = [self.initial_param_dict[i]['value'] for i in self.varied_params]

        # Small ball
        pos = [start_point + 5.e-2 * (randn(self.ndim) - 0.5)
               for i in range(nwalker)]

        sampler = EnsembleSampler(nwalker, self.ndim, self.lnprob)

        if not slow:
            if self.CheckConvergence:
                idx = 0
                self.autocorr = empty((int(nsteps/1000)+1, self.ndim))
                old_tau = inf

                for sample in sampler.sample(pos,iterations=nsteps,progress=progress):
                    if sampler.iteration % 1000: continue
                    tau = sampler.get_autocorr_time(tol=0)
                    self.autocorr[idx] = tau
                    idx += 1

                    try:
                        converged = all(tau*100 < sampler.iteration) & all(abs(old_tau - tau)/tau < 0.01)
                    except:
                        converged = False
                    if converged or sampler.iteration == nsteps:
                        burnin = int(2 * max(tau))
                        thin = int(0.1 * min(tau))
                        break
                    old_tau =  tau
                print('\n'.join('\t'.join(str(cell) for cell in row)
                                for row in self.autocorr[:idx]))
            else:
                sampler.run_mcmc(pos, nsteps, progress=True)
                tau = sampler.get_autocorr_time(tol=0)
                burnin = int(2 * max(tau))
                thin = int(0.1 * min(tau))

            Storedchain   =sampler.get_chain(flat=True,discard=burnin,thin=thin)
            Storedlogprob =sampler.get_log_prob(flat=True,discard=burnin,thin=thin)
            Storedchi2tot =sampler.get_blobs(flat=True,discard=burnin,thin=thin)

            self.Chain = c_[Storedchain, Storedlogprob, Storedchi2tot]

            names = [self.prior_dictionary[p]['latex']
                                         for p in self.varied_params]
            chaindict = {'samples': self.Chain[:,:-2],
                         'logp': self.Chain[:,-2],
                         'chisquare': self.Chain[:,-1]}

            parhandler.add_names_to_chain(self.prior_dictionary, self.varied_params,
                                          chaindict)

            np.savez(os.path.join(self.outfolder, self.chainfile+".npz"), **chaindict)
            if savetau:
                savetxt(self.taufile, self.autocorr,header='\t'.join(self.varied_params))

        else:
            WalkerPos = zeros([nwalker,self.ndim+2])
            if self.Resume:
                chain = read_csv(self.chainfile)
                pos = chain[-1,:-2]
                lenchain = chain.shape[0]
            else:
                f = open(self.chainfile, "w")
                f.close()
                lenchain = 0
                chain = np.zeros([0,self.ndim+2])

            for sample in sampler.sample(pos, iterations=nsteps-lenchain,
                                         progress=progress):
                position, logprob, randomstate, blob = sample
                f = open(self.chainfile, "a")
                WalkerPos[:,:-2] = position
                WalkerPos[:,-2] = logprob
                WalkerPos[:,-1] = blob
                for j in range(nwalker):
                    String = '\t'.join(str(i) for i in WalkerPos[j])
                    f.write(String+"\n")
                f.close()
                chain = np.vstack((chain, WalkerPos))

#-------------------------------------------------------------------------------

    def PBJMetropolis(self, NmaxP=0, NminP=0, NmaxB=0, NmaxB2=None, NmaxP4=None,
                      NmaxB4=None, nsteps: Optional[int] = 100000):
        """
        Implementation of the Metropolis-Hastings sampler from emcee.

        Arguments:
            NmaxP: kmaxP in units of fundamental frequency
            NmaxB: kmaxB in units of fundamental frequency
            NmaxP4: kmaxP4 in units of fundamental frequency
            NmaxB2: kmaxB2 in units of fundamental frequency
            NmaxB4: kmaxB4 in units of fundamental frequency
            nsteps: max number of iterations in the MCMC
        """
        nwalkers = 1

        self._prepare_sampler(NmaxP, NminP, NmaxB, NmaxB2, NmaxP4, NmaxB4, False)

        WalkerPos = [0. for i in range(self.ndim+2)]
        sampler = EnsembleSampler(nwalkers, self.ndim, self.lnprob,
                                  moves=moves.GaussianMove(self.ParCov))

        if self.Resume:
            chain = np.array(read_csv(self.chainfile, header=None, sep="\s+"))
            pos = chain[-1,:-2]
            lenchain = chain.shape[0]
        else:
            f = open(self.chainfile, "w")
            f.close()
            pos = [self.initial_param_dict[i]['value'] for i in self.varied_params]
            lenchain = 0
            chain = np.zeros([0,self.ndim+2])

        for sample in sampler.sample(pos, iterations=nsteps-lenchain,
                                     progress=True,skip_initial_state_check=True):
            position, logprob, randomstate, blob = sample
            f = open(self.chainfile, "a")
            for i in range(self.ndim): WalkerPos[i] = position[0,i]
            WalkerPos[-2] = logprob[0]
            WalkerPos[-1] = blob[0]
            String = '\t'.join(str(i) for i in WalkerPos)
            f.write(String+"\n")
            f.close()
            chain = np.vstack((chain, WalkerPos))

#-------------------------------------------------------------------------------

    def PBJmultinest(self, NmaxP=0, NminP=0, NmaxB=0, NmaxB2=None, NmaxP4=None,
                     NmaxB4=None, n_live_points: Optional[int] = 1500,
                     evidence_tolerance: Optional[float] = 0.1,
                     importance_nested_sampling: Optional[bool] = True,
                     multimodal: Optional[bool] = True,
                     savelog: Optional[bool] = False,
                     resume: Optional[bool] = True):
        """
        Implementation of the multinest sampler.

        Arguments:
            NmaxP: kmaxP in units of fundamental frequency
            NmaxB: kmaxB in units of fundamental frequency
            NmaxP4: kmaxP4 in units of fundamental frequency
            NmaxB2: kmaxB2 in units of fundamental frequency
            NmaxB4: kmaxB4 in units of fundamental frequency
            n_live_points: number of live points
            evidence_tolerance: tolerance on the error on the evidence
            importance_nested_sampling: do INS (default True)
            multimodal: default True
            savelog: saves log to file. If False prints it to the stdout
        """

        self._prepare_sampler(NmaxP, NminP, NmaxB, NmaxB2, NmaxP4, NmaxB4, savelog)

        self.result_PMN = pmn.solve(LogLikelihood=self.lnprob_nested,
                                    Prior=self.prior_nested,
                                    n_dims=self.ndim,
                                    outputfiles_basename=self.chainfile,
                                    resume=resume, verbose=True,
                                    importance_nested_sampling=importance_nested_sampling,
                                    sampling_efficiency=0.8,
                                    n_live_points=n_live_points,
                                    multimodal=multimodal,
                                    evidence_tolerance=evidence_tolerance)

#-------------------------------------------------------------------------------

    def PBJultranest(self, NmaxP=0, NminP=0, NmaxB=0, NmaxB2=None, NmaxP4=None,
                     NmaxB4=None, savelog: Optional[bool] = True,
                     dlogz: Optional[float] = 0.5,
                     frac_remain: Optional[float] = 0.01,
                     min_num_live_points: Optional[int] = 800,
                     print_results: Optional[bool] = False,
                     verbose: Optional[bool] = True):
        """
        Implementation of the ultranest sampler.

        Arguments:
            NmaxP: kmaxP in units of fundamental frequency
            NmaxB: kmaxB in units of fundamental frequency
            NmaxP4: kmaxP4 in units of fundamental frequency
            NmaxB2: kmaxB2 in units of fundamental frequency
            NmaxB4: kmaxB4 in units of fundamental frequency
            savelog: saves log to file. If False prints it to the stdout
            resume: whether to 'resume', 'overwrite', 'subfolder', or 'resume-similar'
            logdir: output directory
            dlogz: target evidence uncertainty
            min_num_live_points: minimum number of live points
        """

        self._prepare_sampler(NmaxP, NminP, NmaxB, NmaxB2, NmaxP4, NmaxB4, savelog)
        if self.Resume == False:
            resume='overwrite'
        elif self.Resume == True:
            resume='resume'
        else:
            resume=self.resume

        sampler = ultranest.ReactiveNestedSampler(self.varied_params,
                                                  self.lnprob_nested,
                                                  self.prior_nested,
                                                  log_dir=self.outfolder,
                                                  resume=resume,
                                                  vectorized=False)
        self.result = sampler.run(dlogz=dlogz, frac_remain = frac_remain,
                                  min_num_live_points=min_num_live_points,
                                  show_status=verbose)
        if print_results:
            sampler.print_results()

#-------------------------------------------------------------------------------

    def PBJpocomc(self, NmaxP=0, NminP=0, NmaxB=0, NmaxB2=None, NmaxP4=None,
                  NmaxB4=None, n_ess: Optional[int] = 1000,
                  n_prior: Optional[int] = 2000,
                  n_cpus: Optional = 1,
                  print_progress: Optional[bool] = False,
                  save_every: Optional = None,
                  savelog: Optional[bool] = True):
        """
        Implementation of the pocoMC sampler.

        Arguments:
            NmaxP: kmaxP in units of fundamental frequency
            NmaxB: kmaxB in units of fundamental frequency
            NmaxP4: kmaxP4 in units of fundamental frequency
            NmaxB2: kmaxB2 in units of fundamental frequency
            NmaxB4: kmaxB4 in units of fundamental frequency
            n_particles: number of live points (default 1000)
            n_add: additional points sampled from posterior (default 20000)
            gamma: correlation coefficient threshold (default 0.75)
            bridge_sampling: do bridge sampling for accurate Z (default False)
            print_progress: if True, print progress bar (default False)
            save_every: if specified, saves status every N steps (default None)
            savelog: if True the log of the run is saved to file (default False)
        """
        self._prepare_sampler(NmaxP, NminP, NmaxB, NmaxB2, NmaxP4, NmaxB4, savelog)

        prior_list = [0]*self.ndim
        for i in range(self.ndim):
            v = self.prior_dictionary[self.varied_params[i]]

            if v['prior'] == 'uniform':
                prior_list[i] = scipy.stats.uniform(
                    loc=v['prior_params']['min'],
                    scale=v['prior_params']['max']-v['prior_params']['min'])

            elif v['prior'] == 'gaussian':
                prior_list[i] = scipy.stats.norm(
                    loc=v['prior_params']['mean'], scale=v['prior_params']['sigma'])

            elif v['prior'] == 'log-normal':
                prior_list[i] = scipy.stats.lognormal(loc=v['prior_params']['mean'],
                                                      scale=v['prior_params']['sigma'])

            elif v['prior'] == 'log-uniform':
                prior_list[i] = scipy.stats.loguniform(
                    loc=v['prior_params']['min'],
                    scale=v['prior_params']['max']-v['prior_params']['min'])
            else:
                raise ValueError('Prior distribution not allowed for',
                                 self.varied_params[i])

        prior = pocomc.Prior(prior_list)

        if self.Resume:
            resume_state_path = os.path.join(self.outfolder,self.chainfile+".state")
        else:
            resume_state_path = None

        with Pool(n_cpus) as pool:
            sampler = pocomc.Sampler(prior  = prior,
                                     likelihood = self.lnlike_pocomc,
                                     n_dim = self.ndim,
                                     n_ess = n_ess,
                                     pool = pool,
                                     n_prior = n_prior,
                                     output_dir = self.outfolder,
                                     output_label = self.chainfile)
            sampler.run(progress = print_progress,
                        resume_state_path = resume_state_path,
                        save_every = save_every)

        parhandler.add_names_to_chain(self.prior_dictionary, self.varied_params,
                                      sampler.results)

        np.savez(os.path.join(self.outfolder, self.chainfile+".npz"), **sampler.results)

        self.Chain = c_[sampler.results['x'],
                        # sampler.results['logw'],
                        sampler.results['logl']+\
                        sampler.results['logp'], # log-posterior
                        -2*sampler.results['logl']] # chi2=-2logL

#-------------------------------------------------------------------------------

    def PBJnautilus(self, NmaxP=0, NminP=0, NmaxB=0, NmaxB2=None, NmaxP4=None,
                    NmaxB4=None, n_live: Optional[int] = 2000,
                    f_live: Optional[float] = 0.01,
                    pool: Optional[int] = 1,
                    savelog: Optional[bool] = True,
                    resume: Optional[bool] = False,
                    checkpointing: Optional[bool] = True,
                    verbose: Optional[bool] = False,
                    n_eff: Optional[int] = 10000):
        """
        Implementation of the nautilus sampler.

        Arguments:
            NmaxP: kmaxP in units of fundamental frequency
            NmaxB: kmaxB in units of fundamental frequency
            NmaxP4: kmaxP4 in units of fundamental frequency
            NmaxB2: kmaxB2 in units of fundamental frequency
            NmaxB4: kmaxB4 in units of fundamental frequency
            n_live: number of live points (default 1000)
            pool: multithread (default 1)
            savelog: saves log to file. If False prints it to the stdout
            resume: if False, overwrites previous run. If True resumes
            verbose: prints more info while running
        """

        self._prepare_sampler(NmaxP, NminP, NmaxB, NmaxB2, NmaxP4, NmaxB4, savelog)

        # Priors
        def myadd_parameter(priorsamples, p, dist, loc, scale):
            if dist == scipy.stats.uniform:
                scale = np.array(scale)-np.array(loc)
            if isinstance(loc, list):
                [priorsamples.add_parameter(p+'_z'+str(i), dist=dist(loc[i],
                                                                     scale[i]))
                 for i in range(len(loc))]
            else:
                priorsamples.add_parameter(p, dist=dist(loc, scale))

        prior_samples = Prior()

        for p in self.varied_params:
            v = self.prior_dictionary[p]

            if v['prior'] == 'uniform':
                myadd_parameter(prior_samples, p, scipy.stats.uniform,
                                v['prior_params']['min'], v['prior_params']['max'])

            elif v['prior'] == 'gaussian':
                myadd_parameter(prior_samples, p, scipy.stats.norm,
                                v['prior_params']['mean'], v['prior_params']['sigma'])

            elif v['prior'] == 'log-normal':
                myadd_parameter(prior_samples, p, np.random.lognormal,
                                v['prior_params']['mean'], v['prior_params']['sigma'])

            elif v['prior'] == 'log-uniform':
                myadd_parameter(prior_samples, p, scipy.reciprocal.rvs,
                                v['prior_params']['min'], v['prior_params']['max'])
            else:
                raise ValueError('Prior distribution not allowed for', p)

        if checkpointing:
            filepath = os.path.join(self.outfolder, self.chainfile+"_checkpoint.hdf5")
        else:
            filepath = None

        sampler = Sampler(prior_samples,
                          self.lnlike_nautilus,
                          filepath=filepath,
                          resume=resume,
                          pool=pool,
                          n_live=n_live)
        sampler.run(f_live=f_live, n_eff=n_eff, verbose=verbose,
                    discard_exploration=True)
        log_z = sampler.evidence()

        samples, log_w, log_l = sampler.posterior(return_blobs=False)

        self.Chain = {'samples': samples, 'log_weights': log_w,
                      'log_like': log_l, 'log_z': log_z}

        parhandler.add_names_to_chain(self.prior_dictionary, self.varied_params,
                                      self.Chain)

        np.savez(os.path.join(self.outfolder, self.chainfile+".npz"), **self.Chain)

#-------------------------------------------------------------------------------

    def chain_statistics(self):
        """
        Computes basic statistics from the MCMC run: means and std of the
        parameters, MAP, mean chisq, 95% cl for the chisq, deviance information
        criterion (DIC) and effective number of parameters (pV)
        """
        if self.sampler in ('emcee', 'pocomc'):
            params_mean = self.Chain.mean(axis=0)
            params_std  = self.Chain.std(axis=0)
            chi2_mean = params_mean[-1]
            imaxlike = np.argmin(self.Chain[:,-1])

        elif self.sampler == 'nautilus':
            params_mean = self.Chain['samples'].mean(axis=0)
            params_std = self.Chain['samples'].std(axis=0)
            chi2_mean = -2*np.average(self.Chain['log_like'],
                                      weights=np.exp(self.Chain['log_weights']))
            imaxlike = np.argmax(np.exp(self.Chain['log_like']))

        elif self.sampler == 'ultranest':
            params_mean = self.result['posterior']['mean']
            params_std = self.result['posterior']['stdev']
            chi2_mean = -2*self.result['weighted_samples']['logl'].mean()

        else:
            print("\033[1;31m[WARNING]: \033[00m"+\
                  "Chain stats not implemented for sampler "+self.sampler)

        print("\n# Parameter constraints (means and standard deviations)")
        for i in range(len(params_mean)):
            print(f"{self.varied_params[i]} = {params_mean[i]:.4f}  ± {params_std[i]:.4f}")

        print("\n# MAP")
        for i in range(len(params_mean)):
            if self.sampler == 'ultranest':
                print(f"# {self.varied_params[i]} = {self.result['maximum_likelihood']['point'][i]:.4f}")
            elif self.sampler == 'nautilus':
                if len(params_mean) == self.ndim:
                    print(f"{self.varied_params[i]} = {self.Chain['samples'][imaxlike,i]:.4f}")
                else:
                    print(f"Parameter {i} = {self.Chain['samples'][imaxlike,i]:.4f}")
            else:
                print(f"{self.varied_params[i]} = {self.Chain[imaxlike,i]:.4f}")

        dof = int(self.Npts * self.NSets - len(params_mean))
        print("\n# Average Chi-square")
        print(f"# chisq/dof = {chi2_mean:.2f}/{dof} = {chi2_mean/dof:.5f}")
        print(f"# 95% reference value: chisq = {scipy.stats.chi2.isf(0.05,dof):.2f}")

        if self.sampler == "emcee":
            print("\n# Deviance Information Criterion")
            Dev = -2. * self.Chain[:,-2]
            pV  = 0.5 * Dev.var()
            DIC = Dev.mean() + pV
            print(f"# pV = {pV:.2f}, Npars = {self.ndim}")
            print(f"# DIC = {DIC:.2f}")

#-------------------------------------------------------------------------------

    def set_file_names(self):
        """
        Generates filenames in which different quantities are saved.
        Must be called after _set_nmax that sets the Nmax[OBS] attributes
        """

        if self.kmax_string:
            Nmax = "kmax-"+self.kmax_string+"_"
        else:
            Nmax = "".join(
                ["Nmax"+oo+"-"+
                 str(self.nmax_dict[oo]).replace(", ","-").replace("[","").replace("]","")+
                 "_" for oo in self.Obs])
        obs = "-".join(self.Obs)
        dk = f"dk{self.dk:.1f}"

        commonstr = f"{obs}_{self.binning}_{dk}_{Nmax}{self.covtype}_{self.comment}"

        self.plotfile  = commonstr+".pdf"
        self.logfile   = commonstr+".log"
        self.chainfile = commonstr

        if self.sampler == "emcee":
            self.taufile = commonstr+".tau"

#-------------------------------------------------------------------------------

    def plot_contours(self, fiducial_dic: Optional[dict] = None):
        """
        Routine to generate a contour plot from the saved MCMC run.

        Arguments
            fiducial_dic: fiducial values of the given parameters
        """
        Ranges = {}
        for p in self.varied_params:
            if self.prior_dictionary[p]['prior'] == 'uniform':
                Ranges[p] = [self.prior_dictionary[p]['prior_params']['min'],
                             self.prior_dictionary[p]['prior_params']['max']]

        names = [p for p in self.varied_params]
        labels = [self.prior_dictionary[p]['latex'] for p in self.varied_params]

        if self.sampler == 'multinest':
            samples = self.result_PMN['samples']
            weights = None

        elif self.sampler == 'ultranest':
            samples = self.result['samples']
            weights = self.result['weights']

        elif self.sampler == 'nautilus':
            samples = self.Chain['samples']
            weights = np.exp(self.Chain['log_weights'])

        elif self.sampler == 'emcee':
            samples = self.Chain[:,:self.ndim]
            weights = None

        elif self.sampler == 'pocomc':
            samples = self.Chain['samples']
            # weights = np.exp(self.Chain['log_weights'])

        Sample = getdist.MCSamples(samples = samples,
                                   names = names,
                                   weights=weights,
                                   labels = labels,
                                   label=r'Chain',
                                   ranges=Ranges)

        g = getdist.plots.getSubplotPlotter(subplot_size=1.3)
        g.settings.axes_fontsize = 22
        g.settings.lab_fontsize = 22
        g.triangle_plot(Sample, filled=True)

        if fiducial_dic != None:
            for i in range(self.ndim):
                for j in range(i+1):
                    ax = g.subplots[i,j]
                    if i != j and self.varied_params[i] in fiducial_dic:
                        ax.axhline(fiducial_dic[self.varied_params[i]],
                                   lw=1.,color='tab:gray')
                    if self.varied_params[j] in fiducial_dic:
                        ax.axvline(fiducial_dic[self.varied_params[j]],
                                   lw=1.,color='tab:gray')

        if not os.path.exists(os.path.join(self.outfolder,"plots")):
            os.makedirs(os.path.join(self.outfolder,"plots"))
        plt.savefig(os.path.join(self.outfolder, "plots", self.plotfile),
                    bbox_inches='tight')
        plt.close()
