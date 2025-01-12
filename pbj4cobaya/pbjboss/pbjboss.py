r"""
.. module:: pbj BOSS analysis for cobay
"""
# Global
import importlib.util, sys
import numpy as np
# Local
from cobaya.log import LoggedError
from cobaya.likelihood import Likelihood

class pbjboss(Likelihood):

    def initialize(self):
        if self.pbj_path is None:  # pragma: no cover
            raise LoggedError("No path to PBJ folder provided: define pbj_path please")
        sys.path.insert(0, self.pbj_path)
        try:
            # Import the module
            self.pbj = importlib.import_module('pbjcosmo')
            self.pbj = importlib.reload(self.pbj)
        finally:
            # Remove the folder from sys.path after importing
            sys.path.pop(0)

        # Start building the PBJ init dictionary
        self.pbjobj = self.pbj.Pbj(self.pbj_Dict)

        # Initialize the boss data as provided by the files
        self.initialize_data_boss()

        # Cut the P vectors
        self.CutVectors_boss()

        # Force select the likelihood
        if hasattr(self.pbjobj, 'z_bins'):
            self.pbjobj.model_function = self.pbjobj.model_varied_cosmology_analytic_marg_multiz
            self.log.info(f"Using {self.pbjobj.model_function.__name__} as likelihood model function")
        else:
            raise LoggedError(self.log, "You must specify 'redshift_bins' for BOSS likelihood")

        # Setup for analytic marginalisation
        self.pbjobj.do_analytic_marg = self.pbj_Dict['likelihood'].get('do_analytic_marg', False)
        self.pbjobj.do_jeffreys_priors = self.pbj_Dict['likelihood'].get('do_jeffreys_priors', False)
        self.pbjobj.store_theorydict = self.pbj_Dict['likelihood'].get('store_theorydict', False)

        if self.pbjobj.do_analytic_marg or self.pbjobj.do_jeffreys_priors:
            marg_params_dict = self.pbj_Dict['marg_params']

            marg_params = list(marg_params_dict.keys())
            self.pbjobj.index_marg_param = [marg_params.index(key) for key in marg_params]

            # Then compute quantities to be added to the chi2
            self.pbjobj.prior_vector = np.array(
                [marg_params_dict[key]['mean']
                 for key in marg_params])

            if self.pbjobj.prior_vector.ndim == 1:
                prior_cov_mat = np.diag([
                    marg_params_dict[key]['sigma']**2
                    for key in marg_params])

                self.pbjobj.prior_cov_inv = np.linalg.inv(prior_cov_mat)

                self.pbjobj.prior_term_F0 = np.einsum(
                    'i,ij,j->', self.pbjobj.prior_vector, self.pbjobj.prior_cov_inv, self.pbjobj.prior_vector)
                self.pbjobj.prior_term_F1i = np.einsum('ij,j->i',
                                                self.pbjobj.prior_cov_inv, self.pbjobj.prior_vector)
            elif self.pbjobj.prior_vector.ndim == 2:
                prior_cov_mat = [np.diag([
                    marg_params_dict[key]['sigma'][i]**2
                    for key in marg_params]) for i in range(self.pbjobj.prior_vector.shape[1])]

                self.pbjobj.prior_cov_inv = np.linalg.inv(prior_cov_mat)

                self.pbjobj.prior_term_F0 = np.einsum('ik,kij,jk->k', self.pbjobj.prior_vector,
                                               self.pbjobj.prior_cov_inv, self.pbjobj.prior_vector)
                self.pbjobj.prior_term_F1i = np.einsum('kij,jk->ki',
                                                self.pbjobj.prior_cov_inv, self.pbjobj.prior_vector)

        # Now deal with the required parameters
        pbj_cosmo_pars=["h", "Obh2", "Och2", "ns", "As", "tau", "Tcmb", "z", "Mnu"]

        # For the bias parameters check whether they are analytically marginalizes, otherwise add them to the requirements
        pbj_bias_pars=['b1', 'b2', 'bG2', 'c0', 'c2', 'aP', 'e0k2', 'e2k2']
        pbj_vary_bias_pars = [p for p in pbj_bias_pars if p not in marg_params]

        self.pbj_allpars = set(pbj_cosmo_pars) | set(pbj_vary_bias_pars)
        self.pbjobj.varied_params = self.pbj_allpars
        self.pbjobj.full_param_dict = {key: 0. for key in self.pbj_allpars}
        self.pbjobj.prior_dictionary = {}
        self.renames_pbjtocob = {value: key for key, value in self.renames.items()}

        # Log recap
        self.log.info(f'Analyzing boss at redshifts {self.pbjobj.z_bins}')
        self.log.info(f'Analytically marginalizing on {marg_params}')
    

    def initialize_data_boss(self):
        """
        Specific data inizialization for boss windowless measurement file structure
        """
        self.pbjobj.Obs        = self.pbj_Dict['likelihood']['observables']
        self.log.info("Observables: "+str(self.pbjobj.Obs))

        self.pbjobj.do_preal = False; self.pbjobj.do_breal = False; self.pbjobj.do_pell = True; self.pbjobj.do_pell_bao = False; self.pbjobj.do_bell = False

        self.datafolder  = self.pbj_Dict['data_folder']
        self.datafiles  = self.pbj_Dict['data_files']
        self.pbjobj.SN  = self.pbj_Dict['SN']

        # Load the measures and contextually the k values
        self.pbjobj.DataDict = {}
        self.pbjobj.kPE = None
        for o in self.pbjobj.Obs:
            mesfile={}
            datum = []
            for f in self.datafiles:
                fcont = np.loadtxt(self.datafolder + f,unpack=True)
                if fcont.shape[0] == 4:
                    mesfile['k'], mesfile['P0'], mesfile['P2'], mesfile['P4'] = fcont

                    # Store the k values and check they are consistent with the previous
                    if self.pbjobj.kPE is None:
                        self.pbjobj.kPE = mesfile['k']
                    elif not np.array_equal(self.pbjobj.kPE, mesfile['k']):
                        raise ValueError(f"The first column of {f} is not equal to the first column of the previous files.")
                    # If fine append the data
                    datum.append(mesfile[o])
                else:
                    raise LoggedError(self.log, f"The datafile {f} has {fcont.shape[0]} and not 4 columns as expected")
            self.pbjobj.DataDict[o] = np.squeeze(np.array(datum)) 

        # Shot noise
        if isinstance(self.pbjobj.SN, list):
            if len(self.pbjobj.SN) != len(self.pbjobj.z_bins):
                raise ValueError("Number of redshift bins and SN values must match")
            self.pbjobj.Psn  = np.asarray(self.pbjobj.SN)
        else:
            self.pbjobj.Psn  = 0

        # covariance
        covfile    = self.pbj_Dict['covariance']['file']
        if isinstance(covfile, list):
            if len(covfile) != len(self.pbjobj.z_bins):
                raise ValueError("Number of redshift bins and covariance files must match")
        covlist=[]
        for f in covfile:
            covlist.append(np.loadtxt(self.datafolder + f))
        self.pbjobj.Cov  = np.array(covlist)

    def CutVectors_boss(self):
        # Some support functions

        def to_list(x, length=1):
            return [x] * length if isinstance(x, (int, float)) else x

        def create_bool_mask(k, Nmax, Nmin=None):
            if Nmin is not None:
                return [(k <= Nmax[i]) & (k >= Nmin[i]) for i in range(len(Nmax))]
            return [(k <= Nmax[i]) for i in range(len(Nmax))]
        
        # Scale cut of the fit
        if self.k_max is None: raise LoggedError(self.log, "You must declare the k_max of the fit")
        self.k_max = to_list(self.k_max)
        num_bins = len(self.k_max)

        self.pbjobj.IdxP = create_bool_mask(self.pbjobj.kPE, self.k_max)
        self.pbjobj.IdxP4 = create_bool_mask(self.pbjobj.kPE, self.k_max) # For the moment same as IdxP
        # Initialize vectors and masks
        self.pbjobj.CutDataVecs = [[] for _ in range(num_bins)]
        IdxTot = [[] for _ in range(num_bins)]
        self.pbjobj.invCov = [[] for _ in range(num_bins)]

        idx_dict = {'Pk': self.pbjobj.IdxP, 'P0': self.pbjobj.IdxP, 'P2': self.pbjobj.IdxP, 'P4': self.pbjobj.IdxP4}
        
        for i in range(num_bins):
            self.pbjobj.CutDataVecs[i] = [self.pbjobj.DataDict[obs][i][idx_dict[obs][i]]
                                    for obs in self.pbjobj.Obs]
            IdxTot[i] = np.hstack([idx_dict[obs][i] for obs in self.pbjobj.Obs])
            cut_cov = self.pbjobj.Cov[i][np.ix_(IdxTot[i].astype(bool),
                                        IdxTot[i].astype(bool))]
            self.pbjobj.invCov[i] = np.linalg.inv(cut_cov)

    def get_requirements(self):
        requirements = []
        for k in self.pbj_allpars:
            if k in self.renames.values():
                for v in self.renames:
                    if self.renames[v] == k:
                        requirements.append((v, None))
                        break
            else:
                requirements.append((k, None))
        return requirements
  
    def translate_param(self, p):
        return self.renames_pbjtocob.get(p, p)
    
    def logp(self, **_params_values):
        parvals = [_params_values[self.translate_param(k)] for k in self.pbj_allpars]
        chi2r = self.pbjobj.model_function(parvals)
        lnL = -0.5*chi2r
        return lnL
